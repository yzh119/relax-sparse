import io

from typing import List, IO, Any, Union, Tuple, Dict
from collections import defaultdict
from pathlib import Path

from .object_def import Namespace, ObjectDefinition, ObjectType, BaseType, QName, TypeCtor
from .generator import Generator, ns_to_path, LICENSE

def translate_ns(ns: QName) -> QName:
    mapping: Dict[Tuple[str, ...], QName] = {
        ("runtime", "Array"): ["List"],
        ("runtime", "String"): ["str"],
        ("runtime", "Optional"): ["Optional"],
        ("relay", "Id"): ["relay", "base", "Id"],
    }

    key = tuple(ns)
    if key in mapping:
        return mapping[key]
    else:
        return ns

class PythonGenerator(Generator):

    def object_type_str(self, oty: Union[str, ObjectType]) -> str:
        if isinstance(oty, BaseType):
            return ".".join(translate_ns(oty.name))
        elif isinstance(oty, TypeCtor):
            ty_str = ".".join(translate_ns(oty.ty_ctor)) + "["
            for i, arg in enumerate(oty.ty_args):
                if i != 0:
                    ty_str += ","

                ty_str += self.object_type_str(arg)
            return ty_str + "]"
        elif isinstance(oty, str):
            return oty
        else:
            raise Exception(f"unsupported type {type(oty)}")

    def source_for(self, ns: Namespace) -> Path:
        ns_path = ns_to_path(ns)
        assert self.config.python_root, "Python root must be set"
        path = Path(self.config.python_root.joinpath(ns_path)).resolve()
        path.parents[0].mkdir(parents=True, exist_ok=True)
        return path.with_suffix(".py")

    def ffi_for(self, ns: Namespace) -> Path:
        ns_path = ns_to_path(ns)
        assert self.config.python_root, "Python root must be set"
        path = Path(self.config.python_root.joinpath(ns_path)).resolve()
        path.parents[0].mkdir(parents=True, exist_ok=True)
        return path.with_suffix(".py")

    def generate(self, definitions: List[ObjectDefinition]) -> None:
        by_ns = defaultdict(list)

        # Group definitions by namespaces.
        for defn in definitions:
            defn_ns = self.qualified_path(defn)
            by_ns[tuple(defn_ns)].append(defn)

        for ns_key in by_ns:
            ns = list(ns_key)
            ns = ns[:-1]
            ffi_file = self.ffi_for(list(ns) + ["_ffi_api.py"])
            api_ns = ".".join(ns)
            license_str =("\n").join([f"# {line}" for line in LICENSE.splitlines()])

            with self.open_file(ffi_file, 'w') as file:
                print(f"FFI File: {ffi_file}")
                file.seek(0)
                file.truncate()
                file.write(license_str)
                file.write("\nfrom tvm import _ffi\n")
                file.write(f"_ffi._init_api(\"{api_ns}\", __name__)\n")
                file.write("\n")

        # Generate each NS to a set of files.
        for ns_key in by_ns:
            ns = list(ns_key)
            source = io.StringIO("")

            self.generate_ns(source, ns, by_ns[ns_key])

            # Ensure directory exists.
            source_file = self.source_for(ns)
            print(f"SourceFile: {source_file}")

            license_str =("\n").join([f"# {line}" for line in LICENSE.splitlines()])

            with self.open_file(source_file, 'w') as file:
                file.seek(0)
                file.truncate()
                file.write(license_str)
                file.write("\n")
                # TODO(@jroesch): add more support here
                file.write("from typing import List, Optional, Union, Dict\n")
                file.write("import tvm._ffi\n")
                file.write("from ..ir.base import Node, Span\n")
                file.write("from ..tir import PrimExpr\n")
                file.write("from . import _ffi_api\n")
                file.write("from .. import relay")
                file.write("\n")
                file.write("ObjectRef = Node\n")
                file.write(source.getvalue())

    def generate_ns(self, source_buf: IO[Any], namespace: Namespace, defs: List[ObjectDefinition]) -> None:
        for defn in defs:
            source_buf.write(f"@tvm._ffi.register_object(\"{defn.type_key()}\")\n")
            source_buf.write(f"class {defn.ref_name()}({defn.parent_ref_name()}):\n")
            for i, field in enumerate(defn.fields):
                fty = self.object_type_str(field.field_type)
                source_buf.write(f"{4 * ' '}{field.field_name}: {fty}\n")

            source_buf.write(f"{4 * ' '}def __init__(self, ")

            for i, field in enumerate(defn.fields):
                fty = self.object_type_str(field.field_type)
                source_buf.write(f"{field.field_name}: {fty}")

                if i != len(defn.fields) - 1:
                    source_buf.write(", ")

            source_buf.write(") -> None:\n")
            source_buf.write(f"{8 * ' '}self.__init_handle_by_constructor__(_ffi_api.{defn.ref_name()}, ")

            for i, field in enumerate(defn.fields):
                source_buf.write(f" {field.field_name}")
                if i != len(defn.fields) - 1:
                    source_buf.write(", ")

            # TODO(@jroesch): the ignore here is because we need to generate pyi files for the FFI
            source_buf.write(") # type: ignore\n")
            source_buf.write("\n\n")


    def generate_gitignore(self, ns: Namespace) -> None:
        # TODO(@jroesch): unify with above code
        # TODO(@jroesch): the generated .gitignores are kind of broke
        ns_path = ns_to_path(ns)
        assert self.config.python_root, "Python root must be set"
        source_path = Path(self.config.python_root.joinpath(ns_path)).resolve()
        source_path.parents[0].mkdir(parents=True, exist_ok=True)
        source_path = source_path.parents[0]

        source_ignore_path = source_path.joinpath(".gitignore")

        with open(source_ignore_path, 'w') as source_ignore:
            for file_name in set(self.generated_files):
                file_to_ignore = file_name.relative_to(source_path)
                source_ignore.write(f"{file_to_ignore}\n")
