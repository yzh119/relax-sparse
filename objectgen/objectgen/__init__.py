import attr
import io
from pathlib import Path
from typing import List, Optional, Union
from collections import defaultdict
from .cpp_generator import CPPGenerator
from .python_generator import PythonGenerator
from .generator import ObjectGenConfig
from .object_def import BaseType, ObjectDefinition, ObjectField, ObjectMethod, ObjectType, TypeCtor

def resolve_parent_fields(definitions):
    parent_map = {}
    for defn in definitions:
        parent_map[defn.name] = defn

    for defn in definitions:
        if defn.inherits_from != "ObjectRef":
            parent_fields = parent_map[defn.inherits_from].fields
            defn.fields = defn.fields + parent_fields

def from_python(config, definitions):
    resolve_parent_fields(definitions)

    if config.cpp_include_root and config.cpp_source_root:
        cpp_gen = CPPGenerator(config)
        cpp_gen.generate(definitions)

    if config.python_root:
        py_gen = PythonGenerator(config)
        py_gen.generate(definitions)

def in_ns(ns, imports, defs):
    for defn in defs:
        defn.namespace = ns + defn.namespace
        defn.imports = imports + defn.imports

    return defs

# TODO(@jroesch): TVM type helpers, move?

def Array(elem_ty: Union[str, ObjectType]) -> ObjectType:
    """Construct a type corresponding to tvm::runtime::Array."""
    if isinstance(elem_ty, str):
        normalized_elem_ty: ObjectType = BaseType([elem_ty])
    else:
        normalized_elem_ty = elem_ty

    return TypeCtor(["runtime", "Array"], [normalized_elem_ty])

def String() -> ObjectType:
    """Construct a type corresponding to tvm::runtime::String."""
    return BaseType(["runtime", "String"])
