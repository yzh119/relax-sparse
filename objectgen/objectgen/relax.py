from pathlib import Path
import objectgen
from typing import List
# TODO(@jroesch): move relax out of objectgen?
from . import *

def main() -> None:
    config = ObjectGenConfig(
        python_root = Path("./python/"),
        cpp_include_root = Path("./include"),
        cpp_source_root = Path("./src/"),
        root_namespace = ["tvm"])

    RegName = "tvm::runtime::vm::RegName"
    Index = "tvm::runtime::vm::Index"

    relax_expr_imports = [
        ["tvm", "ir", "span"],
        ["tvm", "node", "node"],
        ["tvm", "runtime", "container", "array"],
        ["tvm", "runtime", "container", "map"],
        ["tvm", "runtime", "object"],
        ["tvm", "relay", "expr"],
        ["tvm", "ir", "expr"],
        ["tvm", "tir", "expr"]
    ]

    relax_expr: List[ObjectDefinition] = in_ns(["relax", "expr"], relax_expr_imports, [
        ObjectDefinition(
            name="Type",
            fields=[
                ObjectField("span", "Span", use_in_sequal_reduce=False)
            ],
            final = False,
        ),
        ObjectDefinition(
            name="Expr",
            fields=[
                ObjectField("span", "Span", use_in_sequal_reduce=False)
            ],
            final = False,
        ),
        ObjectDefinition(
            name="Var",
            inherits_from="Expr",
            fields=[
                ObjectField("id", Id),
                ObjectField("ty", Option("Type")),
            ],
        ),
        ObjectDefinition(
            name="GlobalVar",
            inherits_from="Expr",
            fields=[
                ObjectField("id", Id),
                ObjectField("ty", Option("Type")),
            ],
        ),
        ObjectDefinition(
            name="Intrinsic",
            inherits_from="Expr",
            fields=[
                ObjectField("name", String),
            ]
        ),
        ObjectDefinition(
            name="Binding",
            fields=[
                ObjectField("var", "Var"),
                ObjectField("val", "Expr"),
            ]
        ),
        ObjectDefinition(
            name="Let",
            inherits_from="Expr",
            fields=[
                ObjectField("bindings", Array("Binding")),
                ObjectField("body", "Expr"),
            ]
        ),
        ObjectDefinition(
            name="Call",
            inherits_from="Expr",
            fields=[
                ObjectField("fn", "Expr"),
                ObjectField("args", Array("Expr")),
            ]
        ),
        ObjectDefinition(
            name="Function",
            inherits_from="Expr",
            fields=[
                ObjectField("name", Option(String), use_in_sequal_reduce=False),
                ObjectField("params", Array("Var")),
                ObjectField("body", "Expr"),
                ObjectField("ret_type", "Type"),
            ]
        ),
        ObjectDefinition(
            name="BroadcastShape",
            inherits_from="Expr",
            fields=[
                ObjectField("lhs", "Expr"),
                ObjectField("rhs", "Expr"),
            ]
        ),
        ObjectDefinition(
            name="ShapeOf",
            inherits_from="Expr",
            fields=[
                ObjectField("tensor", "Expr"),
            ]
        ),
        ObjectDefinition(
            name="TensorSlice",
            inherits_from="Expr",
            fields=[
                ObjectField("tensor", "Expr"),
                ObjectField("slice", Array("Expr"))
            ]
        ),
        ObjectDefinition(
            name="Compute",
            inherits_from="Expr",
            fields=[
                ObjectField("out_shape", "Expr"),
                ObjectField("compute_body", "Expr")
            ]
        ),
        ObjectDefinition(
            name="Add",
            inherits_from="Expr",
            fields=[
                ObjectField("lhs", "Expr"),
                ObjectField("rhs", "Expr")
            ]
        ),
        ObjectDefinition(
            name="TIRExpr",
            inherits_from="Expr",
            fields=[
                ObjectField("expr", "PrimExpr")
            ]
        ),
        ObjectDefinition(
            name="Tuple",
            inherits_from="Expr",
            fields=[
                ObjectField("elements", Array("Expr"))
            ]
        ),
        ObjectDefinition(
            name="DataflowBlock",
            inherits_from="Expr",
            fields=[
                ObjectField("calls", Array("Expr")),
            ]
        ),
        ObjectDefinition(
            name="DataflowIndex",
            inherits_from="Expr",
            fields=[
                ObjectField("index", "int")
            ]
        ),
        ObjectDefinition(
            name="RelayPrimFn",
            inherits_from="Expr",
            fields=[
                ObjectField("elements", BaseType(["relay", "Expr"])),
            ]
        ),
        ObjectDefinition(
            name="Dim",
            inherits_from="Type",
            fields=[],
        ),
        ObjectDefinition(
            name="Shape",
            inherits_from="Type",
            fields=[],
        ),
        ObjectDefinition(
            name="Tensor",
            inherits_from="Type",
            fields=[
                ObjectField("shape", Option("Expr")),
                ObjectField("dtype", Option("Expr"))
            ]
        ),
    ])

    relax_vm_imports = [
        ["tvm", "ir", "span"],
        ["tvm", "node", "node"],
        ["tvm", "runtime", "container", "array"],
        ["tvm", "runtime", "container", "map"],
        ["tvm", "runtime", "object"],
        ["tvm", "relay", "expr"],
        ["tvm", "ir", "expr"],
        ["tvm", "tir", "expr"],
        ["tvm", "runtime", "vm", "vm"],
        ["tvm", "runtime", "vm", "bytecode"],
    ]

    relax_vm = in_ns(["relax", "vm"], relax_vm_imports, [
        ObjectDefinition(
            name="Instruction",
            fields=[
                ObjectField("span", "Span", use_in_sequal_reduce=False)
            ],
            final = False,
        ),
        ObjectDefinition(
            name="AllocTensor",
            inherits_from="Instruction",
            fields=[
                ObjectField("storage", RegName),
                ObjectField("offset", Index),
                ObjectField("shape_register", RegName),
                # ObjectField("dtype", "DLDataType")
            ]
        ),
    ])

    objectgen.from_python(config, relax_expr + relax_vm)

if __name__ == "__main__":
    main()
