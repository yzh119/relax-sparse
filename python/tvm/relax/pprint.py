from tvm import tir, IRModule
from tvm.driver.build_module import lower, build
from tvm.tir import ir_builder
from tvm.relax import expr as _expr
from tvm.relax.op import Op
from tvm.relax.visitors import ExprVisitor
from tvm.tir.stmt_functor import substitute

class PrettyPrinter(ExprVisitor):
    """Pretty print a Relax expression using a simple printing aglorithm."""
    def __init__(self):
        self.buffer = ""

    def emit(self, string):
        self.buffer += string

    def emit_line(self, string):
        self.buffer += string
        self.buffer += "\n"

    def visit_type(self, ty):
        if isinstance(ty, _expr.Tensor):
            self.emit("Tensor")
        else:
            raise Exception("unsupported type")


    def visit_func(self, func):
        self.emit_line(f"fn {func.name}(")
        for param in func.params:
            self.emit("    ")
            self.emit(f"{param.id.name_hint}: ")
            self.visit_type(param.ty)
            self.emit(",\n")
        self.emit(") -> ")
        self.visit_type(func.ret_type)
        self.emit(" {\n")
        self.visit(func.body)
        self.emit("}")

        import pdb; pdb.set_trace()

    def visit_let(self, let: _expr.Let) -> None:
        for binding in let.bindings:
            self.emit("let")
            self.visit(binding.var)
            self.emit(" = ")
            self.visit(binding.val)
            self.emit(";\n")

        self.visit(let.body)
        self.emit("\n")

    def visit_call(self, call: _expr.Call) -> None:
        self.visit(call.fn)
        self.emit("(")
        for i, arg in enumerate(call.args):
            if i != 0:
                self.emit(", ")
            self.visit(arg)
        self.emit(")")

    def visit_var(self, var: _expr.Var):
        self.emit(var.id.name_hint)

    def visit_global_var(self, gvar: _expr.GlobalVar):
        self.emit(gvar.id.name_hint)

    def visit_op(self, op: Op) -> None:
        self.emit(op.name)

def pretty_print(relax_program):
    pp = PrettyPrinter()
    pp.visit(relax_program)
    return pp.buffer
