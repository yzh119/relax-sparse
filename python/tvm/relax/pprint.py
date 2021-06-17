from tvm import tir, IRModule
from tvm.driver.build_module import lower, build
from tvm.tir import ir_builder
from tvm.relax import expr as _expr
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
        self.emit("}")
        self.visit(func.body)

        import pdb; pdb.set_trace()

    def visit_var(self, var):
        pass

    def visit_global_var(self, var):
        pass

def pretty_print(relax_program):
    pp = PrettyPrinter()
    pp.visit(relax_program)
    return pp.buffer
