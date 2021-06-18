from typing import Optional, Union, List
import attr

import tvm
from tvm import tir, IRModule
from tvm.driver.build_module import lower, build
from tvm.tir import ir_builder
from tvm.relax import expr as _expr
from tvm.relax.op import Op
from tvm.tir.stmt_functor import substitute

class ExprVisitor:
    """Visit a Relax expression."""
    def visit(self, expr: _expr.Expr):
        if isinstance(expr, _expr.Var):
            return self.visit_var(expr)
        elif isinstance(expr, _expr.GlobalVar):
            return self.visit_global_var(expr)
        elif isinstance(expr, _expr.Let):
            return self.visit_let(expr)
        elif isinstance(expr, _expr.Call):
            return self.visit_call(expr)
        elif isinstance(expr, _expr.Function):
            return self.visit_func(expr)
        elif isinstance(expr, _expr.BroadcastShape):
            return self.visit_bs(expr)
        elif isinstance(expr, _expr.ShapeOf):
            return self.visit_shape_of(expr)
        elif isinstance(expr, _expr.TensorSlice):
            return self.visit_tensor_slice(expr)
        elif isinstance(expr, _expr.Compute):
            return self.visit_compute(expr)
        elif isinstance(expr, _expr.Tuple):
            return self.visit_tuple(expr)
        elif isinstance(expr, Op):
            return self.visit_op(expr)
        else:
            raise Exception(f"unsupported type {type(expr)}")

    def visit_var(self, var: _expr.Var) -> None:
        pass

    def visit_global_var(self, var: _expr.GlobalVar) -> None:
        pass

    def visit_let(self, let: _expr.Let) -> None:
        for binding in let.bindings:
            self.visit(binding.var)
            self.visit(binding.val)

        self.visit(let.body)

    def visit_call(self, call: _expr.Call) -> None:
        for arg in call.args:
            self.visit(arg)
        self.visit(call.fn)

    def visit_op(self, op: Op) -> None:
        pass

class ExprTransformer:
    """Immutably transform one Relax expression into another."""
    def visit(self, expr):
        if isinstance(expr, _expr.Var):
            return self.visit_var(expr)
        elif isinstance(expr, _expr.GlobalVar):
            return self.visit_global_var(expr)
        elif isinstance(expr, _expr.Let):
            return self.visit_let(expr)
        elif isinstance(expr, _expr.Call):
            return self.visit_call(expr)
        elif isinstance(expr, _expr.Function):
            return self.visit_func(expr)
        elif isinstance(expr, _expr.BroadcastShape):
            return self.visit_bs(expr)
        elif isinstance(expr, _expr.ShapeOf):
            return self.visit_shape_of(expr)
        elif isinstance(expr, _expr.TensorSlice):
            return self.visit_tensor_slice(expr)
        elif isinstance(expr, _expr.Compute):
            return self.visit_compute(expr)
        elif isinstance(expr, _expr.Tuple):
            return self.visit_tuple(expr)
        else:
            assert False

    def visit_var(self, var):
        return var

    def visit_global_var(self, var):
        return var

    def visit_let(self, let):
        assert False
