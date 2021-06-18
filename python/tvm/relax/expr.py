#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from typing import List, Optional, Union, Dict
import tvm._ffi
from ..ir.base import Node, Span
from ..tir import PrimExpr
from . import _ffi_api
from .. import relay
ObjectRef = Node
@tvm._ffi.register_object("relax.expr.Type")
class Type(ObjectRef):
    span: Span
    def __init__(self, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Type,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.Expr")
class Expr(ObjectRef):
    span: Span
    def __init__(self, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Expr,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.Var")
class Var(Expr):
    id: relay.base.Id
    ty: Optional[Type]
    span: Span
    def __init__(self, id: relay.base.Id, ty: Optional[Type], span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Var,  id,  ty,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.GlobalVar")
class GlobalVar(Expr):
    id: relay.base.Id
    ty: Optional[Type]
    span: Span
    def __init__(self, id: relay.base.Id, ty: Optional[Type], span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.GlobalVar,  id,  ty,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.Intrinsic")
class Intrinsic(Expr):
    name: str
    span: Span
    def __init__(self, name: str, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Intrinsic,  name,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.Binding")
class Binding(ObjectRef):
    var: Var
    val: Expr
    def __init__(self, var: Var, val: Expr) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Binding,  var,  val) # type: ignore


@tvm._ffi.register_object("relax.expr.Let")
class Let(Expr):
    bindings: List[Binding]
    body: Expr
    span: Span
    def __init__(self, bindings: List[Binding], body: Expr, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Let,  bindings,  body,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.Call")
class Call(Expr):
    fn: Expr
    args: List[Expr]
    span: Span

    def __init__(self, fn: Expr, args: List[Expr], span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Call,  fn,  args,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.Function")
class Function(Expr):
    name: Optional[str]
    params: List[Var]
    body: Expr
    ret_type: Type
    span: Span
    def __init__(self, name: Optional[str], params: List[Var], body: Expr, ret_type: Type, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Function,  name,  params,  body,  ret_type,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.BroadcastShape")
class BroadcastShape(Expr):
    lhs: Expr
    rhs: Expr
    span: Span
    def __init__(self, lhs: Expr, rhs: Expr, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.BroadcastShape,  lhs,  rhs,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.ShapeOf")
class ShapeOf(Expr):
    tensor: Expr
    span: Span
    def __init__(self, tensor: Expr, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ShapeOf,  tensor,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.TensorSlice")
class TensorSlice(Expr):
    tensor: Expr
    slice: List[Expr]
    span: Span
    def __init__(self, tensor: Expr, slice: List[Expr], span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.TensorSlice,  tensor,  slice,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.Compute")
class Compute(Expr):
    out_shape: Expr
    compute_body: Expr
    span: Span
    def __init__(self, out_shape: Expr, compute_body: Expr, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Compute,  out_shape,  compute_body,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.Add")
class Add(Expr):
    lhs: Expr
    rhs: Expr
    span: Span
    def __init__(self, lhs: Expr, rhs: Expr, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Add,  lhs,  rhs,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.TIRExpr")
class TIRExpr(Expr):
    expr: PrimExpr
    span: Span
    def __init__(self, expr: PrimExpr, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.TIRExpr,  expr,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.Tuple")
class Tuple(Expr):
    elements: List[Expr]
    span: Span
    def __init__(self, elements: List[Expr], span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Tuple,  elements,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.DataflowBlock")
class DataflowBlock(Expr):
    calls: List[Expr]
    span: Span
    def __init__(self, calls: List[Expr], span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.DataflowBlock,  calls,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.DataflowIndex")
class DataflowIndex(Expr):
    index: int
    span: Span
    def __init__(self, index: int, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.DataflowIndex,  index,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.RelayPrimFn")
class RelayPrimFn(Expr):
    elements: relay.Expr
    span: Span
    def __init__(self, elements: relay.Expr, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.RelayPrimFn,  elements,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.Dim")
class Dim(Type):
    span: Span
    def __init__(self, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Dim,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.Shape")
class Shape(Type):
    span: Span
    def __init__(self, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Shape,  span) # type: ignore


@tvm._ffi.register_object("relax.expr.Tensor")
class Tensor(Type):
    shape: Optional[Expr]
    dtype: Optional[Expr]
    span: Span
    def __init__(self, shape: Optional[Expr], dtype: Optional[Expr], span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Tensor,  shape,  dtype,  span) # type: ignore
