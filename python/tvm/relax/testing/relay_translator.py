# # Licensed to the Apache Software Foundation (ASF) under one
# # or more contributor license agreements.  See the NOTICE file
# # distributed with this work for additional information
# # regarding copyright ownership.  The ASF licenses this file
# # to you under the Apache License, Version 2.0 (the
# # "License"); you may not use this file except in compliance
# # with the License.  You may obtain a copy of the License at
# #
# #   http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing,
# # software distributed under the License is distributed on an
# # "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# # KIND, either express or implied.  See the License for the
# # specific language governing permissions and limitations
# # under the License.
# """Relay to Relax translator."""

# from __future__ import annotations
# from typing import Dict
# import tvm
# from tvm.relay import Call, TupleGetItem
# from tvm.relax.testing.topi import *
# from tvm import relax, relay, topi, te
# from tvm.relax.testing import nn
# from tvm.relay.op.transform import broadcast_to, unique


# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-argument, invalid-name, no-else-return
"""Relay to Relax translator."""

from __future__ import annotations
from typing import Dict
import tvm
from tvm.ir.module import IRModule
from tvm.relay import Call, TupleGetItem
from tvm.relax.testing.topi import mean, variance, reshape, reverse_reshape, bias_add, collapse_sum
from tvm import relax, relay, topi, te
from tvm.relax.testing import nn
from tvm.relax.expr import te_tensor
import os
from tvm.script import relax as R, tir as T

# load a relay program in text format to an IRModule
def load_text(file_path: str) -> tvm.IRModule:
    if os.path.isfile(file_path):
        text_file = open(file_path, "r")
        data = text_file.read()
        text_file.close()
        mod = tvm.parser.fromtext(data)
        return mod
    else:
        raise RuntimeError(f"File at path {file_path} does not exist")


class RelayOpConverter(object):
    """A helper class for holding Relay op converters."""

    @classmethod
    def get_converter(cls):
        """Get converter.

        :return: converter, which should be `_impl`.
        """

        if hasattr(cls, "_impl"):
            return getattr(cls, "_impl")
        raise tvm.error.OpNotImplemented("Operator {} is not supported.".format(cls.__name__))


# class Unary(RelayOpConverter):
#     """A helper class for unary op converters."""

#     name = ""

#     @classmethod
#     def _impl(cls, inputs, attrs):
#         assert len(inputs) == 1, "Unary op takes 1 inputs, but {} given".format(len(inputs))
#         op_name = cls.name
#         topi_func = getattr(topi, op_name)
#         return nn.emit_te(topi_func, *inputs)


# class Elemwise(RelayOpConverter):
#     """A helper class for elemwise op converters."""

#     name = ""

#     @classmethod
#     def _impl(cls, inputs, attrs):
#         assert len(inputs) == 2, "Elemwise op takes 2 inputs, but {} given".format(len(inputs))
#         op_name = cls.name
#         topi_func = getattr(topi, op_name)
#         return nn.emit_te(topi_func, *inputs)


# class Add(Elemwise):
#     """Operator converter for add."""

#     name = "add"


# class Subtract(Elemwise):
#     """Operator converter for subtract."""

#     name = "subtract"


# class Divide(Elemwise):
#     """Operator converter for divide."""

#     name = "divide"


# class Multiply(Elemwise):
#     """Operator converter for multiply."""

#     name = "multiply"


# class Power(Elemwise):
#     """Operator converter for power."""

#     name = "power"


# class Sqrt(Unary):
#     """Operator converter for sqrt."""

#     name = "sqrt"


# class Exp(Unary):
#     """Operator converter for exp."""

#     name = "exp"


# class Negative(Unary):
#     """Operator converter for negative."""

#     name = "negative"


# class Erf(Unary):
#     """Operator converter for erf."""

#     name = "erf"


class Dense(RelayOpConverter):
    """Operator converter for dense."""

    @classmethod
    def _impl(cls, inputs, attrs):
        return nn.emit_te(topi.nn.dense, *inputs)


class Softmax(RelayOpConverter):
    """Operator converter for softmax."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = AttrCvt(attrs)
        return nn.emit_te(topi.nn.softmax, *inputs, **new_attrs)


def AttrCvt(attrs) -> Dict:
    """Convert attributes to a dict."""
    attrs_dict = {}

    for k in attrs.keys():
        attrs_dict[k] = attrs[k]

    return attrs_dict


class Conv2D(RelayOpConverter):
    """Operator converter for conv2d."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_inputs = [*inputs]
        if attrs is not None:
            new_inputs.append(attrs["strides"])
            new_inputs.append(attrs["padding"])
            new_inputs.append(attrs["dilation"])
        else:
            raise RuntimeError("attrs must be provided to conv2d op.")
        return nn.emit_te(topi.nn.conv2d_nchw, *new_inputs)


class BatchMatmul(RelayOpConverter):
    """Operator converter for nn.batch_matmul."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = attr_convert(attrs)
        if "out_dtype" in new_attrs:
            new_attrs["out_dtype"] = None
        if "transpose_a" in new_attrs:
            new_attrs["transpose_a"] = bool(new_attrs["transpose_a"])
        if "transpose_b" in new_attrs:
            new_attrs["transpose_b"] = bool(new_attrs["transpose_b"])
        return nn.emit_te(topi.nn.batch_matmul, *inputs, **new_attrs)


class Embedding(RelayOpConverter):
    """Operator converter for nn.embedding."""

    @classmethod
    def _impl(cls, inputs, attrs):
        def embedding(table, indices):
            oshape = list(indices.shape) + [table.shape[1]]
            return te.compute(oshape, lambda *i: table(indices(*i[:-1]), i[-1]), name="embedding")

        return nn.emit_te(embedding, *inputs)


@T.prim_func
def embedding_grad(table: T.handle, indices: T.handle, grad_in: T.handle, grad_out: T.handle):
    T.func_attr({"global_symbol": "embedding_grad"})
    m = T.var("int32")
    n = T.var("int32")
    k = T.var("int32")
    A = T.match_buffer(table, (m, n))
    B = T.match_buffer(indices, (k), "int32")
    C = T.match_buffer(grad_in, (k, n))
    D = T.match_buffer(grad_out, (m, n))

    for i in range(m):
        for j in range(n):
            D[i, j] = 0.0
    for i in range(k):
        for j in range(n):
            D[B[i], j] += C[i, j]


class EmbeddingGrad(RelayOpConverter):
    """Operator converter for nn.embedding_grad."""

    @classmethod
    def _impl(cls, inputs, attrs):
        tir_func = embedding_grad
        func_name = relax.BlockBuilder.current().get_unique_name(tir_func.__name__)
        gvar = relax.BlockBuilder.current().add_func(tir_func, func_name)
        output_shape = inputs[0].shape
        call = relax.call_tir(output_shape, gvar, inputs)
        return relax.BlockBuilder.current().emit(call)


# class BatchFlatten(RelayOpConverter):
#     """Operator converter for batch_flatten."""

#     @classmethod
#     def _impl(cls, inputs, attrs):
#         return nn.emit_te(topi.nn.flatten, inputs[0])


# convert_map defines maps of name to converter functor(callable)
# use attr_convert if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping (fusion), write custom topi func

# Minimal set of ops for transformer
def get_convert_map():
    return {
        # "add": Add.get_converter(),
        # "subtract": Subtract.get_converter(),
        # "divide": Divide.get_converter(),
        # "multiply": Multiply.get_converter(),
        # "power": Power.get_converter(),
        # "sqrt": Sqrt.get_converter(),
        # "exp": Exp.get_converter(),
        # "erf": Erf.get_converter(),
        # "negative": Negative.get_converter(),
        # "reshape": Reshape.get_converter(),
        "nn.dense": Dense.get_converter(),
        # "nn.batch_norm": BatchNorm.get_converter(),
        "nn.conv2d": Conv2D.get_converter(),
        # "nn.relu": Relu.get_converter(),
        "nn.batch_matmul": BatchMatmul.get_converter(),
        # "zeros": Zeros.get_converter(),
        # "mean": Mean.get_converter(),
        # "variance": Variance.get_converter(),
        # "contrib_reverse_reshape": ReverseReshape.get_converter(),
        # "nn.bias_add": BiasAdd.get_converter(),
        # "transpose": Transpose.get_converter(),
        # "expand_dims": ExpandDims.get_converter(),
        # "cast": Cast.get_converter(),
        # "broadcast_to": BroadcastTo.get_converter(),
        # "nn.log_softmax": LogSoftmax.get_converter(),
        "nn.softmax": Softmax.get_converter(),
        # "one_hot": Onehot.get_converter(),
        # "sum": Sum.get_converter(),
        # "not_equal": NotEqual.get_converter(),
        # "collapse_sum_to": CollapseSumTo.get_converter(),
        # "cast_like": CastLike.get_converter(),
        # "squeeze": Squeeze.get_converter(),
        # "nn.max_pool2d": MaxPool2D.get_converter(),
        # "nn.global_avg_pool2d": GlobalAvgPool2D.get_converter(),
        # "nn.batch_flatten": BatchFlatten.get_converter(),
        "nn.embedding_grad": EmbeddingGrad.get_converter(),
        "nn.embedding": Embedding.get_converter(),
    }


def convert_operator(op_type, inputs, attrs=None):
    """Convert from Relay operator to Relax operator/topi function.
    The converter must specify conversions explicitly for incompatible name, and
    apply handlers to operator attributes.

    Parameters
    ----------
    op_type : str
        Operator name, such as Convolution, FullyConnected
    inputs : list of Expr
        List of input inputs.
    attrs : dict
        Dict of operator attributes

    Returns
    -------
    func : tvm.relay.function.Function
        Converted relay function
    """
    convert_map = get_convert_map()
    if op_type in convert_map:
        func = convert_map[op_type](inputs, attrs)
    else:
        raise tvm.error.OpNotImplemented("Operator {} is not supported.".format(op_type))
    return func


def attr_convert(attrs) -> Dict:
    """Convert attributes to a dict."""
    attrs_dict = {}

    for k in attrs.keys():
        attrs_dict[k] = attrs[k]

    return attrs_dict


def from_relay(func: relay.Function):
    """Convert a Relay model into an equivalent Relax Function.
    Parameters
    ----------
    func : relay.Function
        Relay function to be converted
    Returns
    -------
    mod : tvm.IRModule
        The Relax IRModule for compilation
    """
    var_map = {}
    # old tuple -> new tuple
    tuple_map = {}
    last_var = None
    params = []
    convert_map = get_convert_map()

    def visit_func(node):
        nonlocal last_var
        if isinstance(node, relay.Var):
            var_map[node] = nn.Placeholder(
                tuple(node.type_annotation.shape), node.type_annotation.dtype, node.name_hint
            )
            params.append(var_map[node])
        elif isinstance(node, relay.Call):
            args = node.args
            new_args = []
            for arg in args:
                if arg in var_map:
                    new_args.append(var_map[arg])
                else:
                    new_args.append(arg)

            op_name = node.op.name

            attrs = node.attrs
            compute_func = node.op.get_attr("FTVMCompute")

            if compute_func is None:
                if node.op.name not in convert_map:
                    raise tvm.error.OpNotImplemented(
                        "Operator {} is not supported.".format(op_name)
                    )
                else:
                    var = convert_operator(op_name, new_args, attrs)
            else:
                var = bb.emit_te(compute_func, attrs, new_args, node.checked_type)

            # var = convert_operator(op_name, new_args, attrs)
            last_var = var
            var_map[node] = var
        elif isinstance(node, relay.Constant):
            new_constant = relay.Constant(node.data)
            var_map[node] = new_constant
        elif isinstance(node, relay.Tuple):
            new_fields = []
            for field in node.fields:
                if field in var_map:
                    new_fields.append(var_map[field])
                else:
                    raise RuntimeError("field is not in var_map")
            new_tuple = relax.Tuple(new_fields)
            tuple_map[node] = new_tuple
            new_tuple_var = relax.BlockBuilder.current().emit(new_tuple)
            var_map[node] = new_tuple_var
            last_var = new_tuple_var
        elif isinstance(node, relay.TupleGetItem):
            if node.tuple_value in var_map:
                new_tuple = tuple_map[node.tuple_value]
                new_tuple_get_item_node = TupleGetItem(new_tuple, node.index)
                new_tuple_get_item_var = relax.BlockBuilder.current().emit(new_tuple_get_item_node)
                var_map[node] = new_tuple_get_item_var
                last_var = new_tuple_get_item_var
            else:
                raise RuntimeError("tuple is not in var_map")
        elif isinstance(node, relay.Function):
            relax.BlockBuilder.current().emit_func_output(last_var, params)

    bb = relax.BlockBuilder()
    with bb.function("main"):
        relay.analysis.post_order_visit(func, visit_func)

    return bb.get()


# def from_relay(func: relay.Function) -> IRModule:
#     """Convert a Relay function into a Relax program.

#     Parameters
#     ----------
#     func : relay.Function
#         Relay function to be converted

#     Returns
#     -------
#     mod : tvm.IRModule
#         The Relax IRModule for compilation
#     """
#     # A map to store the mapping of Relay Expr to its corresponding Relax var
#     var_map = {}
#     tuple_map = {}
#     # The output of the function
#     output_var = None
#     params = []
#     convert_map = get_convert_map()

#     def visit_func(node):
#         nonlocal output_var
#         if isinstance(node, relay.Var):
#             if isinstance(node.type_annotation, relay.TensorType):
#                 var_map[node] = nn.Placeholder(
#                     tuple(node.type_annotation.shape), node.type_annotation.dtype, node.name_hint
#                 )
#                 params.append(var_map[node])
#             else:
#                 raise TypeError("The type of relay.Var to be translated must be of TensorType.")
#         elif isinstance(node, relay.Call):
#             args = node.args
#             new_args = []
#             for arg in args:
#                 if arg in var_map:
#                     new_args.append(var_map[arg])

#             op_name = node.op.name

#             attrs = node.attrs
#             compute_func = node.op.get_attr("FTVMCompute")
#             print(op_name, compute_func)
#             # if compute_func is None or "reshape" in op_name:
#             #     if node.op.name not in convert_map:
#             #         raise tvm.error.OpNotImplemented(
#             #             "Operator {} is not supported.".format(op_name)
#             #         )
#             #     else:
#             var = convert_operator(op_name, new_args, attrs)
#             # else:
#             #     var = bb.emit_te(compute_func, attrs, new_args, node.checked_type)

#             output_var = var
#             var_map[node] = var
#         elif isinstance(node, relay.Constant):
#             new_constant = relax.expr.Constant(node.data)
#             var_map[node] = new_constant
#         elif isinstance(node, relay.Tuple):
#             new_fields = []
#             for field in node.fields:
#                 if field in var_map:
#                     new_fields.append(var_map[field])
#                 else:
#                     raise RuntimeError("field is not in var_map.")
#             new_tuple = relax.Tuple(new_fields)
#             new_tuple_var = relax.BlockBuilder.current().emit(new_tuple)
#             var_map[node] = new_tuple_var
#             output_var = new_tuple_var
#         elif isinstance(node, relay.Tuple):
#             new_fields = []
#             for field in node.fields:
#                 if field in var_map:
#                     new_fields.append(var_map[field])
#                 else:
#                     raise RuntimeError("field is not in var_map")
#             new_tuple = relax.Tuple(new_fields)
#             tuple_map[node] = new_tuple
#             new_tuple_var = relax.BlockBuilder.current().emit(new_tuple)
#             var_map[node] = new_tuple_var
#             output_var = new_tuple_var
#         elif isinstance(node, relay.TupleGetItem):
#             if node.tuple_value in var_map:
#                 new_tuple = tuple_map[node.tuple_value]
#                 new_tuple_get_item_node = TupleGetItem(new_tuple, node.index)
#                 new_tuple_get_item_var = relax.BlockBuilder.current().emit(new_tuple_get_item_node)
#                 var_map[node] = new_tuple_get_item_var
#                 output_var = new_tuple_get_item_var
#             else:
#                 raise RuntimeError("tuple is not in var_map")
#         elif isinstance(node, relay.Function):
#             relax.BlockBuilder.current().emit_func_output(output_var, params)
#         elif isinstance(node, tvm.ir.Op):
#             pass
#         else:
#             raise TypeError("{} is not supported yet.".format(str(type(node))))

#     bb = relax.BlockBuilder()
#     with bb.function("main"):
#         relay.analysis.post_order_visit(func, visit_func)

#     return bb.get()


if __name__ == "__main__":
    RELAY_MODEL = """
    #[version = "0.0.5"]
    def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
        %0 = add(%a, %b);
        %1 = add(%c, %d);
        subtract(%0, %1)
    }
    """

    mod = tvm.parser.fromtext(RELAY_MODEL)

    mod = load_text("bert_16_128.txt")

    mod = from_relay(mod["main"])
    from tvm.script import relax as R

    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
