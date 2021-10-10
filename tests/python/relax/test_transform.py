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

from __future__ import annotations  # must import to defer parsing of annotations
import tvm
from tvm import tir
from tvm import relax as rx
from tvm.ir import structural_equal
import numpy as np


def test_fma_rewrite():
    @rx.script
    class FMAMod:
        def foo(x: Tensor[(m, n), "float16"], y: Tensor[(m, n), "float16"]):
            with relax.dataflow():
                lv0 = relax.multiply(x, y)
                lv1 = relax.add(lv0, y)
                gv0 = lv1
                relax.output(gv0)
            return gv0

    mod = FMAMod()

    # before rewrite
    func = mod["foo"]
    s0 = func.body.blocks[0].bindings[1].value
    assert isinstance(s0, tvm.relay.Call)
    assert s0.op.name == "relax.add"
    input = func.params[0]
    input_shape = input.shape
    assert input_shape[0].name == "m"
    assert input_shape[1].name == "n"

    # after rewrite
    new_mod = rx.transform.fma_rewrite(mod)
    assert isinstance(new_mod, tvm.IRModule)
    assert isinstance(new_mod["foo"], tvm.relax.expr.Function)
    code = rx.parser.astext(new_mod)
    assert "relax.ewise_fma" in code

    new_func = new_mod["foo"]
    v1 = new_func.body.blocks[0].bindings[1].var
    s1 = new_func.body.blocks[0].bindings[1].value
    assert isinstance(s1, tvm.relay.Call)
    assert s1.op.name == "relax.ewise_fma"

    # the shape and type fields are auto filled during the rewriting by the Normalize function of IRBuilder
    assert structural_equal(s1.shape, input.shape)
    assert structural_equal(s1.shape, v1.shape)
    assert v1.checked_type.rank == 2
    assert v1.checked_type.dtype == "float16"

    assert type(new_func.body.blocks[0]) == rx.DataflowBlock
    assert type(new_func.body.blocks[0].bindings[2].var) == rx.Var
    assert type(new_func.body.blocks[0].bindings[2].value) == rx.DataflowVar


def test_explicit_memory_rewrite():
    @rx.script
    class CallDPSMod:
        def foo(x: Tensor[(m, n), "float32"]):
            with relax.dataflow():
                gv0 = relax.call_dps((m, n), "test.op.identity", (x,))
                relax.output(gv0)
            return gv0

    mod = CallDPSMod()
    new_mod = rx.transform.explicit_memory_lower(mod)
    assert isinstance(new_mod, tvm.IRModule)
    assert isinstance(new_mod["foo"], tvm.relax.expr.Function)
    code = rx.parser.astext(new_mod)
    assert "relax.builtin.alloc_tensor" in code
    assert "test.op.identity" in code

    new_func = new_mod["foo"]
    # the DataflowBlock changes to BindingBlock after the explicit memory rewriting
    assert type(new_func.body.blocks[0]) == rx.BindingBlock


def test_shape_lowering():
    @rx.script
    class Mod:
        def foo(x: Tensor[_, "float32"]) -> Shape:
            relax.match_shape(x.shape, (n, m))
            return (n * 2, m * 3)

    mod = Mod()
    new_mod = rx.transform.shape_lower(mod)
    assert isinstance(new_mod, tvm.IRModule)
    assert isinstance(new_mod["shape_func0"], tvm.tir.function.PrimFunc)
    assert isinstance(new_mod["foo"], tvm.relax.expr.Function)
    code = rx.parser.astext(new_mod)
    assert "alloc_shape_heap" in code
    assert "decode_shape" in code
    assert "construct_shape" in code


if __name__ == "__main__":
    test_fma_rewrite()
    test_explicit_memory_rewrite()
    test_shape_lowering()
