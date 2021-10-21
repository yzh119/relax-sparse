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
import pytest
import tvm
from tvm import relax as rx
from tvm import tir, relay
from tvm.ir import structural_equal


def test_op_infer():
    @rx.script
    class TestOpInfer:
        def f(x: Tensor[(n, m), "float32"], y: Tensor[(b, n, m), "float32"]):
            with relax.dataflow():
                z = relax.add(x, y)
                relax.output(z)
            w = relax.add(z, z)
            return w

    mod = TestOpInfer()
    mod = rx.transform.type_inference(mod)

    f = mod["f"]
    z_bind = f.body.blocks[0].bindings[0]
    w_bind = f.body.blocks[1].bindings[0]

    ty = rx.DynTensorType(3, "float32")
    assert structural_equal(z_bind.var.checked_type, ty)
    assert structural_equal(z_bind.value.checked_type, ty)
    assert structural_equal(w_bind.var.checked_type, ty)
    assert structural_equal(w_bind.value.checked_type, ty)
    assert structural_equal(f.body.checked_type, ty)
    assert structural_equal(f.checked_type.ret_type, ty)


def test_tuple():
    @rx.script
    class TestTuple:
        def f(x: Tensor[(n, m), "float32"], y: Tensor[(b, n, m), "float32"]):
            z = (x, y)
            return z

    mod = TestTuple()
    mod = rx.transform.type_inference(mod)

    f = mod["f"]

    z_bind = f.body.blocks[0].bindings[0]

    ty = relay.TupleType([rx.DynTensorType(2, "float32"), rx.DynTensorType(3, "float32")])
    assert structural_equal(z_bind.var.checked_type, ty)
    assert structural_equal(z_bind.value.checked_type, ty)
    assert structural_equal(f.body.checked_type, ty)
    assert structural_equal(f.checked_type.ret_type, ty)

    rx.parser.pretty_print(mod)


def test_if():
    @rx.script
    class TestIf:
        def f(x: Tensor[(n, m), "float32"], y: Tensor[(n, m), "float32"], cond: Tensor[(), "bool"]):
            if cond:
                z = x
            else:
                z = y
            return z

    mod = TestIf()
    mod = rx.transform.type_inference(mod)

    f = mod["f"]
    z_bind = f.body.blocks[0].bindings[0]

    ty = rx.DynTensorType(2, "float32")
    assert structural_equal(z_bind.var.checked_type, ty)
    assert structural_equal(z_bind.value.checked_type, ty)


def test_if_tensor_generalize():
    # TODO(@altanh): do we want to support this kind of dtype generalization?

    @rx.script
    class TestIfTensorGeneralize:
        def f(x: Tensor[(n, m), "float32"], y: Tensor[(b, n, m), "int32"], cond: Tensor[(), "bool"]):
            if cond:
                z = x
            else:
                z = y
            return z

    mod = TestIfTensorGeneralize()
    mod = rx.transform.type_inference(mod)

    f = mod["f"]
    z_bind = f.body.blocks[0].bindings[0]

    ty = rx.DynTensorType(-1, "")
    assert structural_equal(z_bind.var.checked_type, ty)
    assert structural_equal(z_bind.value.checked_type, ty)


# TODO(@altanh): test_tuple_projection once parser supports projection


def test_match_shape():
    @rx.script
    class TestMatchShape:
        def f(x: Tensor[_, "float32"], y: Tensor[(n, m), "float32"]):
            with relax.dataflow():
                x_refined = relax.match_shape(x, (n, m))
                z0 = relax.add(x, y)
                z1 = relax.add(x_refined, y)
                sh = relax.match_shape(z0.shape, (a, b))
                relax.output(z0, z1, sh)
            tup = (z0, z1, sh)
            return tup


    mod = TestMatchShape()
    mod = rx.transform.type_inference(mod)

    f = mod["f"]
    ret_ty = relay.TupleType([rx.DynTensorType(-1, "float32"), rx.DynTensorType(2, "float32"), rx.ShapeType()])
    assert structural_equal(f.checked_type.ret_type, ret_ty)


def test_multiple_globals():
    @rx.script
    class TestMultipleGlobals:
        def f(x: Tensor[_, "float32"]):
            x_refined = relax.match_shape(x, (n, m))
            return x_refined

        def g(y: Tensor[_, "float32"]):
            # TODO(@altanh):
            # y_refined = f(y)
            f2 = f
            return f2

    mod = TestMultipleGlobals()
    mod = rx.transform.type_inference(mod)

    # TODO(@altanh): function type pretty print, maybe "Function[(Args...), RetType]"?

    g = mod["g"]
    ret_ty = relay.FuncType([rx.DynTensorType(-1, "float32")], rx.DynTensorType(2, "float32"))
    assert structural_equal(g.checked_type.ret_type, ret_ty)


def test_extern_annotation():
    @rx.script
    class TestExternAnnotation:
        def f(x: Tensor[_, "float32"]):
            y: Tensor[(n, m), "float32"] = relax.call_packed("my_op", x)
            z = relax.add(y, y)
            w: Tensor[(2, n, m), "float32"] = relax.call_dps((2, n, m), "my_dps_op", (y, z))
            return (y, z, w)

    mod = TestExternAnnotation()
    mod = rx.transform.type_inference(mod)
    print(rx.parser.astext(mod))


@pytest.mark.xfail
def test_if_mismatch_fail():
    @rx.script
    class TestIfMismatchFail:
        def f(x: Tensor[(3, 4), "float32"], cond: Tensor[(), "bool"]):
            sh = x.shape
            if cond:
                r = sh
            else:
                r = x
            return r

    mod = TestIfMismatchFail()
    mod = rx.transform.type_inference(mod)


@pytest.mark.xfail
def test_if_cond_fail():
    @rx.script
    class TestIfCondFail:
        def f(x: Tensor[(3, 4), "float32"], cond: Tensor[(), "float32"]):
            if cond:
                r = x
            else:
                r = x
            return r

    mod = TestIfCondFail()
    mod = rx.transform.type_inference(mod)


@pytest.mark.xfail
def test_if_cond_fail2():
    @rx.script
    class TestIfCondFail2:
        def f(x: Tensor[(3, 4), "float32"], cond: Tensor[(2,), "bool"]):
            if cond:
                r = x
            else:
                r = x
            return r

    mod = TestIfCondFail2()
    mod = rx.transform.type_inference(mod)


@pytest.mark.xfail
def test_extern_annotation_missing_fail():
    @rx.script
    class TestExternAnnotationMissingFail:
        def f(x: Tensor[(n, m), "float32"]):
            y = relax.call_packed("my_op", x)
            z = relax.add(y, y)
            w: Tensor[(2, n, m), "float32"] = relax.call_dps((2, n, m), "my_dps_op", (y, z))
            return (y, z, w)

    mod = TestExternAnnotationMissingFail()
    mod = rx.transform.type_inference(mod)
