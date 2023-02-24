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
import tvm
import tvm.testing
import pytest

from tvm import relax
from tvm import TVMError
from tvm.relax.sparse.axis import dense_fixed, dense_variable, dense_padded
from tvm.relax.sparse import TensorStructInfo
from tvm.script import relax as R, tir as T


def _check_equal(x, y, map_free_vars=False):
    tvm.ir.assert_structural_equal(x, y, map_free_vars)
    tvm.ir.assert_structural_equal(y, x, map_free_vars)

    xhash = tvm.ir.structural_hash(x, map_free_vars)
    yhash = tvm.ir.structural_hash(y, map_free_vars)

    assert xhash == yhash


def _check_json_roundtrip(x):
    xret = tvm.ir.load_json(tvm.ir.save_json(x))
    _check_equal(x, xret, map_free_vars=True)
    return xret


def test_sparse_tensor_struct_info():
    b = T.var("int64")
    nnz1 = T.var("int64")
    indptr = relax.Var("indptr", R.Tensor((b + 1,), "int64"))
    indptr1 = relax.Var("indptr1", R.Tensor((nnz1 + 1,), "int64"))
    df = dense_fixed(b)
    dv = dense_variable(df, indptr)
    dv1 = dense_variable(dv, indptr1)
    df1 = dense_fixed(8, name=None)
    df2 = dense_fixed(64, name=None)

    s0 = TensorStructInfo([df, dv, dv1], "float32")
    s1 = TensorStructInfo([df, dv, dv1], "float32")

    assert s0.axes[0] == df
    assert s0.axes[1] == dv
    assert s0.axes[2] == dv1
    assert s0.dtype == "float32"
    _check_json_roundtrip(s0)

    _check_equal(s0, s1)
    assert s0 == s1

    s2 = relax.sparse.TensorStructInfo([df, dv])

    assert s2.dtype == ""
    _check_json_roundtrip(s2)

    s3 = relax.sparse.TensorStructInfo([df, dv, df1, df2], "float32")
    assert s3.axes[0] == df
    assert s3.axes[1] == dv
    assert s3.axes[2] == df1
    assert s3.axes[3] == df2
    _check_json_roundtrip(s3)

    # Parent does not appear in the list.
    with pytest.raises(TVMError):
        TensorStructInfo([dv], "float32")
    with pytest.raises(TVMError):
        TensorStructInfo([dv1, dv], "float32")


def test_dense_padded():
    b = T.var("int64")
    max_length = T.var("int64", name="max_length")
    indptr = relax.Var("indptr", R.Tensor((b + 1,), "int64"))
    df = dense_fixed(b)
    dv = dense_variable(df, indptr)
    df1 = dense_fixed(8, name=None)
    df2 = dense_fixed(64, name=None)
    dp = dense_padded(dv, max_length, name="dp")
    sinfo = TensorStructInfo([df, df1, dp, df2], "float32")

    assert sinfo.axes[0] == df
    assert sinfo.axes[1] == df1
    assert sinfo.axes[2] == dp
    assert sinfo.axes[3] == df2
    _check_json_roundtrip(sinfo)


if __name__ == "__main__":
    tvm.testing.main()
