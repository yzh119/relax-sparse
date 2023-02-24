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

from tvm import TVMError
from tvm import relax
from tvm.relax.sparse.axis import Axis, AxisKind
from tvm.relax.sparse.axis import dense_fixed, dense_variable, dense_padded
from tvm.relax.sparse.axis import sparse_fixed, sparse_variable
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


def test_dense_fixed():
    b = T.var("int64", name="b")
    df0 = dense_fixed(b, name="")
    df1 = dense_fixed(b)
    df2 = dense_fixed(b, name=None)

    assert isinstance(df0, Axis)
    assert df0.length == b
    assert df0.parent is None
    assert df0.indptr is None
    assert df0.indices is None
    assert df0.nnz_col is None
    assert df0.name == ""
    assert df2.name is None
    assert df0.kind == AxisKind.kDenseFixed

    _check_equal(df0, df1)
    assert not tvm.ir.structural_equal(df0, df2)

    _check_json_roundtrip(df0)


def test_dense_variable():
    b = T.var("int64")
    df0 = dense_fixed(b)
    df1 = dense_fixed(b, name=None)
    indptr = relax.Var("indptr", R.Tensor((b + 1,), "int64"))

    dv0 = dense_variable(df0, indptr, name="dv")
    dv1 = dense_variable(df0, indptr, name="dv")

    assert isinstance(dv0, Axis)
    assert dv0.parent == df0
    assert dv0.length is None
    assert dv0.indptr == indptr
    assert dv0.indices is None
    assert dv0.nnz_col is None
    assert dv0.name == "dv"
    assert dv0.kind == AxisKind.kDenseVariable

    _check_equal(dv0, dv1)
    _check_json_roundtrip(dv0)

    # Parent axis cannot be implicit.
    with pytest.raises(TVMError):
        dense_variable(df1, indptr)


def test_dense_padded():
    b = T.var("int64")
    max_len = T.var("int64")
    df0 = dense_fixed(b)
    df1 = dense_fixed(b, name=None)
    indptr = relax.Var("indptr", R.Tensor((b + 1,), "int64"))
    dv = dense_variable(df0, indptr)

    dp0 = dense_padded(dv, max_len)
    dp1 = dense_padded(dv, max_len)

    assert isinstance(dp0, Axis)
    assert dp0.length == max_len
    assert dp0.parent == dv
    assert dp0.indptr == None
    assert dp0.indices == None
    assert dp0.nnz_col == None
    assert dp0.name == ""
    assert dp0.kind == AxisKind.kDensePadded

    _check_equal(dp0, dp1)
    _check_json_roundtrip(dp0)

    # Parent axis must be dense-fixed or dense-variable
    with pytest.raises(TVMError):
        dense_padded(dp0, max_len)
    # Parent axis cannot be implicit.
    with pytest.raises(TVMError):
        dense_padded(df1, max_len)


def test_sparse_fixed():
    b = T.var("int64")
    length = T.var("int64")
    nnz_col = T.var("int64")
    df0 = dense_fixed(b)
    df1 = dense_fixed(b, name=None)
    indices = relax.Var("indices", R.Tensor((b * nnz_col,), "int64"))

    sf0 = sparse_fixed(df0, length, nnz_col, indices)
    sf1 = sparse_fixed(df0, length, nnz_col, indices)

    assert isinstance(sf0, Axis)
    assert sf0.length == length
    assert sf0.parent == df0
    assert sf0.indptr is None
    assert sf0.indices == indices
    assert sf0.nnz_col == nnz_col
    assert sf0.name == ""
    assert sf0.kind == AxisKind.kSparseFixed

    _check_equal(sf0, sf1)
    _check_json_roundtrip(sf0)

    # Parent axis cannot be implicit.
    with pytest.raises(TVMError):
        sparse_fixed(df1, length, nnz_col, indices)


def test_sparse_variable():
    b = T.var("int64")
    length = T.var("int64")
    nnz = T.var("int64")
    df0 = dense_fixed(b)
    df1 = dense_fixed(b, name=None)
    indptr = relax.Var("indptr", R.Tensor((b + 1,), "int64"))
    indices = relax.Var("indices", R.Tensor((nnz,), "int64"))

    sv0 = sparse_variable(df0, length, indptr, indices)
    sv1 = sparse_variable(df0, length, indptr, indices)

    assert isinstance(sv0, Axis)
    assert sv0.length == length
    assert sv0.parent == df0
    assert sv0.indptr == indptr
    assert sv0.indices == indices
    assert sv0.nnz_col is None
    assert sv0.name == ""
    assert sv0.kind == AxisKind.kSparseVariable

    _check_equal(sv0, sv1)
    _check_json_roundtrip(sv0)

    # Parent axis cannot be implicit.
    with pytest.raises(TVMError):
        sparse_variable(df1, length, indptr, indices)


def test_use_constructor_failure():
    b = T.var("int64")
    length = T.var("int64")
    nnz_col = T.var("int64")
    nnz = T.var("int64")
    indptr = relax.Var("indptr", R.Tensor((b + 1,), "int64"))
    indices = relax.Var("indices", R.Tensor((nnz,), "int64"))

    df = dense_fixed(b)
    dv = dense_variable(df, indptr)

    with pytest.raises(TVMError):
        Axis(b)
    with pytest.raises(TVMError):
        Axis(df, indptr)
    with pytest.raises(TVMError):
        Axis(dv, length)
    with pytest.raises(TVMError):
        Axis(df, length, nnz_col, indices)
    with pytest.raises(TVMError):
        Axis(df, length, indptr, indices)


if __name__ == "__main__":
    tvm.testing.main()
