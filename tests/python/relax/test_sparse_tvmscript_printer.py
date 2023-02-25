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
import tvm.testing
from tvm import relax
from tvm.relax.sparse import TensorStructInfo
from tvm.relax.sparse.axis import dense_fixed, dense_variable, dense_padded
from tvm.relax.sparse.axis import sparse_fixed, sparse_variable
from tvm.script import relax as R, tir as T

b = T.var("int64", name="b")
d = T.var("int64", name="d")
nnz = T.var("int64", name="nnz")
length = T.var("int64", name="length")
indptr = relax.Var("indptr", R.Tensor((b + 1,), "int64"))
df = dense_fixed(b, name="df")
dv = dense_variable(df, d, nnz, indptr, name="dv")
df1 = dense_fixed(8, name=None)
df2 = dense_fixed(64, name=None)


def _assert_print(obj, expected):
    if not isinstance(obj, str):
        obj = obj.script(verbose_expr=True)
    obj = obj.strip()
    assert obj == expected.strip(), "\n" + obj


# Todo(relax-sparse): switch to use TVMScript when parser is ready.


def test_sparse_tensor_struct_info():
    indptr1 = relax.Var("indptr1", R.Tensor((nnz + 1,), "int64"))
    d1 = T.var("int64", name="d1")
    nnz1 = T.var("int64", name="nnz1")
    dv1 = dense_variable(df, d1, nnz1, indptr1, "dv1")

    obj0 = TensorStructInfo([df, dv, dv1], "float32")
    # NOTE(Zihao): such combination (df, dv, dv1) where both dv and dv1 depends on df, looks weird to me.
    assert obj0.__str__() == 'R.sp.Tensor([df, dv, dv1], "float32")'

    obj1 = TensorStructInfo([df, dv])
    assert obj1.__str__() == "R.sp.Tensor([df, dv])"

    obj2 = TensorStructInfo([df, dv, df1, df2], "float32")
    assert obj2.__str__() == 'R.sp.Tensor([df, dv, 8, 64], "float32")'


def test_func_dense_variable():
    sp_sinfo = TensorStructInfo([df, dv, df1, df2], "float32")
    var = relax.Var("gv", sp_sinfo)
    binding = relax.VarBinding(var, R.call_packed("my_func", [indptr], sinfo_args=sp_sinfo))
    func = relax.Function(
        params=[indptr],
        body=relax.SeqExpr([relax.BindingBlock([binding])], var),
        ret_struct_info=relax.ObjectStructInfo(),
    )

    _assert_print(
        func,
        """
# from tvm.script import tir as T
# from tvm.script import relax as R

@R.function
def main(indptr: R.Tensor(("b + 1",), dtype="int64")) -> R.Object:
    b = T.Var("b", "int64")
    df = R.sp.axis.dense_fixed(b)
    d = T.Var("d", "int64")
    nnz = T.Var("nnz", "int64")
    dv = R.sp.axis.dense_variable(df, d, nnz, indptr)
    gv: R.sp.Tensor([df, dv, 8, 64], "float32") = R.call_packed("my_func", (indptr,), sinfo_args=(R.sp.Tensor([df, dv, 8, 64], "float32"),))
    return gv""",
    )


def test_func_dense_padded():
    max_length = T.var("int64", name="max_length")
    dp = dense_padded(dv, max_length, name="dp")
    sp_sinfo = TensorStructInfo([df, df1, dp, df2], "float32")
    var = relax.Var("gv", sp_sinfo)
    binding = relax.VarBinding(var, R.call_packed("my_func", [indptr], sinfo_args=sp_sinfo))
    func = relax.Function(
        params=[indptr],
        body=relax.SeqExpr([relax.BindingBlock([binding])], var),
        ret_struct_info=relax.ObjectStructInfo(),
    )

    _assert_print(
        func,
        """
# from tvm.script import tir as T
# from tvm.script import relax as R

@R.function
def main(indptr: R.Tensor(("b + 1",), dtype="int64")) -> R.Object:
    b = T.Var("b", "int64")
    df = R.sp.axis.dense_fixed(b)
    d = T.Var("d", "int64")
    nnz = T.Var("nnz", "int64")
    dv = R.sp.axis.dense_variable(df, d, nnz, indptr)
    max_length = T.Var("max_length", "int64")
    dp = R.sp.axis.dense_padded(dv, max_length)
    gv: R.sp.Tensor([df, 8, dp, 64], "float32") = R.call_packed("my_func", (indptr,), sinfo_args=(R.sp.Tensor([df, 8, dp, 64], "float32"),))
    return gv""",
    )


def test_func_sparse_fixed():
    nnz_col = T.var("int64", "nnz_col")
    indices = relax.Var("indices", R.Tensor((b * nnz_col,), "int64"))
    sf = sparse_fixed(df, length, nnz_col, indices, name="sf")
    sp_sinfo = TensorStructInfo([df, sf], "float32")
    var = relax.Var("gv", sp_sinfo)
    binding = relax.VarBinding(var, R.call_packed("my_func", [indptr], sinfo_args=sp_sinfo))
    func = relax.Function(
        params=[indptr],
        body=relax.SeqExpr([relax.BindingBlock([binding])], var),
        ret_struct_info=relax.ObjectStructInfo(),
    )

    _assert_print(
        func,
        """
# from tvm.script import tir as T
# from tvm.script import relax as R

@R.function
def main(indptr: R.Tensor(("b + 1",), dtype="int64")) -> R.Object:
    b = T.Var("b", "int64")
    df = R.sp.axis.dense_fixed(b)
    length = T.Var("length", "int64")
    nnz_col = T.Var("nnz_col", "int64")
    sf = R.sp.axis.sparse_fixed(df, length, nnz_col, indices)
    indices: R.Tensor((b * nnz_col,), dtype="int64")
    gv: R.sp.Tensor([df, sf], "float32") = R.call_packed("my_func", (indptr,), sinfo_args=(R.sp.Tensor([df, sf], "float32"),))
    return gv""",
    )


def test_func_sparse_variable():
    indices = relax.Var("indices", R.Tensor((nnz,), "int64"))
    sv = sparse_variable(df, length, nnz, indptr, indices, name="sv")
    sp_sinfo = TensorStructInfo([df, sv], "float32")
    var = relax.Var("gv", sp_sinfo)
    binding = relax.VarBinding(var, R.call_packed("my_func", [indptr], sinfo_args=sp_sinfo))
    func = relax.Function(
        params=[indptr],
        body=relax.SeqExpr([relax.BindingBlock([binding])], var),
        ret_struct_info=relax.ObjectStructInfo(),
    )

    print(func.script())

    _assert_print(
        func,
        """
# from tvm.script import tir as T
# from tvm.script import relax as R

@R.function
def main(indptr: R.Tensor(("b + 1",), dtype="int64")) -> R.Object:
    b = T.Var("b", "int64")
    df = R.sp.axis.dense_fixed(b)
    length = T.Var("length", "int64")
    nnz = T.Var("nnz", "int64")
    sv = R.sp.axis.sparse_variable(df, length, nnz, indptr, indices)
    indices: R.Tensor((nnz,), dtype="int64")
    gv: R.sp.Tensor([df, sv], "float32") = R.call_packed("my_func", (indptr,), sinfo_args=(R.sp.Tensor([df, sv], "float32"),))
    return gv""",
    )


if __name__ == "__main__":
    tvm.testing.main()
