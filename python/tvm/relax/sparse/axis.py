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
"""The axis data structure of sparse Relax."""
import enum
from typing import Optional

import tvm
import tvm._ffi
import tvm.relax

from ..expr import Var
from ...ir import Node, PrimExpr
from . import _ffi_api
from tvm import TVMError


class AxisKind(enum.IntEnum):
    kDenseFixed = 0
    kDenseVariable = 1
    kDensePadded = 2
    kSparseFixed = 3
    kSparseVariable = 4


@tvm._ffi.register_object("relax.sparse.Axis")
class Axis(Node):
    """The axis node, which denotes an axis (or dimension) of a sparse tensor.

    Attributes
    ----------
    length : Optional[PrimExpr]
        The length of this axis. Should be defined for axes other than dense-variable axis.

    parent : Optional[Axis]
        The parent of the axis, which represents the axis dependency.

    indptr : Optional[Var]
        The indptr array of the axis, which should be a 1-dim Tensor.

    indices : Optional[Var]
        The indices array of the axis, which should be a 1-dim Tensor.

    nnz_col : Optional[PrimExpr]
        The number non-zero elements per instance along this axis.

    kind : AxisKind
        The kind of this axis.

    name : Optional[str]
        The optional name for of the axis. Undefined means the axis is an
        implicitly defined dense-fixed axis.
        This field can only be undefined for dense-fixed axis.

    Notes
    -----
    We require the parent axis for every axis to be explicit, as long as the
    parent axis exists.
    """

    parent: Optional["Axis"]
    length: Optional[PrimExpr]
    indptr: Optional[Var]
    indices: Optional[Var]
    nnz_col: Optional[PrimExpr]
    kind: AxisKind
    name: Optional[str]

    def __init__(self, *args, **kwargs) -> None:
        raise TVMError(
            "Creating Axis through constructor is disabled. Please use functions like "
            "`dense_fixed`, `dense_variable`, `sparse_variable` to create an Axis."
        )


def dense_fixed(length: PrimExpr, name: Optional[str] = "") -> Axis:
    """Creating a dense-fixed axis.

    Parameters
    ----------
    length : PrimExpr
        The length of the axis.

    name : Optional[str]
        The optional name for of the axis.
        - By default the name is an empty string.
        - If the input name is `None`, it means the axis is an implicitly
        defined dense-fixed axis.

    Return
    ------
    axis : Axis
        The created dense-fixed axis.
    """
    return _ffi_api.DenseFixedAxis(length, name)


def dense_variable(parent: Axis, indptr: Var, name: str = "") -> Axis:
    """Creating a dense-variable axis.

    Parameters
    ----------
    parent : Axis
        The parent axis of this axis, which should be explicit.

    indptr : Var
        The indptr array of this axis.

    name : str
        The optional name for of the axis.

    Return
    ------
    axis : Axis
        The created dense-variable axis.
    """
    return _ffi_api.DenseVariableAxis(parent, indptr, name)


def dense_padded(parent: Axis, length: PrimExpr, name: str = "") -> Axis:
    """Creating a dense-padded axis.

    Parameters
    ----------
    parent : Axis
        The parent axis of this axis, which should be explicit.

    length : PrimExpr
        The padded maximum length of this axis.

    name : str
        The optional name for of the axis.

    Return
    ------
    axis : Axis
        The created dense-padded axis.
    """
    return _ffi_api.DensePaddedAxis(parent, length, name)


def sparse_fixed(
    parent: Axis, length: PrimExpr, nnz_col: PrimExpr, indices: Var, name: str = ""
) -> Axis:
    """Creating a sparse-fixed axis.

    Parameters
    ----------
    parent : Axis
        The parent axis of this axis, which should be explicit.

    length : PrimExpr
        The length of this axis.

    nnz_col : PrimExpr
        The number of non-zero elements per instance on this axis.

    indices : Var
        The indices array of this axis.

    name : str
        The optional name for of the axis.

    Return
    ------
    axis : Axis
        The created sparse-fixed axis.
    """
    return _ffi_api.SparseFixedAxis(parent, length, nnz_col, indices, name)


def sparse_variable(
    parent: Axis, length: PrimExpr, indptr: Var, indices: Var, name: str = ""
) -> Axis:
    """Creating a sparse-variable axis.

    Parameters
    ----------
    parent : Axis
        The parent axis of this axis, which should be explicit.

    length : PrimExpr
        The length of this axis.

    indptr : Var
        The indptr array of this axis.

    indices : Var
        The indices array of this axis.

    name : str
        The optional name for of the axis.

    Return
    ------
    axis : Axis
        The created sparse-variable axis.
    """
    return _ffi_api.SparseVariableAxis(parent, length, indptr, indices, name)
