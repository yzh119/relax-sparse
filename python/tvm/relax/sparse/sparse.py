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
# pylint: disable=invalid-name, unused-import
"""The constructs of sparse Relax."""
from typing import List, Optional

import tvm._ffi
import tvm
from tvm.ir import Span

from ..struct_info import StructInfo
from .axis import Axis
from . import _ffi_api


@tvm._ffi.register_object("relax.sparse.TensorStructInfo")
class TensorStructInfo(StructInfo):
    """StructInfo of a SparseTensor value.

    Parameters
    ----------
    axes : List[Axis]
        The axes of the sparse tensor.

    dtype : str
        The content data type, with default value an empty string, denoting void type.

    Note
    ----
    Do not specify shape and ndim at the same time.
    """

    axes: List[Axis]
    dtype: str
    span: Span

    def __init__(
        self,
        axes: List[Axis],
        dtype: str = "",
        span: Span = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.TensorStructInfo, axes, dtype, span  # type: ignore
        )
