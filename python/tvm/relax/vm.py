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
@tvm._ffi.register_object("relax.vm.Instruction")
class Instruction(ObjectRef):
    span: Span
    def __init__(self, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Instruction,  span) # type: ignore


@tvm._ffi.register_object("relax.vm.AllocTensor")
class AllocTensor(Instruction):
    storage: tvm::runtime::vm::RegName
    offset: tvm::runtime::vm::Index
    shape_register: tvm::runtime::vm::RegName
    span: Span
    def __init__(self, storage: tvm::runtime::vm::RegName, offset: tvm::runtime::vm::Index, shape_register: tvm::runtime::vm::RegName, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.AllocTensor,  storage,  offset,  shape_register,  span) # type: ignore


