/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relax/ir/sparse/sparse.cc
 * \brief Sparse relax constructs.
 */
#include <tvm/relax/sparse/sparse.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/struct_info_functor.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace relax {
namespace sparse {

// sparse::TensorStructInfo
void CheckAxesValidity(Array<Axis> axes) {
  // To be valid, the input axes should conform to two requirements:
  // 1. for an axis in the input array, if it is not padded axis and it depends
  // on some other axis, that axis should appear in the input array as well.
  // 2. no cyclic dependency,
  int ndim = axes.size();
  for (const Axis& axis : axes) {
    if (!axis->parent.defined()) {
      continue;
    }

    // Check requirement 1.
    if (axis->kind != AxisKind::kDensePadded) {
      CHECK(std::find(axes.begin(), axes.end(), axis->GetParent()) != axes.end())
          << "The parent axis of xxx does not appear in the input axes. Therefore, the input array "
             "of axes is invalid.";
    }
    // Check requirement 2.
    Axis _axis = axis;
    int depth = 0;
    while (depth < ndim && _axis->parent.defined()) {
      _axis = _axis->GetParent();
      ++depth;
    }
    CHECK_LT(depth, ndim) << "The input axes have cyclic dependency, and is thereby invalid.";
  }
}

TensorStructInfo::TensorStructInfo(Array<Axis> axes, DataType dtype, Span span) {
  CheckAxesValidity(axes);
  ObjectPtr<TensorStructInfoNode> n = make_object<TensorStructInfoNode>();
  n->axes = std::move(axes);
  n->dtype = dtype;
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TensorStructInfoNode);

TVM_REGISTER_GLOBAL("relax.sparse.TensorStructInfo")
    .set_body_typed([](Array<Axis> axes, DataType dtype, Span span) {
      return TensorStructInfo(std::move(axes), dtype, span);
    });

}  // namespace sparse
}  // namespace relax
}  // namespace tvm
