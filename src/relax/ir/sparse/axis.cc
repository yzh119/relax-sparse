/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/relax/sparse/axis.h>

namespace tvm {
namespace relax {
namespace sparse {

// Axis
TVM_REGISTER_NODE_TYPE(AxisNode);

Axis AxisNode::GetParent() const {
  CHECK(parent.defined()) << "GetParent can only be applied to axis who has a parent.";
  return Downcast<Axis>(parent);
}

Axis Axis::DenseFixed(PrimExpr length, Optional<String> name) {
  ObjectPtr<AxisNode> n = make_object<AxisNode>();
  n->length = std::move(length);
  n->nnz = n->length;
  n->name = std::move(name);
  n->kind = AxisKind::kDenseFixed;
  return Axis(std::move(n));
}

// Todo(relax-sparse): Check indptr/indices are 1-dim integer tensor in normalization.

Axis Axis::DenseVariable(Axis parent, PrimExpr length, PrimExpr nnz, Var indptr, String name) {
  CHECK(parent->name.defined()) << "The parent axis of any axis should be explicitly defined.";
  ObjectPtr<AxisNode> n = make_object<AxisNode>();
  n->parent = std::move(parent);
  n->length = std::move(length);
  n->nnz = std::move(nnz);
  n->indptr = std::move(indptr);
  n->name = std::move(name);
  n->kind = AxisKind::kDenseVariable;
  return Axis(std::move(n));
}

Axis Axis::DensePadded(Axis parent, PrimExpr length, String name) {
  CHECK(parent->name.defined()) << "The parent axis of any axis should be explicitly defined.";
  CHECK(parent->kind == AxisKind::kDenseVariable || parent->kind == AxisKind::kDenseFixed)
      << "The parent axis of dense-padded axis should be dense-variable or dense-fixed.";
  ObjectPtr<AxisNode> n = make_object<AxisNode>();
  n->parent = std::move(parent);
  // TODO(zihao): item to discuss, what is the nnz for dense padded axis?
  n->length = std::move(length);
  n->nnz = length;
  n->name = std::move(name);
  n->kind = AxisKind::kDensePadded;
  return Axis(std::move(n));
}

Axis Axis::SparseFixed(Axis parent, PrimExpr length, PrimExpr nnz_col, Var indices, String name) {
  CHECK(parent->name.defined()) << "The parent axis of any axis should be explicitly defined.";
  ObjectPtr<AxisNode> n = make_object<AxisNode>();
  PrimExpr nnz = parent->nnz * nnz_col;
  n->parent = std::move(parent);
  n->length = std::move(length);
  n->nnz_col = std::move(nnz_col);
  n->nnz = nnz;
  n->indices = std::move(indices);
  n->name = std::move(name);
  n->kind = AxisKind::kSparseFixed;
  return Axis(std::move(n));
}

Axis Axis::SparseVariable(Axis parent, PrimExpr length, PrimExpr nnz, Var indptr, Var indices,
                          String name) {
  CHECK(parent->name.defined()) << "The parent axis of any axis should be explicitly defined.";
  ObjectPtr<AxisNode> n = make_object<AxisNode>();
  n->parent = std::move(parent);
  n->length = std::move(length);
  n->nnz = std::move(nnz);
  n->indptr = std::move(indptr);
  n->indices = std::move(indices);
  n->name = std::move(name);
  n->kind = AxisKind::kSparseVariable;
  return Axis(std::move(n));
}

TVM_REGISTER_GLOBAL("relax.sparse.DenseFixedAxis")
    .set_body_typed([](PrimExpr length, Optional<String> name) {
      return Axis::DenseFixed(std::move(length), std::move(name));
    });

TVM_REGISTER_GLOBAL("relax.sparse.DenseVariableAxis")
    .set_body_typed([](Axis parent, PrimExpr length, PrimExpr nnz, Var indptr, String name) {
      return Axis::DenseVariable(std::move(parent), std::move(length), std::move(nnz),
                                 std::move(indptr), std::move(name));
    });

TVM_REGISTER_GLOBAL("relax.sparse.DensePaddedAxis")
    .set_body_typed([](Axis parent, PrimExpr length, String name) {
      return Axis::DensePadded(std::move(parent), std::move(length), std::move(name));
    });

TVM_REGISTER_GLOBAL("relax.sparse.SparseFixedAxis")
    .set_body_typed([](Axis parent, PrimExpr length, PrimExpr nnz_col, Var indices, String name) {
      return Axis::SparseFixed(std::move(parent), std::move(length), std::move(nnz_col),
                               std::move(indices), std::move(name));
    });

TVM_REGISTER_GLOBAL("relax.sparse.SparseVariableAxis")
    .set_body_typed([](Axis parent, PrimExpr length, PrimExpr nnz, Var indptr, Var indices,
                       String name) {
      return Axis::SparseVariable(std::move(parent), std::move(length), std::move(nnz),
                                  std::move(indptr), std::move(indices), std::move(name));
    });

}  // namespace sparse
}  // namespace relax
}  // namespace tvm
