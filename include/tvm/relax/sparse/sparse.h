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
#ifndef TVM_RELAX_SPARSE_SPARSE_H_
#define TVM_RELAX_SPARSE_SPARSE_H_

#include <tvm/relax/sparse/axis.h>
#include <tvm/relax/struct_info.h>

namespace tvm {
namespace relax {
namespace sparse {

/*! \brief StructInfo of SparseTensor */
class TensorStructInfoNode : public StructInfoNode {
 public:
  /*!
   * \brief The axes of the sparse tensor. Corresponding to the shape of a normal Tensor.
   * \note We do not allow axis-unknown sparse tensors yet.
   */
  Array<Axis> axes;
  /*! \brief The sparse tensor content data type. Use void to denote the dtype is unknown. */
  DataType dtype;

  /*! \return Whether the struct info contains unknown dtype. */
  bool IsUnknownDtype() const { return dtype.is_void(); }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("axes", &axes);
    v->Visit("dtype", &dtype);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const TensorStructInfoNode* other, SEqualReducer equal) const {
    return equal(axes, other->axes) && equal(dtype, other->dtype);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(axes);
    hash_reduce(dtype);
  }

  static constexpr const char* _type_key = "relax.sparse.TensorStructInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to TensorStructInfoNode.
 * \sa TensorStructInfoNode
 */
class TensorStructInfo : public StructInfo {
 public:
  /*!
   * \brief Constructor with axes and dtype.
   * \param axes The axes of the sparse tensor.
   * \param dtype The content dtype of the sparse tensor.
   */
  TVM_DLL explicit TensorStructInfo(Array<Axis> axes, DataType dtype, Span span = Span());

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TensorStructInfo, StructInfo, TensorStructInfoNode);
};

}  // namespace sparse
}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_STRUCT_INFO_H_
