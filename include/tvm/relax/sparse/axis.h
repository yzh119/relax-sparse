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

/*!
 * \file include/tvm/relax/sparse/axis.h
 * \brief The axis data structures for sparse Relax.
 */

#ifndef TVM_RELAX_SPARSE_AXIS_H_
#define TVM_RELAX_SPARSE_AXIS_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {
namespace sparse {

enum class AxisKind : int {
  kDenseFixed = 0,
  kDenseVariable = 1,
  kDensePadded = 2,
  kSparseFixed = 3,
  kSparseVariable = 4,
};

class Axis;

/*! \brief The axis node, which denotes an axis (or dimension) of a sparse tensor. */
class AxisNode : public Object {
 public:
  /*! \brief The length of this axis. */
  PrimExpr length;
  /*! \brief The number of non-zeros in sparse iteration space composed of ancestor(including self)
   * axes. */
  PrimExpr nnz;
  /*!
   * \brief The parent of the axis, which represents the axis dependency.
   * \note We require the parent axis for every axis to be explicit, as long as
   * the parent axis exists.
   */
  Optional<ObjectRef> parent;
  /*! \brief The indptr array of the axis, which should be a 1-dim Tensor. */
  Optional<Var> indptr;
  /*! \brief The indices array of the axis, which should be a 1-dim Tensor. */
  Optional<Var> indices;
  /*! \brief The number non-zero elements per instance along this axis. */
  Optional<PrimExpr> nnz_col;
  /*! \brief The kind of this axis. */
  AxisKind kind;

  /*!
   * \brief The optional name for of the axis. Undefined means the axis is an
   * implicitly defined dense-fixed axis.
   * \note This field can only be undefined for dense-fixed axis.
   */
  Optional<String> name;

  /*! \brief Return the parent field as an Axis. */
  Axis GetParent() const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("length", &length);
    v->Visit("parent", &parent);
    v->Visit("indptr", &indptr);
    v->Visit("indices", &indices);
    v->Visit("nnz_col", &nnz_col);
    v->Visit("kind", &kind);
    v->Visit("nnz", &nnz);
    v->Visit("name", &name);
  }

  bool SEqualReduce(const AxisNode* other, SEqualReducer equal) const {
    return equal(length, other->length) && equal(parent, other->parent) &&
           equal(indptr, other->indptr) && equal(indices, other->indices) &&
           equal(nnz_col, other->nnz_col) && equal(kind, other->kind) && equal(nnz, other->nnz) &&
           equal(name, other->name);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(length);
    hash_reduce(parent);
    hash_reduce(indptr);
    hash_reduce(indices);
    hash_reduce(nnz_col);
    hash_reduce(kind);
    hash_reduce(nnz);
    hash_reduce(name);
  }

  static constexpr const char* _type_key = "relax.sparse.Axis";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(AxisNode, Object);
};

/*!
 * \brief Managed reference to AxisNode.
 * \sa AxisNode
 */
class Axis : public ObjectRef {
 public:
  /*!
   * \brief Constructor for dense-fixed axis.
   * \param length The length of the axis.
   * \param name The optional name for of the axis. It is by default an empty
   * string. Being undefined means the axis is an implicitly defined dense-fixed
   * axis.
   */
  TVM_DLL static Axis DenseFixed(PrimExpr length, Optional<String> name = String(""));
  /*!
   * \brief Constructor for dense-variable axis.
   * \param parent The parent axis of this axis, which should be explicit.
   * \param length The length of this axis.
   * \param nnz The number of non-zeros in sparse iteration space composed of ancestor(including
   * self) axes. \param indptr The indptr array of this axis. \param length The length of this axis.
   * \param name The name of the axis.
   */
  TVM_DLL static Axis DenseVariable(Axis parent, PrimExpr length, PrimExpr nnz, Var indptr,
                                    String name);
  /*!
   * \brief Constructor for dense-padded axis.
   * \param parent The parent axis of this axis, which should be explicit.
   * \param length The padded maximum length of this axis.
   * \param name The name of the axis.
   */
  TVM_DLL static Axis DensePadded(Axis parent, PrimExpr length, String name);
  /*!
   * \brief Constructor for sparse-fixed axis.
   * \param parent The parent axis of this axis, which should be explicit.
   * \param length The length of this axis.
   * \param nnz_col The number of non-zero elements per instance on this axis.
   * \param indices The indices array of this axis.
   * \param name The name of the axis.
   */
  TVM_DLL static Axis SparseFixed(Axis parent, PrimExpr length, PrimExpr nnz_col, Var indices,
                                  String name);
  /*!
   * \brief Constructor for sparse-variable axis.
   * \param parent The parent axis of this axis, which should be explicit.
   * \param length The length of this axis.
   * \param nnz The number of non-zeros in sparse iteration space composed of ancestral axes.
   * \param indptr The indptr array of this axis.
   * \param indices The indices array of this axis.
   * \param name The name of the axis.
   */
  TVM_DLL static Axis SparseVariable(Axis parent, PrimExpr length, PrimExpr nnz, Var indptr,
                                     Var indices, String name);

  TVM_DEFINE_OBJECT_REF_METHODS(Axis, ObjectRef, AxisNode);
};

inline const char* AxisKind2String(AxisKind kind) {
  switch (kind) {
    case AxisKind::kDenseFixed:
      return "dense_fixed";
    case AxisKind::kDenseVariable:
      return "dense_variable";
    case AxisKind::kDensePadded:
      return "dense_padded";
    case AxisKind::kSparseFixed:
      return "sparse_fixed";
    case AxisKind::kSparseVariable:
      return "sparse_variable";
  }
  throw;
}

}  // namespace sparse
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_SPARSE_AXIS_H_
