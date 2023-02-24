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
#include <tvm/ir/expr.h>
#include <tvm/relax/sparse/axis.h>
#include <tvm/relax/sparse/sparse.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

// Axis
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::sparse::Axis>(
        "", [](relax::sparse::Axis n, ObjectPath n_p, IRDocsifier d) -> Doc {
          // If the axis has no name, it means the axis is implicitly defined and
          // we can directly print its length.
          if (!n->name.defined()) {
            ICHECK(n->kind == relax::sparse::AxisKind::kDenseFixed);
            return d->AsDoc<Doc>(n->length.value(), n_p->Attr("length"));
          }

          if (!d->IsVarDefined(n)) {
            // Find the outmost Relax function frame. If not exist, the outmost Relax frame.
            RelaxFrameNode* f = nullptr;
            for (const Frame& frame : d->frames) {
              if (const auto* relax_frame = frame.as<RelaxFrameNode>()) {
                if (relax_frame->is_func) {
                  f = const_cast<RelaxFrameNode*>(relax_frame);
                  break;
                } else if (f == nullptr) {
                  f = const_cast<RelaxFrameNode*>(relax_frame);
                }
              }
            }
            // There should be at least one Relax frame
            if (f == nullptr) {
              LOG(FATAL) << "IndexError: No relax environment is found when printing a sparse Axis "
                            "under relax's dispatch token";
            }

            // Axis name
            IdDoc var =
                d->Define(n, GetRef<Frame>(f), n->name.value().empty() ? "ax" : n->name.value());
            var->source_paths.push_back(n_p);

            // Axis kind
            ExprDoc kind =
                Relax(d, "sp")->Attr("axis")->Attr(relax::sparse::AxisKind2String(n->kind));

            // Axis fields
            Array<ExprDoc> args;
            if (n->parent.defined()) {
              args.push_back(d->AsDoc<ExprDoc>(n->parent.value(), n_p->Attr("parent")));
            }
            if (n->length.defined()) {
              args.push_back(d->AsDoc<ExprDoc>(n->length.value(), n_p->Attr("length")));
            }
            if (n->nnz_col.defined()) {
              args.push_back(d->AsDoc<ExprDoc>(n->nnz_col.value(), n_p->Attr("nnz_col")));
            }
            if (n->indptr.defined()) {
              args.push_back(d->AsDoc<ExprDoc>(n->indptr.value(), n_p->Attr("indptr")));
            }
            if (n->indices.defined()) {
              args.push_back(d->AsDoc<ExprDoc>(n->indices.value(), n_p->Attr("indices")));
            }

            f->stmts.push_back(AssignDoc(var, kind->Call(args), NullOpt));
          }
          Optional<ExprDoc> doc = d->GetVarDoc(n);
          ICHECK(doc.defined()) << "IndexError: Axis is not defined in the environment: " << n;
          return doc.value();
        });

// sparse::TensorStructInfo
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::sparse::TensorStructInfo>("", [](relax::sparse::TensorStructInfo n,
                                                          ObjectPath n_p, IRDocsifier d) {
      Array<ExprDoc> args;
      args.push_back(d->AsDoc<ExprDoc>(n->axes, n_p->Attr("axes")));
      if (!n->IsUnknownDtype()) {
        args.push_back(LiteralDoc::DataType(n->dtype, n_p->Attr("dtype")));
      }
      ExprDoc head = Relax(d, "sp")->Attr("Tensor");
      if (args.empty()) {
        return head;
      }
      return head->Call(args);
    });

TVM_SCRIPT_REPR(relax::sparse::AxisNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::sparse::TensorStructInfoNode, ReprPrintRelax);

}  // namespace printer
}  // namespace script
}  // namespace tvm
