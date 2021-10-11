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
 * \file src/relax/transform/memory_rewrite.cc
 * \brief
 */
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

#include "../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relax {

// ==================
// ExplicitMemMutator
// Lower call_dps to a form with explicit tensor allocation.
// After this lowering, we can perform memory planning passes and furthur compile it to VM
// Example:
// y: Tensor[n, m] = rx.call_dps((n, m), op.identity, (x))
// -->
// lv0 = rx.call("relax.builtin.alloc_tensor", [n, m])
// rx.call_packed(op.identity, x, lv0)

class ExplicitMemMutator : public DataflowMutator {
 public:
  explicit ExplicitMemMutator(IRModule mod) { mod_ = mod; }

  IRModule Lower() {
    ret_mod_ = IRModule();
    for (auto& p : mod_->functions) {
      if (!p.second->IsInstance<FunctionNode>()) {
        continue;
      }
      Expr new_func = this->Mutate(p.second);
      ret_mod_->Add(p.first, Downcast<BaseFunc>(new_func));
    }
    return ret_mod_;
  }

  BindingBlock VisitDataflowBlock(const DataflowBlock& block) override {
    this->builder_ = LazyIRBuilderNode::Create(block);
    {
      With<DataflowScope> scope(this->builder_);
      // switch from building a DataflowBlock to building an impure BindingBlock becasue the program
      // after memory rewriting has side effects
      this->builder_->is_dataflow_ = false;

      for (auto binding : block->bindings) {
        if (auto* var_binding = binding.as<VarBindingNode>()) {
          Var var = this->VisitVarBinding(Downcast<VarBinding>(binding), this->builder_);
          this->pre_post_var_map_[var_binding->var] = var;
        }
      }
    }
    return this->builder_->GetBlocks().back();
  }

  Var VisitVarBinding(const VarBinding& binding, IRBuilder& ir_builder) override {
    static const Op& call_dps_op = Op::Get("relax.call_dps");
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");

    const CallNode* op = binding->value.as<CallNode>();
    if (op && op->op == call_dps_op) {
      Var tensor = ir_builder->Emit(Call(alloc_tensor_op, {op->args[0]}));
      return ir_builder->Emit(binding->var, Call(op->args[1], {op->args[2], tensor}));
    }
    return ir_builder->Emit(binding);
  }

 private:
  IRModule mod_;
  IRModule ret_mod_;
};

TVM_REGISTER_GLOBAL("relax.transform.explicit_memory_rewrite").set_body_typed([](IRModule mod) {
  return ExplicitMemMutator(mod).Lower();
});

// ==================
// MemLowerMutator
// Lower the relax.builtin.alloc_tensor op to call VM builtin packed functions.
// Example:
// x = relax.builtin.alloc_tensor((m, n))
// -->
// gv0 = relax.call_packed("vm.builtin.alloc_storage", (m * n), alignment, device_type,
// relax.attrs.AllocStorageAttrs) gv1 = relax.call_packed("vm.builtin.alloc_tensor", gv0, offset,
// (m, n), relax.attrs.AllocTensorAttrs)

class MemLowerMutator : public ExprMutator {
 public:
  explicit MemLowerMutator(IRModule mod) { mod_ = mod; }

  IRModule Lower() {
    ret_mod_ = IRModule();
    for (auto& p : mod_->functions) {
      if (!p.second->IsInstance<FunctionNode>()) {
        continue;
      }
      Expr new_func = this->Mutate(p.second);
      ret_mod_->Add(p.first, Downcast<BaseFunc>(new_func));
    }
    return ret_mod_;
  }

  Expr ComputeStorageSize(const Expr& shape, const Type& type) const {
    DynTensorType tensor_type = Downcast<DynTensorType>(type);
    DataType dtype = DataType(tensor_type->dtype);
    // Question: what if the dtype of tensor_type is unknown?
    // Symbolic/static shape case
    if (auto* shape_expr = shape.as<ShapeExprNode>()) {
      PrimExpr num = PrimExpr(dtype.bits()) * PrimExpr(dtype.lanes());
      PrimExpr add = num + 7;
      PrimExpr ret = 1;
      for (PrimExpr dim : shape_expr->values) {
        ret = ret * dim;
      }
      ret = ret * (add / PrimExpr(8));
      return ShapeExpr({ret});
    }
    // Fully dynamic shape case
    // will need to dedup with ComputeStorageInRelay when we upstream
    Expr prod = relay::Prod(shape, Array<Integer>(nullptr), false, false);
    Expr num = relay::MakeConstantScalar(DataType::Int(64), dtype.bits() * dtype.lanes());
    Expr add = relay::Add(num, relay::MakeConstantScalar(DataType::Int(64), 7));
    Expr div = relay::MakeConstantScalar(DataType::Int(64), 8);
    Expr ret = relay::Multiply(prod, relay::Divide(add, div));
    return ret;
  }

  Var VisitVarBinding(const VarBinding& binding, IRBuilder& builder) {
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");

    const CallNode* op = binding->value.as<CallNode>();
    if (op && op->op == alloc_tensor_op) {
      ShapeExpr tensor_shape = Downcast<ShapeExpr>(op->args[0]);
      // TODO(@yuchen): Get the type of input x, options: add an attr to relax.builtin.alloc_tensor
      Type tensor_type = DynTensorType(2, DataType::Float(32));
      Expr storage_size = ComputeStorageSize(tensor_shape, tensor_type);
      ShapeExpr alignment = ShapeExpr({IntImm(DataType::Int(64), 64)});
      ShapeExpr device_type = ShapeExpr({IntImm(DataType::Int(64), 1)});
      auto storage_attr = make_object<AllocStorageAttrs>();
      storage_attr->dtype = DataType::Float(32);

      Var storage =
          builder->Emit(Call(ExternFunc("vm.builtin.alloc_storage"),
                             {storage_size, alignment, device_type}, Attrs(storage_attr)));

      ShapeExpr offset = ShapeExpr({IntImm(DataType::Int(64), 0)});
      auto tensor_attr = make_object<AllocTensorAttrs>();
      tensor_attr->dtype = DataType::Float(32);
      Expr shape = op->args[0];
      return builder->Emit(binding->var, Call(ExternFunc("vm.builtin.alloc_tensor"),
                                              {storage, offset, shape}, Attrs(tensor_attr)));
    }
    return builder->Emit(binding);
  }

 private:
  IRModule mod_;
  IRModule ret_mod_;
};

TVM_REGISTER_GLOBAL("relax.transform.memory_lower").set_body_typed([](IRModule mod) {
  return MemLowerMutator(mod).Lower();
});

}  // namespace relax
}  // namespace tvm
