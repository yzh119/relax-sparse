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
 * \file src/relax/transform/type_inference.cc
 * \brief Type inference for Relax.
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/ir_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relax {

class TypeInferencer : public IRFunctor<ObjectRef(const ObjectRef&)> {
 public:
  enum InferenceMode {
    /*!
     * \brief Run type inference on all exprs, writing the inferred type directly into the expr if
     * it did not have a type before. Otherwise, if the inferred type differs, the node is
     * reconstructed with the new inferred type.
     */
    kNormal = 0,
    /*!
     * \brief Naively reuse the checked type of expressions if present.
     */
    kReuseCheckedType = 1,
  };

  /*!
   * \param mod The IRModule for resolving global variables.
   * \param diag_ctx The diagnostic context for emitting errors.
   * \param use_checked_type Whether or not to trust and reuse checked types on expressions.
   */
  TypeInferencer(IRModule mod, DiagnosticContext diag_ctx, InferenceMode mode)
      : mod_(mod), diag_ctx_(diag_ctx), mode_(mode) {}

  Expr InferExpr(const Expr& expr) {
    if (!expr_memo_.count(expr)) {
      if (mode_ == InferenceMode::kReuseCheckedType && expr->checked_type_.defined()) {
        expr_memo_[expr] = expr;
      } else {
        ICHECK(mode_ == InferenceMode::kNormal);
        expr_memo_[expr] = Downcast<Expr>(VisitNode(expr));
      }
    }
    return expr_memo_[expr];
  }

  IRModule InferMod() {
    IRModule new_mod;

    for (auto& pr : mod_->functions) {
      if (!pr.second.as<FunctionNode>()) {
        // don't mess with PrimFuncs etc.
        new_mod->Add(pr.first, pr.second);
        continue;
      }

      // FIXME(@altanh): recursive calls will break
      GlobalVar new_var = Downcast<GlobalVar>(InferExpr(pr.first));
      Expr new_func = InferExpr(pr.second);
      new_mod->Add(new_var, Downcast<BaseFunc>(new_func));
    }

    diag_ctx_.Render();

    return new_mod;
  }

  ObjectRef VisitNode_(const CallNode* op) override {
    static const Op& call_dps = Op::Get("relax.call_dps");
    static OpAttrMap<FInferType> op_inference_map = Op::GetAttrMap<FInferType>("FInferType");
    // TODO(@altanh): do we want to use IncompleteType? might not be relevant anymore if we aren't
    //                doing unification based inference
    Call call = GetRef<Call>(op);
    Type new_ty;

    // infer types for op and args, (COW) update the call if necessary
    bool same_args = true;
    Expr new_op = InferExpr(call->op);
    if (!new_op.same_as(call->op)) {
      call.CopyOnWrite()->op = new_op;
    }
    Array<Expr> new_args;
    for (const Expr& arg : call->args) {
      Expr new_arg = InferExpr(arg);
      same_args &= new_arg.same_as(arg);
      new_args.push_back(new_arg);
    }
    if (!same_args) {
      call.CopyOnWrite()->args = new_args;
    }

    if (call->op.as<ExternFuncNode>() || call->op == call_dps) {
      // TODO(@altanh): assuming for now that extern/TIR calls will be typed by a variable binding
    } else if (call->op.as<OpNode>()) {
      // look up registered type inference function for the op
      Op op = Downcast<Op>(call->op);
      if (op_inference_map.count(op)) {
        new_ty = op_inference_map[op](call, diag_ctx_);
      } else if (op == call_dps) {
        // get the type from the annotation later
      } else {
        diag_ctx_.Emit(Diagnostic::Error(call->span) << "no type inference function is registered for " << op->name);
      }
    } else if (call->op->checked_type_.defined()) {
      // TODO(@altanh): infer return type using op func type and arg types, also think about where
      //                polymorphism fits since people hate thinking about that. This is where
      //                unification would have the biggest impact I think.
      diag_ctx_.Emit(Diagnostic::Error(call->span) << "type inference for non-operator/extern calls not yet supported");
    } else {
      diag_ctx_.Emit(Diagnostic::Error(call->span) << "failed to infer type of function being called");
    }

    return UpdateType(call, new_ty);
  }

  ObjectRef VisitNode_(const TupleNode* op) override {
    Tuple tuple = GetRef<Tuple>(op);

    bool same_fields = true;
    bool failed = false;
    Array<Expr> new_fields;
    Array<Type> field_types;
    for (const Expr& field : tuple->fields) {
      Expr new_field = InferExpr(field);
      if (!new_field->checked_type_.defined()) {
        diag_ctx_.Emit(Diagnostic::Error(field->span) << "failed to infer type of tuple field");
        failed = true;
      }
      same_fields &= new_field.same_as(field);
      new_fields.push_back(new_field);
      field_types.push_back(new_field->checked_type_);
    }

    if (failed) {
      diag_ctx_.Emit(Diagnostic::Error(tuple->span)
                     << "failed to infer type of tuple, as some field types could not be inferred");
      return tuple;
    } else if (!same_fields) {
      tuple.CopyOnWrite()->fields = new_fields;
    }

    return UpdateType(tuple, TupleType(field_types));
  }

  ObjectRef VisitNode_(const VarNode* op) override {
    Type new_ty;
    if (op->type_annotation.defined()) {
      new_ty = op->type_annotation.value();
    }
    return UpdateType(GetRef<Var>(op), new_ty);
  }

  ObjectRef VisitNode_(const DataflowVarNode* op) override {
    Type new_ty;
    if (op->type_annotation.defined()) {
      new_ty = op->type_annotation.value();
    }
    return UpdateType(GetRef<DataflowVar>(op), new_ty);
  }

  ObjectRef VisitNode_(const GlobalVarNode* op) override {
    GlobalVar gvar = GetRef<GlobalVar>(op);
    if (mod_->functions.find(gvar) == mod_->functions.end()) {
      diag_ctx_.Emit(Diagnostic::Error(gvar->span)
                     << "no global variable named \"" << gvar->name_hint << "\" in module");
      return gvar;
    }
    BaseFunc func = mod_->Lookup(gvar);
    if (!func.as<FunctionNode>()) {
      // PrimFunc
      return gvar;
    }
    return UpdateType(gvar, InferExpr(Downcast<Function>(func))->checked_type_);
  }

  ObjectRef VisitNode_(const IfNode* op) override {
    If ite = GetRef<If>(op);

    Expr cond = InferExpr(ite->cond);
    Expr true_b = InferExpr(ite->true_branch);
    Expr false_b = InferExpr(ite->false_branch);

    if (!cond->checked_type_.defined()) {
      diag_ctx_.Emit(Diagnostic::Error(cond->span) << "failed to infer the type of if condition");
      return ite;
    }

    if (const DynTensorTypeNode* tty = cond->checked_type_.as<DynTensorTypeNode>()) {
      // TODO(@altanh): check that this is the tensor type we need
      if (!tty->IsUnknownRank() && tty->rank != 0) {
        diag_ctx_.Emit(Diagnostic::Error(ite->span)
                       << "if condition should be a rank-0 (scalar) boolean tensor, but got rank "
                       << std::to_string(tty->rank));
        return ite;
      }
      if (!tty->IsUnknownDtype() && !tty->dtype.is_bool()) {
        diag_ctx_.Emit(Diagnostic::Error(ite->span)
                       << "if condition should be a rank-0 (scalar) boolean tensor, but got dtype "
                       << runtime::DLDataType2String(tty->dtype));
        return ite;
      }
    }

    Type true_ty = true_b->checked_type_;
    Type false_ty = false_b->checked_type_;

    // check that both branches have the same type, or try generalizing in the case of DynTensors
    if (!true_ty.defined()) {
      diag_ctx_.Emit(Diagnostic::Error(true_b->span)
                     << "could not infer a type for the true branch");
      return ite;
    }
    if (!false_ty.defined()) {
      diag_ctx_.Emit(Diagnostic::Error(false_b->span)
                     << "could not infer a type for the false branch");
      return ite;
    }

    Type new_ty;
    if (!StructuralEqual()(true_ty, false_ty)) {
      if (true_ty.as<DynTensorTypeNode>() && false_ty.as<DynTensorTypeNode>()) {
        // generalize the tensor type (if necessary)
        DynTensorType true_tty = Downcast<DynTensorType>(true_ty);
        DynTensorType false_tty = Downcast<DynTensorType>(false_ty);
        int ret_rank = true_tty->rank == false_tty->rank ? true_tty->rank : -1;
        DataType ret_dtype =
            true_tty->dtype == false_tty->dtype ? true_tty->dtype : DataType::Void();
        new_ty = DynTensorType(ret_rank, ret_dtype);
      } else {
        this->diag_ctx_.Emit(
            Diagnostic::Error(ite->span)
            << "incompatible types for true and false branches: they must match or be tensors");
        return ite;
      }
    } else {
      new_ty = true_ty;
    }

    // update node if necessary
    if (!cond.same_as(ite->cond)) {
      ite.CopyOnWrite()->cond = cond;
    }
    if (!true_b.same_as(ite->true_branch)) {
      ite.CopyOnWrite()->true_branch = true_b;
    }
    if (!false_b.same_as(ite->false_branch)) {
      ite.CopyOnWrite()->false_branch = false_b;
    }

    return UpdateType(ite, new_ty);
  }

  ObjectRef VisitNode_(const OpNode* op) override {
    // TODO(@altanh): this good?
    return GetRef<Op>(op);
  }

  ObjectRef VisitNode_(const TupleGetItemNode* op) override {
    TupleGetItem pi = GetRef<TupleGetItem>(op);

    Expr new_tuple = InferExpr(pi->tuple);
    if (!new_tuple.same_as(pi->tuple)) {
      pi.CopyOnWrite()->tuple = new_tuple;
    }

    Type new_ty;
    if (new_tuple->checked_type_.defined()) {
      if (const TupleTypeNode* tup_ty = new_tuple->checked_type_.as<TupleTypeNode>()) {
        new_ty = tup_ty->fields[pi->index];
      } else {
        diag_ctx_.Emit(Diagnostic::Error(new_tuple->span)
                       << "only tuples can be projected, but got " << new_tuple->checked_type_);
      }
    } else {
      diag_ctx_.Emit(Diagnostic::Error(new_tuple->span)
                     << "failed to infer tuple projection type, as the projected expression could "
                        "not be typed");
    }

    return UpdateType(pi, new_ty);
  }

  ObjectRef VisitNode_(const ShapeExprNode* op) override {
    return UpdateType(GetRef<ShapeExpr>(op), ShapeType(Span()));
  }

  ObjectRef VisitNode_(const SeqExprNode* op) override {
    SeqExpr seq = GetRef<SeqExpr>(op);

    bool same_blocks = true;
    Array<BindingBlock> new_blocks;
    for (BindingBlock block : seq->blocks) {
      BindingBlock new_block = Downcast<BindingBlock>(VisitNode(block));
      same_blocks &= new_block.same_as(block);
    }

    Expr new_body = InferExpr(seq->body);
    if (!new_body->checked_type_.defined()) {
      // FIXME(@altanh): just realized that checking InferExpr(...)->checked_type_.defined() may not
      //                 indicate failure as the original expr is returned (which could be typed).
      //                 Either return a new Expr (with undefined type) on failure or figure out
      //                 some other way. Alternatively the current behavior is not totally wrong as
      //                 the emitted error will be reported at the end (indicating failure).
      diag_ctx_.Emit(Diagnostic::Error(new_body->span)
                     << "failed to infer a type for the body of the SeqExpr");
      return seq;
    }

    if (!same_blocks) {
      seq.CopyOnWrite()->blocks = new_blocks;
    }
    if (!new_body.same_as(op->body)) {
      seq.CopyOnWrite()->body = new_body;
    }

    return UpdateType(seq, new_body->checked_type_);
  }

  ObjectRef VisitNode_(const FunctionNode* op) {
    Function func = GetRef<Function>(op);

    bool same_params = true;
    Array<Var> new_params;
    Array<Type> new_param_types;
    for (Var param : func->params) {
      Var new_param = Downcast<Var>(InferExpr(param));
      if (!new_param->checked_type_.defined()) {
        diag_ctx_.Emit(Diagnostic::Error(param->span)
                       << "function parameter types must be annotated");
        return func;
      }
      same_params &= new_param.same_as(param);
      new_params.push_back(new_param);
      new_param_types.push_back(new_param->checked_type_);
    }

    Expr new_body = InferExpr(func->body);
    if (!new_body->checked_type_.defined()) {
      diag_ctx_.Emit(Diagnostic::Error(new_body->span)
                     << "failed to infer a type for the function body");
      return func;
    } else if (func->ret_type.defined() &&
               !StructuralEqual()(func->ret_type, new_body->checked_type_)) {
      diag_ctx_.Emit(Diagnostic::Error(func->span)
                     << "mismatch between inferred and annotated function return type");
      return func;
    }

    if (!same_params) {
      func.CopyOnWrite()->params = new_params;
    }
    if (!new_body.same_as(op->body)) {
      func.CopyOnWrite()->body = new_body;
    }

    return UpdateType(func, FuncType(new_param_types, new_body->checked_type_, {}, {}));
  }

  ObjectRef VisitNode_(const ExternFuncNode* op) { return GetRef<ExternFunc>(op); }

  ObjectRef VisitNode_(const MatchShapeNode* op) {
    MatchShape match = GetRef<MatchShape>(op);

    Var new_var = match->var;
    if (new_var.defined()) {
      new_var = Downcast<Var>(InferExpr(new_var));
    }
    Expr new_value = InferExpr(match->value);

    if (!new_value->checked_type_.defined()) {
      diag_ctx_.Emit(Diagnostic::Error(new_value->span)
                     << "could not infer a type for the value being shape matched");
      return match;
    }

    Type refined_type = new_value->checked_type_;
    if (const DynTensorTypeNode* tty = refined_type.as<DynTensorTypeNode>()) {
      // we can try to refine the rank of the tensor using the matched pattern
      if (tty->IsUnknownRank()) {
        refined_type = DynTensorType(match->pattern.size(), tty->dtype);
      } else if (static_cast<size_t>(tty->rank) != match->pattern.size()) {
        // error case:
        //   x0: Tensor[(n, m), _] = ...
        //   x1: Tensor[(n, m, k), _] = match_shape(x0, (n, m, k))
        diag_ctx_.Emit(Diagnostic::Error(match->span)
                       << "rank mismatch in match shape refinement: " << tty->rank << " vs "
                       << match->pattern.size());
        return match;
      }
    }

    if (new_var.defined()) {
      if (new_var->checked_type_.defined() &&
          !StructuralEqual()(new_var->checked_type_, refined_type)) {
        diag_ctx_.Emit(Diagnostic::Error(match->span)
                       << "mismatch between inferred and annotated variable type");
        return match;
      } else {
        new_var = UpdateType(new_var, refined_type);
      }
    }

    // now guaranteed that new_var (if defined) and new_value have the same type

    if (!new_var.same_as(match->var)) {
      match.CopyOnWrite()->var = new_var;
    }
    if (!new_value.same_as(match->value)) {
      match.CopyOnWrite()->value = new_value;
    }

    return match;
  }

  ObjectRef VisitNode_(const VarBindingNode* op) {
    static const Op& call_dps = Op::Get("relax.call_dps");

    VarBinding binding = GetRef<VarBinding>(op);

    Var new_var = Downcast<Var>(InferExpr(binding->var));
    Expr new_value = InferExpr(binding->value);

    if (!new_value->checked_type_.defined()) {
      if (new_var->checked_type_.defined() && new_value.as<CallNode>()) {
        // use the type of new_var in the case of extern calls
        Call call = Downcast<Call>(new_value);
        ICHECK(call->op.as<ExternFuncNode>() || call->op == call_dps);
        new_value = UpdateType(Downcast<Call>(new_value), new_var->checked_type_);
      } else {
        diag_ctx_.EmitFatal(Diagnostic::Error(binding->span) << "failed to infer type of binding");
        return binding;
      }
    } else if (new_var->checked_type_.defined() &&
               !StructuralEqual()(new_var->checked_type_, new_value->checked_type_)) {
      diag_ctx_.Emit(Diagnostic::Error(binding->span)
                     << "mismatch between inferred and annotated variable type: inferred type is " << new_value->checked_type_);
      return binding;
    } else if (!new_var->checked_type_.defined()) {
      new_var = UpdateType(new_var, new_value->checked_type_);
    }

    if (!new_var.same_as(binding->var)) {
      binding.CopyOnWrite()->var = new_var;
    }
    if (!new_value.same_as(binding->value)) {
      binding.CopyOnWrite()->value = new_value;
    }

    return binding;
  }

  ObjectRef VisitNode_(const BindingBlockNode* op) {
    BindingBlock block = GetRef<BindingBlock>(op);

    bool same_bindings = true;
    Array<Binding> new_bindings;
    for (Binding binding : block->bindings) {
      Binding new_binding = Downcast<Binding>(VisitNode(binding));
      same_bindings &= new_binding.same_as(binding);
      new_bindings.push_back(new_binding);
    }

    if (!same_bindings) {
      ICHECK(!block.as<DataflowBlockNode>());
      return BindingBlock(new_bindings, block->span);
    }

    return block;
  }

  ObjectRef VisitNode_(const DataflowBlockNode* op) {
    DataflowBlock block = GetRef<DataflowBlock>(op);

    bool same_bindings = true;
    Array<Binding> new_bindings;
    for (Binding binding : block->bindings) {
      Binding new_binding = Downcast<Binding>(VisitNode(binding));
      same_bindings &= new_binding.same_as(binding);
      new_bindings.push_back(new_binding);
    }

    if (!same_bindings) {
      return DataflowBlock(new_bindings, block->span);
    }

    return block;
  }

 private:
  template <typename T>
  T UpdateType(T expr, const Type& new_type) {
    if (!new_type.defined()) {
      // case: type inference failed, do nothing
    } else if (!expr->checked_type_.defined()) {
      // case: idempotently write directly into expression
      expr->checked_type_ = new_type;
    } else if (!StructuralEqual()(expr->checked_type_, new_type)) {
      // case: copy and write if old type doesn't match
      expr.CopyOnWrite()->checked_type_ = new_type;
    }
    return expr;
  }

  IRModule mod_;
  DiagnosticContext diag_ctx_;
  InferenceMode mode_;
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> expr_memo_;
};

TVM_REGISTER_GLOBAL("relax.transform.type_inference").set_body_typed([](IRModule mod, int mode) {
  TypeInferencer inferencer(mod, DiagnosticContext::Default(mod),
                            static_cast<TypeInferencer::InferenceMode>(mode));
  return inferencer.InferMod();
});

}  // namespace relax
}  // namespace tvm