# Notes: Shapes in Relax
mbs@octoml.ai  
26-Jul-2021

Part of the Relax project. Reference docs:
* [Relax main page](https://www.notion.so/octoml/RelaX-The-Next-Generation-Training-and-Compiler-28de4a6043d84801b7aaf0aff5839904)
* [Shape constraints Design Document](https://www.notion.so/octoml/Shape-Constraints-Design-Document-b1cd8367d3ce4916a4ab478a0055df00)
These are my notes to test my understanding and get ahead of any gotcha's.

### Goals

* Execute models with shapes which may be difficult or impossible to infer at
  compile time, even if slowly. In particular, execute any ONYX model.
* Support gracefully improving execution efficiency over time by better shape
  analysis to discover shape invariants, which in turn can be exploited by
  inlining and monomorphisation.
* Similarly, improve the ability to reject ill-shaped models at compile time
  with clear diagnostics.
* Allow shape invariants to be captured in code, both in primitives and user
  definitions.
* Subsume existing type relation and dynamic shape function mechanisms.
* Allow the user to see where residual dynamic shapes are impacting their
  model's performance.

### Motivating example

Imagine a 'fast flatten':
```
@ff = fn(x : Tensor([n, m * 4], int8)) : Tensor([n * m], int32) {
  <body>
}
```
Here `<body>` can be Relax, TE, TIR, or we can consider `@ff` a primitive or
extern. The type signature denotes that:
* `@ff` takes a rank-2 tensor of `int8`s and returns a rank-1 tensor of `int32`.
* Dimension 0 of the argument tensor is arbitrary.
* Dimension 1 of the argument tensor must be divisible by 4.
* Dimension 0 of the result tensor is the product of dimensions 0 and 1 of the
  argument tensor, divided by 4. 

* **Test A**: Easy to propagate ground shapes across calls at compile time:
  ```
  let a : Tensor([8, 16], int8) = ...
  let b = @ff(a)
  # b : Tensor([32], int32)
  ```

* **Test B**: Easy to detect 'obviously' ill-shaped expressions at 
  compile time:
  ```
  let a : Tensor([8, 15], int8) = ...
  let b = @ff(a)
  # Shape error: In dimension 1 of argument x to @@ff, m / 4 = 15 has no solution.
  ```

* **Test C**: Easy to propagate non-ground shapes across calls at compile time
and maintain tensor shapes in simplified form: 
  ```
  let a : Tensor([k, 16], int8) = ...
  let b = @ff(a)
  # b : Tensor([k * 4], int32)
  ```

* **Test D**: Easy to separate 'shape' and 'data' computations in TIR (to support DPS):
  ```
  @ff_tir_shape = fn(x : shape, out : shape) {
    %0, %1 = match_shape_2(x)  # implicitly asserts x is rank 2
    %2 = div(%1, 4)
    %3 = mul(%0, %2)
    build_shape_1(out, %3)
  }
  @ff_tir = fn(x : handle, out : handle) { ... }

  @ff(...)
  ==>
  %1 = get_shape(%0)
  %2 = alloc_shape_1()
  %3 = @ff_tir_shape(%1, %2)
  %4 = alloc_buffer(...)
  %5 = alloc_tensor(%3, int32)
  @ff_tir(..., %5)
  ```

* **Test E**: User can assert tensor shapes have a particular form. The
assertion is allowed to fail at runtime (but may be shown to always fail
at compile time as per B).
  ```
  let a : Tensor(?, ?) = ...
  let b ! Tensor([k * 64], int32) = @ff(a)
  # b : Tensor([k * 64], int32) for some k only known at runtime
  ```

* **Test F**: Shapes and their dimensions do not leak into general
  Relay expressions:
  ```
  @ff = fn(x : Tensor([n, m * 4], int8)) : Tensor([n * m], int32) {
    # n, m in scope and can be used in tensor type shapes
    let y ! Tensor([n * 2, k], int32)) = ...       # valid!
    # k now also in scope 
    # But those vars cannot be used in general Relay expressions
    # let x = n * m * k                            # ill-formed!
  }
  ```
  We don't want to force all targets to support scalar size/index calculations (yet). 

* **Test G**: Data dependent shapes supported:
  ```
  @filter = fn(x : Tensor(?, d)) : Tensor(?, d) { ... } 
  ```
  Unless argument is a literal tensor we can't say anything about the
  result shape at compile time. But we still need to separate the output shape
  computation from the data computation for runtime.

* **Test H**: Can express shape propagation rules 'extra linguistically'
  (eg for broadcast rules) while still supporting A-D above.
  
* **Test I**: Relay AST has clear and unambiguous representation for all this.
  Eg desugar to:
  ```
  @ff = fn(x : DynTensor) : DynTensor {
    match_shape(x, [n, m], [n, m * 4]);  # binds n&m, fail if pattern match fails 
    let y = ...
    match_shape(y, [k], [n*2, k]);       # binds k, fail if pattern match fails
    let out = ...
    match_shape(out, [], [n * m]);       # no bindings, fail if pattern match fails
  }
  ```

* **Test J**: Migration path for existing type relations and dynamic shape functions.

**Unresolved**:
* Include dtypes in dynamic world?
* Shape variables (ie express full shape polymorphism)?

-----------
[old]


### Main design decisions

* Support shape and dtype access with Relay/TIR primitives? (eg `e.rank`,
  `e.dim(i)`, `e.dtype`. Or force shape/dtype introspection to use a
  dedicated language construct? (eg `match_shape(e, [x, y, z])`).
  
  => Propose a very restricted form of the former.
  
* Replace existing type relations (static) and shape functions (dynamic) with
  single mechanism? 

  => Propose yes, but still allow rules for shape propagation through primitives
     to be hard coded in the compiler (eg for broadcast semantics).

* Replace existing use of 'size vars' in types?

  => Propose yes.
  
* How to allow user-supplied annotations on shape, both for function arguments
  and on arbitrary sub-expressions?
  
  => Propose syntactic sugar on `Tensor` types, including implicit binding and
     matching. In particular, shape-related assertions can only be associated
     with a binding site of a variable of tensor type, which greatly simplifies
     things.
  
* How to flow shape information from call arguments to call result?

  => Propose all user-defined functions and primitives have an outer, inlinable
     'shape checking' wrapper which invokes the actual implementation. The 
     actual implementation may assume all shape assertions hold.

* Should dtypes be dynamic?

  => Propose yes.

* Can shape assertions deal with polymorphic rank?

  => Propose in general no. However primitives may have hard coded shape
     propagation.

* Can shapes/dtypes be observed by user Relay expressions?

  => Propose yes, but with only limited primitives. Eg batch size as an
     argument which turns up in shapes.

* Support shape constraints on function arguments? ADT constructors?

  => Propose no for now.

### Non-goals

* "Well-shaped programs don't go wrong" is not a goal: it is possible for a
  shape-related assertion to fail at runtime.
* "All shape checks compiled away" is not a goal: it is possible for residual
  shape-related assertions to remain.
* "All shape invariants captured in types" is not a goal: it is possible for
  the user to write shape-related assertions directly, and the types are merely
  a convenient-but-incomplete shorthand.

### 'Sweet' syntax

The user-visible syntax has shorthands for common shape-related assertions in
types:
```
stype ::= Tensor                # All tensors of all ranks, dimensions and dtypes
        | Tensor(shape, dtype)  # Tensor with implied rank, dimensions and dtype
        | DType                 # Type of tensor dtypes
        | Size                  # Type of tensor ranks and dimensions (signed)
        | Shape                 # Type of tensor shapes (array of Sizes)
        | fn (stype, ...) : stype
        | ...
shape ::= ?                     # Unknown shape of unknown rank
        | x                     # expr var of type Shape
        | [dim_1, ..., dim_n]   # Dimensions of rank n tensor
dim   ::= ?                     # Unknown dimension
        | x                     # expr var of type Size
        | <size>                # literal Size
dtype ::= ?                     # Unknown DType
        | x                     # expr var of type DType
        | <dtype>               # literal DType
sexpr ::= value
        | fn (x : stype, ...) : stype { sexpr }
        | let x : stype = sexpr; sexpr
        | assert sexpr          # explicit assertion
        | x.rank                # the rank of tensor bound to x
        | x.dim(<size>)         # the dimension of tensor bound to x on axis
        | x.dtype               # the dtype of tensor bound to x
        | ...
value ::= <tensor>              # Literal Tensor
        | <size>                # Literal Size
        | [<size>, ..., <size>] # Literal Shape
        | <dtype>               # Literal DType
        | ...
```

Tensor types in the 'sweet' syntax have a 'shape' (an array of dimensions with
size equal to the tensor rank) and a dtype (a member of a fixed enum of supported
tensor datatypes). We don't use rank-1 tensors to represent shapes since it
hurts my head too much. Shapes can only be manipulated by the compiler, we do
not make them first-class in relay. Type vars cannot appear in Tensor types.
Expression vars may appear in dimensions, shapes and dtypes, but we'll desugar
them away below.

The stype shorthand allows code to be specific to a particular rank, or be
rank agnostic, but nothing in-between. That is we cannot express something like
'tensor has rank at least 2 and first and second dimensions are 256'.

We restrict use of the `.rank`, `.dim(<size>)` and `.dtype` primitives to
be applied to variables. This will allow us to substitute these expressions
during desugaring below. We also restrict `.dim(<size>)` to use only a literal
axis. Thus all three of these primitives can be considered uninterpreted
functions when simplifying assertions.

Explicit assertions can capture much more than what's expressible in the
stypes:
```
assert x.dim(0) % 4 == 0;
assert y.dim(1) <= 256;
assert out.dim(0) == x.dim(0) * y.dim(1);
```
We allow these in Relay.

### 'Desugared' syntax

Desugaring is an stype-directed translation which replaces all occurrences of
stype `Tensor(dims, dtype)` with the type `Tensor`, and moves any invariants
implied by the stype into the expr in the form of asserts.

```
type   ::= Tensor       # no more Tensor(dims, dtype) form
         | ...
expr   ::= let x : type = expr; expr
         | ...

env    ::= [x |-> expr, ...] 

desugar : (env, stype) |-> (type, expr)
```
Here `env` tracks substitutions for expr variables appearing in types.

Function definitions are split into an inlinable shape-check wrapper and
a private implementation. (Primitives have no implementation.)

Eg:
```
@matmul = fn (x : Tensor([m, k], d), y : Tensor([k, n], d) : Tensor([m, n], d) {
  <body>
}
==>
# @matmul is rewritten to assert all the shape constraints. This is inlinable
# at call sites.
inline @matmul = fn (x : Tensor, y : Tensor) : Tensor {
   assert x.rank == 2;
   # m, k and d are free when desugaring x's type annotation, so we capture:
   #   m |-> x.dim(0)
   #   k |-> x.dim(1)
   #   d |-> x.dtype
   assert y.rank == 2;
   # k and d are bound when desugaring y's type annotation, so we subsitute
   # their bindings and add assertions.
   assert y.dim(0) == x.dim(1);
   assert y.dtype = x.dtype;
   # n is free when desugaring y's type annotation, so we capture:
   #   n |-> y.dim(1)
   let out = @matmul_impl(x, y);  # invoke the original implementation
   assert out.rank == 2;
   # m, n and d are  bound when desugaring the result type annotation.
   assert out.dim(0) == x.dim(0);
   assert out.dim(1) == y.dim(1);
   assert out.dtype == x.dtype;
   out
}
# The original body need not be inlinable, and can only be called from @matmul.
private @matmul_impl = fn (x : Tensor, y : Tensor) : Tensor {
  <body>
}
```

Since most analysis code will want easy access to 'the' shape and dtype of
intermediate expressions of `Tensor` type we support those as fields on every
AST node (cf checked_type, which will now hold only the desugared types). We
also allow as 'is_assertion' boolean field to indicate when an AST node's
shape/dtype is an assertion which may fail at runtime, as opposed to an
irrefutable statement. With that we can represent the above much more
concisely:
```
inline @matmul = fn (x : Tensor, y : Tensor) : Tensor {
  # x's shape = [x.dim(0), x.dim(1)]
  # x's dtype = x.dtype 
  # x's is_assertion = true
  # y's shape = [x.dim(0), y.dim(1)]
  # y's dtype = x.dtype
  # y's is_assertion = true
  let out = @matmul_impl(x, y);
  # out's shape = [x.dim(0), y.dim(1)]
  # out's dtype = x.dtype
  # out's is_assertion = true
  out
}
```

Note that the implied assertions are over the entire shape and dtype. It is
not possible to express that, eg, only a particular dimension needs to be
asserted.

It is possible to set is_assertion = false when the implied assertions is
tautological after simplification or by construction.

### Propagation of shape information through primitives

We retain the existing escape hatch for explicit shape computation. A shape
function takes exprs denoting the arguments (with shape field filled in) and
returns an expr denoting the shape of the result. If the arguments and their
shapes are sufficiently concrete then the result shape expression may be
similarly concrete.

E.g. for:
```
let x : Tensor([3, 5], int8) = ...
let y : Tensor([3, 2], int8) = ...
let z : Tensor([3, 7], int8) = @concat([x, y], axis=1)
```
the shape function for `@concat` will return `[3, 7]` and thus the assertions
implied by the type annotation on z are vacuous.

In general however the result expression will not be concrete and shape
calculation code will remain at runtime.

### Simplification

We rely on the simplifier to constant propagate and reduce shape-related
expressions so as to:
* Eliminate tautological assertions (`assert true`).
* Reject the program at compile time with a sensible diagnostic if all possible
  executions will assert fail (`assert false`).
* Replace general primitives with specific primitives when sound to do so
  for all possible executions.    

Generally simplification is w.r.t. entailment of assumed-true assertions
already accumulated from the context:
```
env ::= [assert expr; ...]
simplify : (env, expr) |-> expr
```
Eg:
```
simplify([assert x == 3], (assert x > 2; assert y < 3)) ==> (assert y < 3)
```
We're at the mercy of the integer constraint solver to help us for anything
non-trival. However it's ok to leave residual assertions in the generated
code and we're not bound to any form of completeness.

### Shape propagation

Shape propagation proceeds by inlining function wrappers at call
sites and simplifying. Ignoring the AST shape fields and just using
annotations we have:

```
shape_prop(@matmul(left, right))
==>
shape_prop(left) [out |-> x];
shape_prop(right) [out |-> y];
shape_prop(@matmul)
```

Let's assume `left` is a literal (2, 3) tensor and `right` a literal (3, 4)
tensor, both of dtype `int8`. Then:
```
shape_prop(left) [out |-> x]
==>
assert x.rank == 2;
assert x.dim(0) == 2;
assert x.dim(1) == 3;
assert x.dtype = int8;
# ie the shape AST field after expansion being [2, 2]
```
and
```
shape_prop(right) [out |-> y]
==>
assert y.rank == 2;
assert y.dim(0) == 3;
assert y.dim(1) == 4;
assert y.dtype = int8;
# ie the shape AST field after expansion is [3, 4]
```

Thus we'll end up with a shape expression:
```
assert x.rank == 2;
assert x.dim(0) == 2;
assert x.dim(1) == 3;
assert x.dtype = int8;
assert y.rank == 2;
assert y.dim(0) == 3;
assert y.dim(1) == 4;
assert y.dtype = int8;
assert x.rank == 2;
assert y.rank == 2;
assert y.dim(0) == x.dim(1);
assert y.dtype = x.dtype;
let out = @matmul_impl(x, y);
assert out.rank == 2;
assert out.dim(0) == x.dim(0);
assert out.dim(1) == y.dim(1);
assert out.dtype == x.dtype;
out
```

By modus ponens of the assertions (`x.rank`, `x.dim(0)` etc can be treated
as uninterpreted functions):
```
assert x.rank == 2;
assert x.dim(0) == 2;
assert x.dim(1) == 3;
assert x.dtype = int8;
assert y.rank == 2;
assert y.dim(0) == 3;
assert y.dim(1) == 4;
assert y.dtype = int8;
assert 2 == 2;
assert 2 == 2;
assert 3 == 3;
assert int8 = int8;
let out = @matmul_impl(x, y);
assert out.rank == 2;
assert out.dim(0) == 2;
assert out.dim(1) == 4;
assert out.dtype == int8;
out
```
The vacuous assertions can then be removed.

That expr is ready for substitution into the shape expression for the next
containing context.

Using the AST shape fields this is more compact, and we end up with the overall
expressio's shape field bound to the Shape literal `[2, 4]` and dtype bound to
the `int8` DType literal.

### Exploiting shape propagation

Overall we can eliminate tautological asserts at runtime.

We can replace general primitives with shape and/or dtype specialized
implementation:
```
@matmul(left, right)
==>
@matmul_16_16_int8(left, right)
```
if
```
shape_prop({left, right}) |- assert out.rank == 2;
                             assert out.dim(0) == 16;
                             assert out.dim(1) == 16;
                             assert out.dtype == int8
```
(?) @matmul can be bound to a list of possible implementations, from most-
to least-specific order. Ie if an earlier implementation is assertion-fail free
then so are all later implementations. Users can plug in specialized impls,
using assertions (or stypes) to quality their domain. Target devices could
be part of this.

We can replace runtime dispatch with inlined calls:
```
assert x.rank == 2 |- if (x.rank == 2) {
                        <body1>
                      } else {
                        <body2>
                      }
==>
<body1>
```

### Asserts under control flow

The shape_prop transformation does not need to preserve all dataflow
dependence. Eg:
```
@f = fn(x : int8, y : Tensor([3], int8)) : Tensor {
  if (x > 3) {
    @concat(y, y)
  } else {
    y
  }
}
```
would be desugared to:
```
@f = fn(x : int8, y : Tensor) : Tensor {
  assert y.rank = 1;
  assert y.dim(0) == 3;
  assert y.dtype == int8;
  let out = @f_impl(x, y);
  out
}
@f_impl = fn(x : int8, y : Tensor) : Tensor {
  if (x > 3) {
    @concat(y, y)
  } else {
    y
  }
}
```
Thus the dependence of the output shape on x is lost.

Alternatively, we could preserve some control flow in the wrapper:
```
@f = fn(x : int8, y : Tensor) : Tensor {
  assert y.rank = 1;
  assert y.dim(0) == 3;
  assert y.dtype == int8;
  let out = if (x > 3) {
    @concat(y, y)
  } else {
    y
  }
  out
```
(There's no `@f_impl` left in this example.)

After shape propagation:
```
@f = fn(x : int8, y : Tensor) : Tensor {
  assert y.rank = 1;
  assert y.dim(0) == 3;
  assert y.dtype == int8;
  let out = if (x > 3) {
    let out' = @concat(y, y)
    assert out'.rank == 1;
    assert out'.dim(0) = 6;
    assert out'.dtype == int8
    out'
  } else {
    let out' = y
    assert out'.rank == 1;
    assert out'.dim(0) == 3;
    assert out'.dtype == int8
    out'
  }
  out
```
A call `@f(4, <literal tensor of [3]>` could then have shape expression
`[1, 3]`.

However the common case will be conditionals with consistent shapes:
```
@f = fn(x : int8, y: Tensor(s, int8)) : Tensor(s, int8) {
  if (x > 3) {
    @add(y, x)   # with broadcast
  } else {
    @add(y, y)
  }
}
```
Desugared as:
```
@f = fn(x : int8, y : Tensor) : Tensor {
  assert y.rank = 1;
  assert y.dim(0) == 3;
  assert y.dtype == int8;
  let out = if (x > 3) {
    @add(y, x)
  } else {
    @add(y, y)
  }
  assert out.rank = 1;
  assert out.dim(0) == 3;
  assert out.dtype == int8
  out 
}
```
and the body shape expression is simplified to:
```
assert y.rank = 1;
assert y.dim(0) == 3;
assert y.dtype == int8;
let out = if (x > 3) {
  let out' = any
  assert out'.rank == 1
  assert out'.dim(0) == 3;
  assert out'.dtype == int8
  out'
} else {
  let out' = any
  assert out'.rank == 1
  assert out'.dim(0) == 3;
  assert out'.dtype == int8
  out'
}
assert out.rank = 1;
assert out.dim(0) == 3;
assert out.dtype == int8
out 
```
But since each arm of the conditional entail the other we have:
```
assert y.rank = 1;
assert y.dim(0) == 3;
assert y.dtype == int8;
let out = any
assert out.rank == 1
assert out.dim(0) == 3;
assert out.dtype == int8
assert out.rank = 1;
assert out.dim(0) == 3;
assert out.dtype == int8
out 
```
thus:
```
let out = any
assert out.rank == 1
assert out.dim(0) == 3;
assert out.dtype == int8
out
```

### Compile-time shape errors

An obvious 'shape error' should show up as an `assert false`. Eg:
```
@matmul(<literal tensor of shape (2, 3)>, <literal tensor of shape (4, 3)>))
==>
assert false # x.dim(1) != y.dim(0)
```

That's all well and good but we want to make sure the 'assertion will always
fail' diagnostic directs the user to the original call sexpr and not some
intermediate term they've never seen. For this we can keep track of lexical
scopes of original program Spans and somehow report failures w.r.t. the original
stypes.
```
Shape mismatch while infering the shapes of expression
  @matmul(<literal>, <literal>)
The function
  @matmul
expects a second argument of type
  Tensor([3, n], int8)
but is being called with an argument of type
  Tensor([4, 3], int8)
```
That's pretty challenging.
