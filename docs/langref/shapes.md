# Notes: Shapes in Relax
mbs@octoml.ai  
26-Jul-2021

Part of the Relax project. Reference docs:
* [Relax main page](https://www.notion.so/octoml/RelaX-The-Next-Generation-Training-and-Compiler-28de4a6043d84801b7aaf0aff5839904)
* [Shape constraints Design Document](https://www.notion.so/octoml/Shape-Constraints-Design-Document-b1cd8367d3ce4916a4ab478a0055df00)


### Goals

* Execute models with shapes which may be difficult or impossible to infer at
  compile time, even if slowly. In particular, execute any ONYX model.
* Support gracefully improving execution efficiency over time by better shape
  analysis to discover shape invariants.
* Similarly, improve the ability to reject ill-shaped models at compile time
  with clear diagnostics.
* Allow shape invariants to be captured in code, both in primitives and user
  definitions.
* (?) Allow the user to see where residual dynamic shapes are impacting their
  model's performance. 

### Non-goals

* "Well-shaped programs don't go wrong" is not a goal: it is possible for a
  shape-related assertion to fail at runtime.
* "All shape checks compiled away" is not a goal: it is possible for residual
  shape-related assertions to remain.
* "All shape invariants captured in types" is not a goal: it is possible for
  the user to write shape-related assertions directly, and the types are merely
  a convenient but incomplete shorthand.

### 'Sweet' syntax

The user-visible syntax has shorthands for common shape-related assertions:
```
stype ::= Tensor               # All tensors of all ranks, dimensions and dtypes
        | Tensor(dims, dtype)  # Tensor with implied rank, dimension and dtype
        | DType                # Type of tensor dtypes
        | Int                  # Type of tensor ranks and dimensions
        | Array                # Type of arrays of Ints
        | ...
dims  ::= x                    # expr var of stype Array 
        | [dim_1, ..., dim_n]  # Array of size n 
dim   ::= x                    # expr var of type Int
        | <int>                # literal int
dtype ::= x                    # expr var of type DType
        | int8
        | ...
sexpr ::= value
        | fn (x : stype, ...) : stype { sexpr }
        | let x : stype = sexpr; sexpr
        | assert sexpr         # explicit assertion
        | out                  # the distinguished name for the sexpr result
        | sexpr.shape          # the dimensions of sexpr tensor
        | sexpr.dtype          # the dtype of sexpr tensor
        | sexpr.size           # the size of sexpr array
        | sexpr.at(sexpr)      # the entry at sexpr int in sexpr array
        | ...
value ::= <tensor>
        | [<int>, ...]         # Literal Array
        | <int>                # Literal Int
        | ...
```

Tensors in the 'sweet' syntax have a 'shape' (an array of dimensions with size
equal to the tensor rank) and a dtype (a member of a fixed enum of supported
datatypes). Tensor shapes are represented by a primitive Array type rather
than a rank-1 tensor to avoid infinite regress in shape inference. Tyvars
cannot appear in Tensor types. It is possible for tensor ranks to only be known
at runtime. Both shapes and types may contain free and bound term variables.
These will be desugared to either let-bound variables or assertion expressions
below.

The stype shorthand allows code to be specific to a particular rank, or be
fully shape polymorphic, but nothing inbetween. It is possible to dispatch
on rank but not possible to express assertions on sub-arrays:
```
if x.shape.size == 2 {
  # special case for rank 2 handling
} else if x.shape.size > 2 {
  # special case which is polymorphic on the dimensions
  # above rank 2
  let m = x.shape.at(0);
  let n = x.shape.at(1);
  ...
  assert out.shape.size = x.shape.size;
  assert out.shape.at(0) == m + 3;
  assert out.shape.at(1) == n * 2;
  # assert out.shape.slice(2) == x.shape.slice(2)   # Not supported!
} else {
  # general case
}
```

A special `out` variable represents the result of a function body, and may
appear in explicit assertions.

Explicit assertions can capture much more than what's expressible in the
stypes:
```
assert x.shape.at(0) % 4 == 0;
assert y.shape.at(1) <= 256;
assert out.shape.at(0) = x.shape.at(0) * y.shape.at(1);
```

### Existing type relations and shape functions

(?) All existing type relations (type-checking time only) and shape functions
(runtime only) can be expressed using expr below.

### 'Desugared' syntax

Desugaring is an stype-directed translation which replaces all occurrences of
stype `Tensor(dims, dtype)` with the type `Tensor`, and moves any invariants
implied by the stype into the expr in the form of let-bindings and asserts.

```
type   ::= Tensor       # no more Tensor(dims, dtype) form
         | ...
expr   ::= let x : type = expr; expr
         | ...

desugar : (env, stype) |-> (type, expr)
```
where `env` tracks the bound expr variables and their types. Bindings are
entered both for the 'normal' expr binders as well as the 'implied' binders
from stypes.

Eg:
```
@matmul = fn (x : Tensor([m, k], d), y : Tensor([k, n], d) : Tensor([m, n], d) {
  <body>
}
==>
@matmul = fn (x : Tensor, y : Tensor) : Tensor {
   assert x.shape.size == 2;
   let m = x.shape.at(0);         # m, k and d are free in x's stype          
   let k = x.shape.at(1);
   let d = x.dtype; 
   assert y.shape.size == 2;
   let n = y.shape.at(1);         # n is free in y's stype
   assert y.shape.at(0) == k;     # k and d are bound in y's stype
   assert y.dtype == d;
   let out = <body>;
   assert out.shape.size == 2;
   assert out.shape.at(0) == m;   # m, n and d are bound in result stype
   assert out.shape.at(1) == n;
   assert out.dtype == d;
   out
}
```

### Simplification

We rely on the simplifier to constant propagate and reduce shape-related
expressions so as to:
* Eliminate tautological assertions (`assert true`).
* Reject the program at compile time with a sensible diagnostic if all possible
  executions will assert fail (`assert false`).
* Replace general primitives with specific primitives when sound to do so
  for all possible executions.    

### Shape propagation

We'll need an abstract interpretation to capture just the shape propagation
of an expr so it can be inlined and simplified:
```
shape_expr : expr |-> expr
```
(Perhaps desugaring can insert some syntatic hints to make this easier?)
We can use an `any` expr to represent an arbitrary computation. Eg
for `@matmul` we get the same defn as above but with `<body>` replaced by
`any`.

Shape propagation proceeds by inlining the result of `shape_expr` at call
sites with appropriate substitution of the distinguished `out` term:

```
shape_expr(@matmul(left, right))
==>
shape_expr(left) [out |-> x];
shape_expr(right) [out |-> y];
shape_expr(@matmul)
```

Let's assume `left` is a literal (2, 3) tensor and `right` a literal (3, 4)
tensor, both of dtype `int8`. Then:
```
shape_expr(left) [out |-> x]
==>
assert x.shape.size == 2;
assert x.shape.at(0) == 2;
assert x.shape.at(1) == 3;
assert x.dtype = int8;
```
and
```
shape_expr(right) [out |-> y]
==>
assert y.shape.size == 2;
assert y.shape.at(0) == 3;
assert y.shape.at(1) == 4;
assert y.dtype = int8;
```

Thus we'll end up with a shape expression:
```
assert x.shape.size == 2;
assert x.shape.at(0) == 2;
assert x.shape.at(1) == 3;
assert x.dtype = int8;
assert y.shape.size == 2;
assert y.shape.at(0) == 3;
assert y.shape.at(1) == 4;
assert y.dtype = int8;
assert x.shape.size == 2;
let m = x.shape.at(0);         # m, k and d are free in x's stype          
let k = x.shape.at(1);
let d = x.dtype; 
assert y.shape.size == 2;
let n = y.shape.at(1);         # n is free in y's stype
assert y.shape.at(0) == k;     # k and d are bound in y's stype
assert y.dtype == d;
let out = any;
assert out.shape.size == 2;
assert out.shape.at(0) == m;   # m, n and d are bound in result stype
assert out.shape.at(1) == n;
assert out.dtype == d;
out
```

By modus ponens of the assertions (.shape.size and .shape.at(n) are uninterpreted
functions):
```
assert x.shape.size == 2;
assert x.shape.at(0) == 2;
assert x.shape.at(1) == 3;
assert x.dtype = int8;
assert y.shape.size == 2;
assert y.shape.at(0) == 3;
assert y.shape.at(1) == 4;
assert y.dtype = int8;
let m = 2;
let k = 3
let d = int8
let n = 4;
assert 3 == k;
assert int8 == d;
let out = any;
assert out.shape.size == 2;
assert out.shape.at(0) == m;
assert out.shape.at(1) == n;
assert out.dtype == d;
out
```
and constant propagation:
```
assert x.shape.size == 2;
assert x.shape.at(0) == 2;
assert x.shape.at(1) == 3;
assert x.dtype = int8;
assert y.shape.size == 2;
assert y.shape.at(0) == 3;
assert y.shape.at(1) == 4;
assert y.dtype = int8;
assert 3 == 3;
assert int8 == int8;
let out = any;
assert out.shape.size == 2;
assert out.shape.at(0) == 2;
assert out.shape.at(1) == 4;
assert out.dtype == int8;
out
```
and local simplification:
```
let out = any;
assert out.shape.size == 2;
assert out.shape.at(0) == 2;
assert out.shape.at(1) == 4;
assert out.dtype == int8;
out
```

That expr is ready for substitution into the shape expression for the next
containing context.

Simplification is always w.r.t. entailment from the current assertion context
which accumulates assumed-true assertions:
```
env |- assert x == 3;
       assert x > 2
==>
env, x == 3 |- assert x > 2
==>
env, x == 2 |- []
```
We're at the mercy of the integer constraint solver to help us for anything
non-trival. However it's  ok to leave residual assertions in the generated
code and we're not bound to any form of completeness.

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
shape_expr(left) |- assert out.shape.size == 2;
                    assert out.shape.at(0) == 16;
                    assert out.shape.at(1) == 16;
                    assert out.dtype == int8
```
(?) @matmul can be bound to a list of possible implementations, from most-
to least-specific order. Ie if an earlier implementation is assertion-fail free
then so are all later implementations. Users can plug in specialized impls,
using assertions (or stypes) to quality their domain. Target devices could
be part of this.

We can replace runtime dispatch with inlined calls:
```
assert x.shape.size == 2 |- 
if (x.shape.size == 2) {
  <body1>
} else {
  <body2>
}
==>
<body1>
```

### Asserts under control flow

The shape_expr abstract interpretation does not need to preserve all dataflow
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
is desugared to:
```
@f = fn(x : int8, y : Tensor) : Tensor {
  assert y.shape.size = 1;
  assert y.shape.at(0) == 3;
  assert y.dtype == int8;
  let out = if (x > 3) {
    @concat(y, y)
  } else {
    y
  }
  out
}
```

The shape expression for the body could be:
```
  assert y.shape.size = 1;
  assert y.shape.at(0) == 3;
  assert y.dtype == int8;
  let out = if (x > 3) {
    let out' = any
    assert out'.shape.size == 1;
    assert out'.shape.at(0) = 6;
    assert out'.dtype == int8
    out'
  } else {
    let out' = any;
    assert out'.shape.size == 1;
    assert out'.shape.at(0) == 3;
    assert out'.dtype == int8
    out'
  }
  out
```
A call `@f(4, <literal tensor of [3]>` would then have shape expression:
```
  let out = any
  assert out.shape.size == 1;
  assert out.shape.at(0) == 3;
  assert out.dtype == int8;
  out
```
which may flow to the outer context.

Or, we could forget the dataflow, in which all calls to @f will need to use
dynamic shape information.

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
  assert y.shape.size = 1;
  assert y.shape.at(0) == 3;
  assert y.dtype == int8;
  let out = if (x > 3) {
    @add(y, x)
  } else {
    @add(y, y)
  }
  assert out.shape.size = 1;
  assert out.shape.at(0) == 3;
  assert out.dtype == int8
  out 
}
```
and the body shape expression is simplified to:
```
assert y.shape.size = 1;
assert y.shape.at(0) == 3;
assert y.dtype == int8;
let out = if (x > 3) {
  let out' = any
  assert out'.shape.size == 1
  assert out'.shape.at(0) == 3;
  assert out'.dtype == int8
  out'
} else {
  let out' = any
  assert out'.shape.size == 1
  assert out'.shape.at(0) == 3;
  assert out'.dtype == int8
  out'
}
assert out.shape.size = 1;
assert out.shape.at(0) == 3;
assert out.dtype == int8
out 
```
But since each arm of the conditional entail the other we have:
```
assert y.shape.size = 1;
assert y.shape.at(0) == 3;
assert y.dtype == int8;
let out = any
assert out.shape.size == 1
assert out.shape.at(0) == 3;
assert out.dtype == int8
assert out.shape.size = 1;
assert out.shape.at(0) == 3;
assert out.dtype == int8
out 
```
thus:
```
let out = any
assert out.shape.size == 1
assert out.shape.at(0) == 3;
assert out.dtype == int8
out
```

In this case the user could omit the shape- and dtype-specific annotations
on the result type without loss of shape information.

Or, if we ignore control flow the user's annotation is necessary to propagate
shape information (and the 'out' assertions will be required at runtime).

Or, if the user's annotation was incorrect we could discover an 'out' assertion
is always false.

### Compile-time shape errors

An obvious 'shape error' should show up as an `assert false`. Eg:
```
shape_expr(@matmul(<literal tensor of shape (2, 3)>, <literal tensor of shape (4, 3)>))
==>
assert x.shape.size == 2;
assert x.shape.at(0) == 2;
assert x.shape.at(1) == 3;
assert x.dtype = int8;
assert y.shape.size == 2;
assert y.shape.at(0) == 4;
assert y.shape.at(1) == 3;
assert y.dtype = int8;
assert x.shape.size == 2;
let m = x.shape.at(0);         # m, k and d are free in x's stype          
let k = x.shape.at(1);
let d = x.dtype; 
assert y.shape.size == 2;
let n = y.shape.at(1);         # n is free in y's stype
assert y.shape.at(0) == k;     # k and d are bound in y's stype
assert y.dtype == d;
let out = any;
assert out.shape.size == 2;
assert out.shape.at(0) == m;   # m, n and d are bound in result stype
assert out.shape.at(1) == n;
assert out.dtype == d;
out
==>
let k = 3;
assert 4 == k
==>
assert false
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
That could get pretty challenging.


