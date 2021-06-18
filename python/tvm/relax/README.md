# Current flow

Synr based frontend -> Relax Input IR
Relax Input IR
    -> Type Check
    -> Normal Form
    -> Specialization?
    -> Compilation
    -> Execution

# Building the state of Relax
In order to build currently, first manually run:
```
PYTHONPATH="./objectgen" python -m objectgen.relax
```
then build TVM like normal.
