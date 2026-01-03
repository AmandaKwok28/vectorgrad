# Basic Autograd Operations

This section covers the core scalar and elementwise operations inherited from *micrograd*, extended to handle tensors.
These operations form the foundation of reverse-mode automatic differentiation and are reused
extensively by higher-level layers.

## Covered operations
0. `_unbroadcast`  (extra context for adapting to tensors)
1. `__add__`
2. `__mul__`
3. `__pow__`
4. `relu`
5. `__neg__`
6. `__sub__`
7. `__rsub__`
8. `__rmul__`
9. `__truediv__`
10. `__rtruediv__`
11. `__repr__`


## General Layout

Each operation is documented in its own file using the following structure:

```
.
├── Operation name
├── Forward pass
├── Backward pass
└── Implementation details