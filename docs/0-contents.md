# Basic Autograd Operations

## Extended from Micrograd
This section lists the core scalar and elementwise operations inherited from *micrograd*, extended to handle tensors.
These operations form the foundation of reverse-mode automatic differentiation and are reused
extensively by higher-level layers.

1. `_unbroadcast`  (not part of the original micrograd but necessary context)
2. `__add__`
3. `__mul__`
4. `__pow__`
5. `relu`
6. `backward`
7. `__neg__`
8. `__sub__`
9. `__rsub__`
10. `__rmul__`
11. `__truediv__`
12. `__rtruediv__`
13. `__repr__`

## Additional Operations 
This section lists the additional operations needed to adapt micrograd into a vector based autodiff engine. Currently working on fully adapting the 
vectorgrad engine to minimally implement the LeNet CNN architecture. 

14. `sum`
15. `mean`
16. `matmul`
17. `conv2d`
18. `avg_pool2d`
19. `flatten`
20. `concat`
21. `softmax`
22. `__getitem__`
23. `exp`
24. `log` 


## General Layout

Each operation is documented in its own file using the following structure:

```
.
├── Operation name
├── Forward pass
├── Backward pass
└── Backward Implementation