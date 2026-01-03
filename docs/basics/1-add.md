## Elementwise Addition:  `__add__`

### Forward pass

Given two tensors \( x \) and \( y \) of the same shape:


$z = x + y$


For example:

$x = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix},
\quad
y = \begin{bmatrix} 2 \\ 2 \\ 2 \end{bmatrix}
\Rightarrow
z = \begin{bmatrix} 3 \\ 3 \\ 3 \end{bmatrix}$



This is implemented as:

```py
out = Tensor(x.data + y.data, (x, y), '+')
```

### Backward pass

Let L be a scalar loss that depends on z.

We have:

$$
\frac{\partial L}{\partial x} = \sum_i \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial x_i}
$$

$$
\frac{\partial L}{\partial y} = \sum_i \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial y_i}
$$

For elementwise addition, the local partial derivatives satisfy  
$$ 
    \frac{\partial z_i}{\partial x_i} = 1 
$$ 
and  
$$ 
\frac{\partial z_i}{\partial y_i} = 1 
$$.

As a result, the upstream gradient  $\frac{\partial L}{\partial z}$ passes
through unchanged to both operands. This is why the backward implementation
only uses `out.grad`, rather than explicitly multiplying by a second term.

Note that since this Tensor class is implemented using NumPy, it often
implicitly broadcasts arrays. When broadcasting occurs, we must sum over
unbroadcasted dimensions to properly accumulate gradient contributions.

---

### Backward Function

Representing the above partials in code assuming `x = self` and `y = other`:

```py
self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
```


