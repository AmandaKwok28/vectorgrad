# Derivation of Conv2D (with Batching)

## Conv2D Shapes

```
X (input)   : B x Cin x H x W
W (weights) : Cout x Cin x Kh x Kw
Y (output)  : B x Cout x Hout x Wout
```

This implementation includes stride, but no padding.

## Definitions:

- `Cin`   - number of input channels
- `Cout`  - number of output channels
- `H, W`  - height and width of input
- `B`     - batch size
- `Hout`  - output height after convolution
- `Wout`  - output width after convolution
- `Kh`    - kernel height
- `Kw`    - kernel width

## Mental Model

Each output channel corresponds to one kernel.
That kernel slides over the input and produces a single feature map.

At each spatial location:
- extract a $K_h \times K_w$ patch from each input channel
- multiply by the corresponding kernel slice
- sum across input channels and spatial dimensions

This implementation follows the common deep learning convention and computes
cross-correlation rather than a mathematically flipped convolution.

## Conv2D Math

Let:
- $b$ index the batch
- $a$ index the output channel
- $c$ index the input channel

The output is defined as:

$$
Y[b, a, i, j] = \sum_{c=0}^{C_{in}-1} \sum_{u=0}^{K_h-1} \sum_{v=0}^{K_w-1} X[b, c, i + u, j + v] \cdot W[a, c, u, v]
$$

In words:
Each output pixel is the sum of elementwise products between a kernel
and the corresponding input patches across all input channels.

## Mapping the Math to Code

Below is the core logic (shown without batching for clarity):

```python
Y = zeros(Cout, Hout, Wout)
for cout in range(Cout):
    for i in range(Hout):
        for j in range(Wout):
            for cin in range(Cin):
                patch = X[cin, i : i + Kh, j : j + Kw]
                Y[cout, i, j] += sum(patch * W[cout, cin])
```

Batching simply adds an outer loop over $b$, applying the same weights
independently to each element in the batch.

## Backward Pass

To learn both features and weights, we compute:
- $\frac{\partial L}{\partial X}$
- $\frac{\partial L}{\partial W}$

We derive $\frac{\partial L}{\partial X}$ as a representative example.

Let the scalar loss be:

$$
L = f(Y)
$$

By the chain rule:

$$
\frac{\partial L}{\partial X[b, p, q]} = \sum_{a, i, j} \frac{\partial L}{\partial Y[b, a, i, j]} \cdot \frac{\partial Y[b, a, i, j]}{\partial X[b, p, q]}
$$

**Interpretation:**
For a fixed input pixel $(b, p, q)$, only output positions $(a, i, j)$
whose receptive field includes that pixel contribute a non-zero gradient.

## Gradient with Respect to Input

From the forward definition, we obtain:

$$
\frac{\partial L}{\partial X[b, p, q]} = \sum_{a, i, j} \frac{\partial L}{\partial Y[b, a, i, j]} \cdot W[a, c, p - i, q - j]
$$

Unlike the forward pass (cross-correlation), this corresponds to a true
convolution. This can be handled either by flipping the kernel or by
re-indexing. The implementation below uses re-indexing.

### Literal Gradient Implementation

```python
for cin in range(Cin):
    for p in range(H):
        for q in range(W):
            for cout in range(Cout):
                for i in range(Hout):
                    for j in range(Wout):
                        u = p - i
                        v = q - j
                        if 0 <= u < Kh and 0 <= v < Kw:
                            dX[cin, p, q] += out.grad[cout, i, j] * W[cout, cin, u, v]
```

### More Idiomatic Accumulation

Instead of asking:
"Which outputs depend on this input pixel?"

We ask:
"For each output pixel, which input pixels did it touch?"

```python
for cout in range(Cout):
    for iout in range(Hout):
        for jout in range(Wout):
            g = out.grad[cout, iout, jout]
            for cin in range(Cin):
                for u in range(Kh):
                    for v in range(Kw):
                        x = iout + u
                        y = jout + v
                        dX[cin, x, y] += g * W[cout, cin, u, v]
```

### Sliced Form

```python
for cout in range(Cout):
    for cin in range(Cin):
        for iout in range(Hout):
            for jout in range(Wout):
                dX[cin, iout : iout + Kh, jout : jout + Kw] += \
                    out.grad[cout, iout, jout] * W[cout, cin]
```