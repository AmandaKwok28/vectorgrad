from vectorgrad.engine import Tensor
import numpy as np
import torch


def assert_allclose(a, b, atol=1e-6, rtol=1e-6):
    assert np.allclose(a, b, atol=atol, rtol=rtol), (
        f"\nExpected:\n{b}\nGot:\n{a}"
    )


def test_sanity_check():
    x = Tensor(3.0)
    y = x * x + 2 * x
    y.backward()

    xt = torch.tensor(3.0, requires_grad=True)
    yt = xt * xt + 2 * xt
    yt.backward()

    assert_allclose(y.data, yt.detach().numpy())
    assert_allclose(x.grad, xt.grad.numpy())


def test_vector_mul_add():
    x = Tensor([1.0, 2.0, 3.0])
    y = (x * x + 2).sum()
    y.backward()

    xt = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    yt = (xt * xt + 2).sum()
    yt.backward()

    assert_allclose(x.grad, xt.grad.numpy())



def test_scalar_vector_broadcast():
    x = Tensor(3.0)
    y = Tensor([1.0, 2.0, 3.0])
    z = (x * y).sum()
    z.backward()

    xt = torch.tensor(3.0, requires_grad=True)
    yt = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    zt = (xt * yt).sum()
    zt.backward()

    assert_allclose(x.grad, xt.grad.numpy())
    assert_allclose(y.grad, yt.grad.numpy())



def test_size1_broadcast():
    x = Tensor(np.random.randn(3, 1))
    y = Tensor(np.random.randn(3, 4))
    z = (x + y).sum()
    z.backward()

    xt = torch.tensor(x.data, requires_grad=True)
    yt = torch.tensor(y.data, requires_grad=True)
    zt = (xt + yt).sum()
    zt.backward()

    assert_allclose(x.grad, xt.grad.numpy())
    assert_allclose(y.grad, yt.grad.numpy())



def test_relu():
    x = Tensor(np.array([-1.0, 0.5, 2.0]))
    y = (x.relu() ** 2).sum()
    y.backward()

    xt = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    yt = (torch.relu(xt) ** 2).sum()
    yt.backward()

    assert_allclose(x.grad, xt.grad.numpy())


def test_sum_axis():
    x = Tensor(np.random.randn(3, 4))
    y = x.sum(axis=1).sum()
    y.backward()

    xt = torch.tensor(x.data, requires_grad=True)
    yt = xt.sum(dim=1).sum()
    yt.backward()

    assert_allclose(x.grad, xt.grad.numpy())


def test_sum_keepdims():
    x = Tensor(np.random.randn(3, 4))
    y = x.sum(axis=1, keepdims=True).sum()
    y.backward()

    xt = torch.tensor(x.data, requires_grad=True)
    yt = xt.sum(dim=1, keepdim=True).sum()
    yt.backward()

    assert_allclose(x.grad, xt.grad.numpy())



def test_mean():
    x = Tensor(np.random.randn(5, 6))
    y = x.mean()
    y.backward()

    xt = torch.tensor(x.data, requires_grad=True)
    yt = xt.mean()
    yt.backward()

    assert_allclose(x.grad, xt.grad.numpy())



def test_matmul_2d():
    A = Tensor(np.random.randn(3, 4))
    B = Tensor(np.random.randn(4, 5))
    y = (A @ B).sum()
    y.backward()

    At = torch.tensor(A.data, requires_grad=True)
    Bt = torch.tensor(B.data, requires_grad=True)
    yt = (At @ Bt).sum()
    yt.backward()

    assert_allclose(A.grad, At.grad.numpy())
    assert_allclose(B.grad, Bt.grad.numpy())



def test_batched_matmul():
    A = Tensor(np.random.randn(10, 3, 4))
    B = Tensor(np.random.randn(10, 4, 5))
    y = (A @ B).sum()
    y.backward()

    At = torch.tensor(A.data, requires_grad=True)
    Bt = torch.tensor(B.data, requires_grad=True)
    yt = (At @ Bt).sum()
    yt.backward()

    assert_allclose(A.grad, At.grad.numpy())
    assert_allclose(B.grad, Bt.grad.numpy())


def test_backward_twice_no_accumulation():
    x = Tensor([1.0, 2.0, 3.0])
    y = (x * x).sum()

    y.backward()
    grad1 = x.grad.copy()

    y.backward()
    grad2 = x.grad.copy()

    assert_allclose(grad1, grad2)


def test_batched_conv2d():
    np.random.seed(0)
    torch.manual_seed(0)

    # shapes
    B = 2
    C_in = 3
    C_out = 4
    H, W = 7, 8
    kH, kW = 3, 3
    stride = 1

    # random input + weights
    x = np.random.randn(B, C_in, H, W)
    w = np.random.randn(C_out, C_in, kH, kW)

    # vectorgrad
    X = Tensor(x)
    W = Tensor(w)
    Y = X.conv2d(W, stride=stride)
    loss = Y.sum()
    loss.backward()

    # torch
    Xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    Wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    Yt = torch.nn.functional.conv2d(Xt, Wt, stride=stride)
    loss_t = Yt.sum()
    loss_t.backward()

    # forward
    assert_allclose(Y.data, Yt.detach().numpy())

    # backward
    assert_allclose(X.grad, Xt.grad.numpy())
    assert_allclose(W.grad, Wt.grad.numpy())
