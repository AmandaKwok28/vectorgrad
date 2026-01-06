import numpy as np

class Tensor:

    # initialize a node in a computation graph
    # data = numpy array
    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)            # gradient is still a vector                            
        self._backward = lambda: None             
        self._prev = set(_children)             
        self._op = _op                
        
    
    # note: forward broadcasting is already done as a part of numpy
    @staticmethod
    def _unbroadcast(grad, shape):
        # remove leading dimensions by summing them together to account for contributions
        # numpy broadcasts by adding leading dimensions so we sum by them
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
            
        # sum over broadcasted axes with size = 1 (collapse them back down) to get contributions to gradient
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:                # only reduce if it was originally broadcast
                grad = grad.sum(axis=i, keepdims=True)
                
        return grad
    
    # method to ensure that operations are done with either tensors, ints, or floats
    # if it's a numpy array, the operations might be overtaken by numpy implementation
    # which will mess up our computation graph
    @staticmethod
    def _ensure_tensor(x):
        if isinstance(x, Tensor):
            return x
        if isinstance(x, (int, float)):
            return Tensor(x)
        raise TypeError(f"Unsupported operand type: {type(x)}")

    

    def __add__(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
            
        out._backward = _backward       
        return out
    
    
    def __mul__(self, other):
        other = Tensor._ensure_tensor(other)
        prod = Tensor(self.data * other.data, (self, other), '*')
    
        def _backward():
            self.grad += Tensor._unbroadcast(prod.grad * other.data, self.data.shape)
            other.grad += Tensor._unbroadcast(prod.grad * self.data, other.data.shape)
            
        prod._backward = _backward
        return prod
        
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        expo = Tensor(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += Tensor._unbroadcast(
                expo.grad * other * (self.data ** (other - 1)), 
                self.data.shape
            ) 
        
        expo._backward = _backward
        return expo
    
    def relu(self):
        val = Tensor(np.maximum(0, self.data), (self,), 'ReLU')
        
        def _backward():
            grad = (self.data > 0) * val.grad
            self.grad += Tensor._unbroadcast(grad, self.data.shape) 

        val._backward = _backward
        return val
    
    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), 'tanh')
        
        def _backward():
            self.grad += out.grad * (1 - out.data ** 2)

        out._backward = _backward
        return out
    
    def backward(self):
        topo = []               
        visited = set()
        
        def dfs(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    dfs(child)
                topo.append(node)
        dfs(self)
        
        # zero all gradients
        for node in topo:
            node.grad = np.zeros_like(node.data)
        
        # seed the output grad to be 1
        self.grad = np.ones_like(self.data)
        
        # backpropagate
        for val in reversed(topo):
            val._backward()           
            
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')
        
        def _backward():
            grad = out.grad
            
            # if dimensions were removed, re-insert them
            if axis is not None and not keepdims:
                if isinstance(axis, int):
                    axes = (axis,)
                else:
                    axes = axis
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)
                    
            # broadcast gradient back to input shape
            grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad
        
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        if axis is None:
            denom = self.data.size
        else:
            if isinstance(axis, int):
                denom = self.data.shape[axis]
            else:
                denom = np.prod([self.data.shape[a] for a in axis])
        
        return self.sum(axis=axis, keepdims=keepdims) * (1 / denom)
    
    
    # unbatched matmul
    # def matmul(self, other):
    #     other = Tensor._ensure_tensor(other)
    #     out = Tensor(self.data @ other.data, (self, other), '@')

    #     def _backward():
    #         # case 1: vector @ vector
    #         if self.data.ndim == 1 and other.data.ndim == 1:
    #             self.grad += out.grad * other.data
    #             other.grad += out.grad * self.data
            
    #         # case 2: vector @ matrix
    #         elif self.data.ndim == 1 and other.data.ndim == 2:
    #             self.grad += out.grad @ other.data.T       
    #             other.grad += np.outer(self.data, out.grad)

    #         # case 3: matrix @ vector
    #         elif self.data.ndim == 2 and other.data.ndim == 1:
    #             self.grad += np.outer(out.grad, other.data)
    #             other.grad += self.data.T @ out.grad

    #         # case 4: matrix @ matrix
    #         else:
    #             self.grad += out.grad @ other.data.T
    #             other.grad += self.data.T @ out.grad

    #     out._backward = _backward
    #     return out
    
    def matmul(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        
        # function to broadcast a 1D vectors to 2D for matmul
        # if left: treat as a row vector, else column vector
        def promote(x, left: bool):
            if x.ndim == 1:
                return x[None,:] if left else x[:,None] 
            return x
        
        def _backward():
            X = promote(self.data, left=True)
            Y = promote(other.data, left=False)
            dZ = out.grad
            
            # recall the dims are output dims x input dims so it's always broadcast on the left
            if dZ.ndim == 1:
                dZ = dZ[None, :]
                
            dX = Tensor._unbroadcast(dZ @ np.swapaxes(Y, -1, -2), self.data.shape)
            dY = Tensor._unbroadcast(np.swapaxes(X, -1, -2) @ dZ, other.data.shape)
            self.grad += dX
            other.grad += dY
            
        out._backward = _backward
        return out

    # intuition: Let Cin = 6, and Cout = 16
    # 16 kernels sees all 6 maps with random weights and grad descent tells us which 16 resolved kernels are the best    
    # def conv2d(self, weight, stride=1):
    #     # get shape of kernel and image
    #     H, W = self.data.shape
    #     kh, kw = weight.data.shape
        
    #     # calculate the output shape
    #     out_h = (H - kh) // stride + 1
    #     out_w = (W - kw) // stride + 1
    #     out = np.zeros((out_h, out_w))
        
    #     # compute the convolution
    #     for i in range(out_h):
    #         for j in range(out_w):
    #             window = self.data[i*stride:i*stride+kh, j*stride:j*stride+kw]   # get the entire window of values you'll multiply with the kernel
    #             tmp = window * weight.data
    #             out[i,j] = tmp.sum()
                
    #     out = Tensor(out, (self, weight), 'conv2d')
        
    #     def _backward():
            
    #         for i in range(out_h):
    #             for j in range(out_w):
    #                 g = out.grad[i,j]
    #                 window = self.data[i*stride:i*stride+kh, j*stride:j*stride+kw]
                    
    #                 # calculate the loss gradient w.r.t. to X (self), get all contributions of this value to the output y
    #                 self.grad[i*stride:i*stride+kh, j*stride:j*stride+kw] += g * weight.data
                    
    #                 # kernel gradient
    #                 weight.grad += g * window
                    
        
    #     out._backward = _backward
    #     return out
    
    # batched implementation
    def conv2d(self, weight, stride=1):
        B, C_in, H, W = self.data.shape
        C_out, _, kH, kW = weight.data.shape

        out_h = (H - kH) // stride + 1
        out_w = (W - kW) // stride + 1

        out = np.zeros((B, C_out, out_h, out_w))

        for b in range(B):
            for oc in range(C_out):
                for i in range(out_h):
                    for j in range(out_w):
                        window = self.data[
                            b,
                            :,
                            i*stride:i*stride+kH,
                            j*stride:j*stride+kW
                        ]  # shape: (C_in, kH, kW)

                        out[b, oc, i, j] = np.sum(
                            window * weight.data[oc]
                        )
                        
        out = Tensor(out, (self, weight), 'conv2d')
        
        def _backward():
            for b in range(B):
                for oc in range(C_out):
                    for i in range(out_h):
                        for j in range(out_w):
                            g = out.grad[b, oc, i, j]

                            window = self.data[
                                b,
                                :,
                                i*stride:i*stride+kH,
                                j*stride:j*stride+kW
                            ]

                            # dX: accumulate over output channels
                            self.grad[
                                b,
                                :,
                                i*stride:i*stride+kH,
                                j*stride:j*stride+kW
                            ] += g * weight.data[oc]

                            # dW: accumulate over batch + spatial
                            weight.grad[oc] += g * window
                            
        out._backward = _backward
        return out

    
    def avg_pool2d(self, kh, kw, stride=1):
        if stride is None:
            stride = kh        # LeNet style default
        
        H, W = self.data.shape
        out_h = (H - kh) // stride + 1
        out_w = (W - kw) // stride + 1
        out = np.zeros((out_h, out_w))
        
        # forward pass
        for i in range(out_h):
            for j in range(out_w):
                window = self.data[i*stride:i*stride+kh, j*stride:j*stride+kw]
                out[i,j] = window.mean()
                
        out = Tensor(out, (self,), 'avg_pool2d')
        
        def _backward():
            for i in range(out_h):
                for j in range(out_w):
                    g = out.grad[i,j]
                    self.grad[i*stride:i*stride+kh, j*stride:j*stride+kw] += g / (kh*kw)
        
        out._backward = _backward
        return out  
    
    def flatten(self):
        out = Tensor(self.data.reshape(-1), (self,), 'flatten')

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out
    
    
    @staticmethod   # come back to this...
    def concat(tensors):
        data = np.concatenate([t.data for t in tensors])
        out = Tensor(data, tuple(tensors), 'concat')

        sizes = [t.data.size for t in tensors]

        def _backward():
            idx = 0
            for t, sz in zip(tensors, sizes):
                t.grad += out.grad[idx:idx+sz].reshape(t.data.shape)
                idx += sz

        out._backward = _backward
        return out
    
    def softmax(self):
        # numerical stability
        exps = (self - self.data.max()).exp()
        return exps / exps.sum()        
    
    # allow slicing
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), 'slice')
        def _backward():
            self.grad[idx] += out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')
        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out


    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):        
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        other = Tensor._ensure_tensor(other)
        return self * other
    
    def __truediv__(self, other):
        other = Tensor._ensure_tensor(other)
        if np.any(other.data == 0):
            raise ZeroDivisionError("division by zero in Tensor")
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        other = Tensor._ensure_tensor(other)
        return other * (self ** -1)
    
    def __matmul__(self, other):
        return self.matmul(other)

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"