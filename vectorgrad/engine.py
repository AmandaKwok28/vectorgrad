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
            if s == 1:
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
    
    
    # def matmul(self, other):
    #     other = Tensor._ensure_tensor(other)
    #     out = Tensor(self.data @ other.data, (self, other), '@')
        
    #     def _backward():
    #         self.grad += Tensor._unbroadcast(
    #             out.grad @ np.swapaxes(other.data, -1, -2),   # reverses last two axes to properly transpose
    #             self.data.shape
    #         )
            
    #         other.grad += Tensor._unbroadcast(
    #             np.swapaxes(self.data, -1, -2) @ out.grad,  
    #             other.data.shape
    #         )
        
    #     out._backward = _backward
    #     return out
    def matmul(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            # CASE 1: vector @ matrix
            if self.data.ndim == 1 and other.data.ndim == 2:
                # dL/dx
                self.grad += out.grad @ other.data.T
                # dL/dW
                other.grad += np.outer(self.data, out.grad)

            # CASE 2: matrix @ matrix (or batched later)
            else:
                self.grad += Tensor._unbroadcast(
                    out.grad @ np.swapaxes(other.data, -1, -2),
                    self.data.shape
                )
                other.grad += Tensor._unbroadcast(
                    np.swapaxes(self.data, -1, -2) @ out.grad,
                    other.data.shape
                )

        out._backward = _backward
        return out

    
    def conv2d(self, weight, stride=1):
        # get shape of kernel and image
        H, W = self.data.shape
        kh, kw = weight.data.shape
        
        # calculate the output shape
        out_h = (H - kh) // stride + 1
        out_w = (W - kw) // stride + 1
        out = np.zeros((out_h, out_w))
        
        # compute the convolution
        for i in range(out_h):
            for j in range(out_w):
                window = self.data[i*stride:i*stride+kh, j*stride:j*stride+kw]   # get the entire window of values you'll multiply with the kernel
                tmp = window * weight.data
                out[i,j] = tmp.sum()
                
        out = Tensor(out, (self, weight), 'conv2d')
        
        def _backward():
            
            for i in range(out_h):
                for j in range(out_w):
                    g = out.grad[i,j]
                    window = self.data[i*stride:i*stride+kh, j*stride:j*stride+kw]
                    
                    # calculate the loss gradient w.r.t. to X (self), get all contributions of this value to the output y
                    self.grad[i*stride:i*stride+kh, j*stride:j*stride+kw] += g * weight.data
                    
                    # kernel gradient
                    weight.grad += g * window
                    
        
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