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
    

    
    # batched matmul    
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
    
    
    def conv2d(self, Cout, kernel_size, stride=1):
        '''
        Performs batched 2d convolution allowing for several input and output channels
        
        :param self: input image of size - B x Cin x H x W
        :param Cout: number of output channels desired
        :param kernel_size: a tuple indicating the width and height of your desired kernel
        :param stride: value of the stride for your kernel
        
        Returns
        out - output batched images of size B x Cout x Hout x Wout where each batch has size Cout x Hout x Wout
        '''
        
        # init the weights
        kh, kw = kernel_size
        s = stride
        
        # get the batch dim and set up output dims
        B, Cin, H, Wim = self.data.shape
        W = Tensor(np.random.randn(Cout, Cin, kh, kw), (), 'Weight')
        Hout = (H - kh) // stride + 1
        Wout = (Wim - kw) // stride + 1

        
        # init output 
        out_data = np.zeros((B, Cout, Hout, Wout))
                            
        # computing the forward pass idiomatically   
        for b in range(B):                                                              # we do this for each "batch of size (Cout, Hout, Wout)"
            for cout in range(Cout):                                                    # for each output channel positioned at (i,j)
                for i in range(Hout):
                    for j in range(Wout):
                        for cin in range(Cin):                                          # we sum the convolution / cross correlation between X and the corresponding kernel
                            img_patch = self.data[b, cin, i*s:i*s+kh, j*s:j*s+kw]
                            out_data[b, cout, i, j] += np.sum(img_patch * W.data[cout, cin])
        
        out = Tensor(out_data, (self, W), 'conv2d')
                            
        # compute the backwards pass for dL/dX and dL/dW                
        def _backward():
            # backwards pass to find dL/dX
            dX = np.zeros((B, Cin, H, Wim))

            # see the conv2d derivation of the code, but I also added the batching and stride back in
            for b in range(B):
                for cout in range(Cout):
                    for cin in range(Cin):
                        for iout in range(Hout):
                            for jout in range(Wout):
                                dX[b, cin,
                                iout*s:iout*s+kh,
                                jout*s:jout*s+kw] += (
                                    out.grad[b, cout, iout, jout] * W.data[cout, cin]
                                )


                        
            self.grad += dX
            
            dW = np.zeros((Cout, Cin, kh, kw))

            for b in range(B):
                for cout in range(Cout):
                    for cin in range(Cin):
                        for iout in range(Hout):
                            for jout in range(Wout):
                                dW[cout, cin] += (
                                    out.grad[b, cout, iout, jout]
                                    * self.data[b, cin,
                                        iout*s : iout*s + kh,
                                        jout*s : jout*s + kw]
                                )
                                
            W.grad += dW

        out._backward = _backward

        return out, W

    
    def avg_pool2d(self, kh, kw, stride=1):
        if stride is None:
            stride = kh  # LeNet-style default

        B, C, H, W = self.data.shape
        out_h = (H - kh) // stride + 1
        out_w = (W - kw) // stride + 1

        out_data = np.zeros((B, C, out_h, out_w))


        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        window = self.data[
                            b, c,
                            i*stride : i*stride + kh,
                            j*stride : j*stride + kw
                        ]
                        out_data[b, c, i, j] = window.mean()

        out = Tensor(out_data, (self,), 'avg_pool2d')

        def _backward():
            for b in range(B):
                for c in range(C):
                    for i in range(out_h):
                        for j in range(out_w):
                            g = out.grad[b, c, i, j]
                            self.grad[
                                b, c,
                                i*stride : i*stride + kh,
                                j*stride : j*stride + kw
                            ] += g / (kh * kw)

        out._backward = _backward
        return out

    
    def flatten(self):
        B = self.data.shape[0]
        out_data = self.data.reshape(B, -1)

        out = Tensor(out_data, (self,), 'flatten')

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    
    
    @staticmethod  # come back to this...
    def concat(tensors, axis=0):
        data = np.concatenate([t.data for t in tensors], axis=axis)
        out = Tensor(data, tuple(tensors), 'concat')

        sizes = [t.data.shape[axis] for t in tensors]

        def _backward():
            idx = 0
            for t, sz in zip(tensors, sizes):
                slicer = [slice(None)] * out.grad.ndim
                slicer[axis] = slice(idx, idx + sz)
                t.grad += out.grad[tuple(slicer)]
                idx += sz

        out._backward = _backward
        return out

    
    def softmax(self):
        # subtract max per batch element for numerical stability
        shifted = self.data - self.data.max(axis=1, keepdims=True)
        exps = np.exp(shifted)

        probs = exps / exps.sum(axis=1, keepdims=True)
        out = Tensor(probs, (self,), 'softmax')

        def _backward():
            # see explanation below
            for b in range(self.data.shape[0]):
                y = out.data[b]      # softmax output (D,)
                g = out.grad[b]      # upstream gradient (D,)

                # Jacobian-vector product for softmax
                self.grad[b] += y * (g - np.dot(g, y))

        out._backward = _backward
        return out
     
    
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