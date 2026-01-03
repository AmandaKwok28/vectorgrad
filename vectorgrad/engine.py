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
            
        # sum over broadcasted axes with size = 1 (collapse them back down)
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
    
    
    def matmul(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        
        def _backward():
            self.grad += Tensor._unbroadcast(
                out.grad @ np.swapaxes(other.data, -1, -2),   # reverses last two axes to properly transpose
                self.data.shape
            )
            
            other.grad += Tensor._unbroadcast(
                np.swapaxes(self.data, -1, -2) @ out.grad,  
                other.data.shape
            )
        
        out._backward = _backward
        return out
    
    # allow slicing
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), 'slice')
        def _backward():
            self.grad[idx] += out.grad
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