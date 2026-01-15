# Functions (unbatched)
This section has some functions that were taken out of the engine but are still useful for learning examples.


```py
def matmul(self, other):
    '''
    Unbatched matmul: good learning example to understand how each case is explicitly handled
    '''
    other = Tensor._ensure_tensor(other)
    out = Tensor(self.data @ other.data, (self, other), '@')

    def _backward():
        # case 1: vector @ vector
        if self.data.ndim == 1 and other.data.ndim == 1:
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        
        # case 2: vector @ matrix
        elif self.data.ndim == 1 and other.data.ndim == 2:
            self.grad += out.grad @ other.data.T       
            other.grad += np.outer(self.data, out.grad)

        # case 3: matrix @ vector
        elif self.data.ndim == 2 and other.data.ndim == 1:
            self.grad += np.outer(out.grad, other.data)
            other.grad += self.data.T @ out.grad

        # case 4: matrix @ matrix
        else:
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

    out._backward = _backward
    return out


# intuition: Let Cin = 6, and Cout = 16
# 16 kernels sees all 6 maps with random weights and grad descent tells us which 16 resolved kernels are the best    
def conv2d(self, weight, stride=1):
    '''
    Unbatched Conv2d
    '''
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
```


```py
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

def softmax(self):
    # numerical stability
    exps = (self - self.data.max()).exp()
    return exps / exps.sum()   

```