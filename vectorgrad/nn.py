from vectorgrad.engine import Tensor
import numpy as np

class Module:
    
    # reset the gradients
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
            
    def parameters(self):
        return []

class Linear(Module):
    
    def __init__(self, nin, nout):
        std = np.sqrt(2 / (nin + nout))
        self.W = Tensor(np.random.randn(nin, nout) * std)
        self.b = Tensor(np.zeros(nout))
        
    # forward pass through all the neurons for this input array x
    def __call__(self, x):
        out = x @ self.W + self.b
        return out
    
    def parameters(self):
        return [self.W, self.b]
    
    def __repr__(self):
        return f"Linear({self.W.data.shape[0]}, {self.W.data.shape[1]})"
    
class ReLU(Module):
    
    def __call__(self, x):
        return x.relu()
    
    def parameters(self):
        return []
    
    def __repr__(self):
        return "ReLU()"
    
class Tanh(Module):
    
    def __call__(self, x):
        return x.tanh()
    
    def parameters(self):
        return []
    
    def __repr__(self):
        return "Tanh()"
    

class MLP(Module):
    
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = []
        
        for i in range(len(nouts)):
            self.layers.append(Linear(sz[i], sz[i+1]))
            if i < len(nouts) - 1:
                self.layers.append(Tanh())
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    


# ----------------------------------------------------------------------------------------------
# LeNet stuff
# ----------------------------------------------------------------------------------------------

class Conv2d(Module):
    '''
    Wrapper for the conv2d operation
    '''
    def __init__(self, kh, kw):
        self.W = Tensor(np.random.randn(kh, kw) * 0.1)
        
    def __call__(self, x):
        return x.conv2d(self.W)
    
    def parameters(self):
        return [self.W]
        
    
    
class LeNet(Module):
    '''
    Modernized LeNet implementation.

    Key differences from the original LeNet-5:
    - Uses full channel connectivity in C3 instead of manual connection tables. Random initialization and stochastic minibatch gradient descent already
      break symmetry and encourage feature diversity.
    - Replaces tanh activations with ReLU for simplicity and improved optimization.
    - Replaces the original RBF-based output layer with a standard linear layer trained using softmax cross-entropy, since we only care about digit
      classification (not rejection of non-character inputs).

    This implementation preserves the core inductive biases of LeNet (local receptive fields, weight sharing, and subsampling) while
    adopting modern training conventions.
    
    '''
    
    def __init__(self):
        
        # feature extractors
        self.conv1 = [Conv2d(5, 5) for _ in range(6)]
        self.conv2 = [
            [Conv2d(5, 5) for _ in range(6)]   # each output channel has 6 kernels (1 per input channel)
            for _ in range(16)
        ]
        
        # classifer
        self.fc1 = Linear(16*5*5, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)
    
    
    # forward pass
    def __call__(self, x):
        
        # C1  + S2
        x = [conv(x).relu().avg_pool2d(2, 2, 2) for conv in self.conv1]
        
        # C3  + S4
        new_maps = []                                       # holds 16 output feature maps of C3
        for kernels in self.conv2:                          # one group of 6 conv kernels
            y = None                                        # accumulate the sum of all 6 convolutions
            for fm, conv in zip(x, kernels):                # for each feature map, kernel pair
                out = conv(fm)                              # take the convolution
                y = out if y is None else y + out           # accumulate the sum 
            y = y.relu().avg_pool2d(2, 2, 2)                   # S4 : apply average pooling
            new_maps.append(y)                              # add the new feature map
        x = new_maps
        
        # flatten
        x = Tensor.concat([fm.flatten() for fm in x])       # flatten into one long tensor with 400 items

        # classifier (basically an MLP)
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)        
        
        return x
    
    def parameters(self):
        params = []

        # C1 kernels (6)
        for conv in self.conv1:
            params += conv.parameters()

        # C3 kernels (16 × 6)
        for kernel_group in self.conv2:
            for conv in kernel_group:
                params += conv.parameters()

        # fully connected layers
        params += self.fc1.parameters()
        params += self.fc2.parameters()
        params += self.fc3.parameters()
        
        return params