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
        self.W = Tensor(np.random.randn(nin, nout))
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
    

class MLP(Module):
    
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = []
        
        for i in range(len(nouts)):
            self.layers.append(Linear(sz[i], sz[i+1]))
            if i < len(nouts) - 1:
                self.layers.append(ReLU())
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
   
   
# these are just theortical... currently doesn't operate in tensor space which is a big problem   
class Conv2d(Module):
    '''
    INPUTS:
    - K: kernel size
    - C_in: number of channels: 1 for grayscale, 3 for rgb
    - C_out: number of feature maps you want

    '''
    
    # kernel = [C_out, C_in, K, K]
    def __init__(self, C_in, C_out, K):
        self.W = Tensor(np.random.randn(C_out, C_in, K, K))        # for every number of feature maps C_out, you need [C_in, K, K] kernels, kernels are shared across the image
        self.b = Tensor(np.zeros(C_out))                    # just this many biases
        self.Cout = C_out
        self.K = K
        
    # forward pass
    def __call__(self, x):
        
        # get shape of input
        [batches, C_in, H, W] = x.data.shape
        
        # calculate output shape
        K = self.K
        C = self.Cout
        H_out = H - K + 1
        W_out = W - K + 1
        out = np.zeros((batches, C, H_out, W_out))
        
        # calculate the feature maps
        for batch in range(batches):
            for cout in range(C):
                for i in range(H_out): 
                    for j in range(W_out):
                        total = 0
                        for chan in range(C_in):
                            tmp = x[batch, chan, i:i+K, j:j+K] * self.W[cout, chan, :, :]    # compute the patch dot product with the weights
                            tmp = tmp.sum()
                            total += tmp
                            
                        out[batch, cout, i, j] = (total + self.b[cout]).data  # add the bias
        
        return out
                
    def parameters(self):
        return [self.W, self.b]
    
            

class AvgPool(Module):
    
    def __init__(self, K):
        self.K = K
        
    def __call__(self, x):
        
        B, C, H, W = x.data.shape
        K = self.K
        
        H_out = H // K
        W_out = W // K
        out = np.zeros((B, C, H_out, W_out))   # note: pooling doesn't change the number of channels
        
        for b in range(B):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h0 = i * K
                        w0 = j * K
                        
                        patch = x[b, c, h0:h0+K, w0:w0+K]
                        out[b, c, i, j] = patch.mean()
        
        return out
    
    # modern average pooling doesn't need any params
    def parameters(self):
        return []  

        
class SoftmaxCrossEntropy(Module):
    
    def __init__(self):
        pass
    
    def __call__(self, logits, targets):

        B, C = logits.data.shape
        
        # compute numerically stable softmax
        z = logits.data - np.max(logits.data, axis=1, keepdims=True)
        exp_z = np.exp(z)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        # do NLL
        log_probs = -np.log(probs[np.arange(B), targets])
        loss = log_probs.mean()
        
        return loss
        

    # no parameters just computes the cross entropy loss
    def parameters(self):
        return []


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
        self.C1 = Conv2d(1, 6, 5)
        self.R1 = ReLU()
        self.S2 = AvgPool(2)
        
        self.C3 = Conv2d(6, 16, 5)
        self.R3 = ReLU()
        self.S4 = AvgPool(2)
        
        # classifiers
        self.FC5 = Linear(16 * 5 * 5, 120)  # flatten
        self.R5 = ReLU()
        
        self.FC6 = Linear(120, 84)
        self.R6 = ReLU()
        
        self.FC7 = Linear(84, 10)   # 10 digit classes -> logits
    
    # forward pass
    def __call__(self, x):
        # conv block 1
        x = self.C1(x)
        x = self.R1(x)
        x = self.S2(x)
        
        # conv block 2
        x = self.C3(x)
        x = self.R3(x)
        x = self.S4(x)
        
        # flatten
        B = x.data.shape[0]
        x = x.reshape(B, -1)
        
        # classifier
        x = self.FC5(x)
        x = self.R5(x)
        
        x = self.FC6(x)
        x = self.R6(x)
        
        x = self.FC7(x)   # logits
        return x
    
    def parameters(self):
        return (
            self.C1.parameters() +
            self.C3.parameters() +
            self.FC5.parameters() +
            self.FC6.parameters() +
            self.FC7.parameters()
        )