import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    '''
    AlexNet (Krizhevsky et al., 2012)

    Historical context:
    - LeNet used tanh activations, which saturate at ±1 and produce small gradients
      for large |x|, leading to slower convergence in deep networks.
    - AlexNet popularized the use of ReLU activations, enabling much faster
      optimization and scaling to large datasets like ImageNet.

    Key contributions:
    - ReLU activations: non-saturating for positive values, sparse activations,
      and significantly faster convergence with SGD (reported ~6x faster than tanh).
    - Overlapping max pooling (z=3, s=2): introduces mild regularization and
      reduces overfitting compared to non-overlapping pooling.
    - Local Response Normalization (LRN): encourages competition between nearby
      feature maps; historically important but largely obsolete today, replaced
      by BatchNorm and related normalization layers.
    - Dropout in fully connected layers to reduce co-adaptation and overfitting.
    - GPU training with explicit model parallelism across two GPUs due to
      memory constraints at the time (no longer necessary on modern hardware).

    Architectural note:
    - AlexNet is structurally similar to LeNet (conv → pool → conv → pool → FC),
      but dramatically scales depth, width, input resolution, and compute,
      demonstrating that CNNs can solve real-world vision tasks.

    Input assumption:
    - ImageNet-style inputs of size 224 x 224 x 3
    
    
    TLDR:
    AlexNext popularized ReLU, dropout for regularization at scale, Max Pooling, 
    and also gpu training
    '''

    
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),           # overlapping pooling

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # no pooling for layers 3, 4, 5 because you're trying not to downsample
            # the complex features too hard at this point
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # this paper also introduced dropouts for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

        
               
        
        
        