import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    '''
    The LeNet paper uses Tanh but this is outdated so I'm using ReLU. Also, they use an rbf decision rule to train which is also outdated...
    
    '''
    def __init__(self):
        super().__init__()
        
        # feature extractor
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # classifier
        self.fc1 = nn.Linear(16*5*5, 120)  # flatten the inputs to feed into fc
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        
    def forward(self, x):
        
        # block 1
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2)         # kernel for pooling = [2,2]
        
        # block 2
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x,2)
        
        # flatten outputs for classification scheme
        x = torch.flatten(x, 1)
        
        # MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    


    
    
    
        
        
        
    