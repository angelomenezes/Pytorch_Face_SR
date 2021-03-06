import torch
import torch.nn as nn
import torch.nn.init as init
from utils.coordconv import CoordConv2d

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.coordconv = CoordConv2d(1, 64, 9, stride=1, padding=4, with_r=False)
        self.conv1 = CoordConv2d(1, 64, kernel_size=9, padding=4, stride=1, with_r=True)
        self.conv2 = CoordConv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = CoordConv2d(32, 1, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        
        #self.conv1 = nn.Conv2d(64, 64, kernel_size=9, stride=1, padding=4)
        #self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        #self.conv3 = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)
        #self.relu = nn.ReLU()
        
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight)