# Super-resolution using an efficient sub-pixel convolutional neural network
# https://github.com/pytorch/examples/tree/master/super_resolution

import torch
import torch.nn as nn
import torch.nn.init as init
from utils.coordconv import CoordConv2d

class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = CoordConv2d(1, 64, 5, stride=1, padding=2, with_r=True)
        self.conv2 = CoordConv2d(64, 64, 3, stride=1, padding=1)
        self.conv3 = CoordConv2d(64, 32, 3, stride=1, padding=1)
        self.conv4 = CoordConv2d(32, upscale_factor ** 2, 3, stride=1, padding=1)
        
        #self.conv1 = nn.Conv2d(64, 64, (1, 1), (1, 1))
        #self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        #self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        #x = self.coordconv(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        #init.orthogonal_(self.coordconv.weight)
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)