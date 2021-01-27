import torch
import torch.nn as nn
import torch.nn.init as init
from utils.coordconv import CoordConv2d

class Net(nn.Module):
    def __init__(self):
    
        super(Net, self).__init__()
        
        self.layers = torch.nn.Sequential(
            #nn.Conv2d(3, 56, 5, padding=2), # 72
            CoordConv2d(3, 56, 5, padding=2),
            #nn.InstanceNorm2d(56),
            nn.PReLU(),
            nn.Conv2d(56, 12, 1),
            #nn.InstanceNorm2d(12),
            nn.PReLU(),
            nn.Conv2d(12, 12, 3, padding=1),
            #nn.InstanceNorm2d(12),
            nn.PReLU(),
            nn.Conv2d(12, 12, 3, padding=1),
            #nn.InstanceNorm2d(12),
            nn.PReLU(),
            nn.Conv2d(12, 12, 3, padding=1),
            #nn.InstanceNorm2d(12),
            nn.PReLU(),
            nn.Conv2d(12, 12, 3, padding=1),
            #nn.InstanceNorm2d(12),
            nn.PReLU(),
            nn.Conv2d(12, 56, 1), # 72
            #nn.InstanceNorm2d(56),
            nn.PReLU(),
            nn.ConvTranspose2d(56, 3, 9, stride=4, output_padding=1, padding =3) #  (72-1) * 2 + 9 - 8 + 1 = 144 
            # stride = 2   padding=4  changed!
        )

    def weight_init(self, mean=0.0, std=0.02):
            for m in self._modules:
                normal_init(self._modules[m], mean, std)

    def forward(self, x):
        out = self.layers(x)
        return out

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, CoordConv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()