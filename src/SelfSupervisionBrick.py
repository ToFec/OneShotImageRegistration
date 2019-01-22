import torch.nn as nn
import torch

class SelfSupervisionBrick(nn.Module):
  def __init__(self ,num_filters, in_channels=2, upSampletimes=0):
      super(SelfSupervisionBrick, self).__init__()
      
      
      self.conv = nn.Conv3d(in_channels, num_filters * 3, 1)
      self.upSampletimes = upSampletimes

      
  def forward(self, x):
    
    x = self.conv(x)
    
    for i in range(self.upSampletimes):
      x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear')
    
    return x
  
