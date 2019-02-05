import torch.nn as nn
import torch

class SelfSupervisionBrick(nn.Module):
  def __init__(self ,num_filters, in_channels=2, upSampletimes=0, padImg=True):
      super(SelfSupervisionBrick, self).__init__()
      
      
      self.conv = nn.Conv3d(in_channels, num_filters * 3, 1)
      self.upSampletimes = upSampletimes
      self.padImg = padImg
  def forward(self, x):
    
    x = self.conv(x)
    
    for i in range(self.upSampletimes):
      x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear')
      if not self.padImg: #if there is no padding in the encoder/decoders we need to pad here
        x = x[:,:,2:-2,2:-2,2:-2]
    
    return x
  
