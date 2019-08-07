import torch.nn as nn
import torch
import Utils

class ScalingAndSquaring(nn.Module):
  def __init__(self ,num_steps=0):
      super(ScalingAndSquaring, self).__init__()
      self.num_steps = num_steps
      
      
  def forward(self, x):
    x = x/(2**self.num_steps)
    for _ in range(self.num_steps):
      xDef = torch.empty(x.shape, device=x.device, requires_grad=False)
      for chanIdx in range(-1, (x.shape[1]/3) - 1):
        chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
        for channel in chanRange:
          imgToDef = x[:, None, channel, ]                
          deformedTmp = Utils.deformWithNearestNeighborInterpolation(imgToDef, x[: , chanRange, ], x.device)#Utils.deformImage(imgToDef, x[: , chanRange, ], x.device, True, False)
          xDef[:, channel, ] = deformedTmp[:, 0, ]
      x = x.add(xDef)
    return x