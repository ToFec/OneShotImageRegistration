import torch
import torch.nn as nn
from torch.nn import init

from EncoderBrick import EncoderBrick
from DecoderBrick import DecoderBrick
from SelfSupervisionBrick import SelfSupervisionBrick

from numpy import power

class UNet(nn.Module):
  def __init__(self, in_channels=2, useBatchNorm=True, concatLayer=True, depth = 5, numberOfFiltersFirstLayer=32, useDeepSelfSupervision = False, padImg=True):
    super(UNet, self).__init__()
    
    if (depth < 2):
      raise ValueError("minimum depth is 2")
    
    self.useBatchNorm=useBatchNorm
    self.concatLayer=concatLayer
    self.in_channels = in_channels
    self.useDeepSelfSupervision = useDeepSelfSupervision
    
    self.encoders = []
    self.decoders = []
    self.pools = []
    self.selfSupervisions = []
    
    self.depth = depth
    for i in range(self.depth):
      currentNumberOfInputChannels = self.in_channels if i == 0 else outputFilterNumber
      outputFilterNumber = numberOfFiltersFirstLayer*(2**i)
      self.encoders.append( EncoderBrick(outputFilterNumber, currentNumberOfInputChannels, self.useBatchNorm, self.concatLayer, padImg) )
      if ( i < self.depth-1):
        self.pools.append( nn.AvgPool3d(2, 2) )
        self.decoders.append( DecoderBrick(outputFilterNumber, outputFilterNumber*2, self.useBatchNorm, self.concatLayer, padImg) )
      if self.useDeepSelfSupervision:
        self.selfSupervisions.append(SelfSupervisionBrick(in_channels, outputFilterNumber, i, padImg))
    
    if not self.useDeepSelfSupervision:
      self.selfSupervisions.append(SelfSupervisionBrick(in_channels, numberOfFiltersFirstLayer, 0, padImg))
      
    self.encoders = nn.ModuleList(self.encoders)
    self.decoders = nn.ModuleList(self.decoders)
    self.pools = nn.ModuleList(self.pools)
    self.selfSupervisions = nn.ModuleList(self.selfSupervisions)

    self.receptiveFieldOffsets = [0] * (depth - 1)
    if not padImg:
      offsetBase = 2
      offsetCenter = 2
      for i in range(depth-1):
        offsetCenter = offsetBase * offsetCenter
        offset = offsetCenter
        for j in range(i):
          offset += 2* power(offsetBase,2+j)
        self.receptiveFieldOffsets[i] = offset
    
    
    self.reset_params()   
    

  def forward(self, x):
    
    encoder_outs = []
    supervisionInputs = list(range(len(self.encoders)))#python3
    for i, encoder in enumerate(self.encoders):
      x = encoder(x)
      rFOffSet = self.receptiveFieldOffsets[self.depth - 2 - i]
      encoder_outs.append(x[:,:,rFOffSet:x.shape[2]-rFOffSet, rFOffSet:x.shape[3]-rFOffSet, rFOffSet:x.shape[4]-rFOffSet])
      if ( i < self.depth-1):
        x = self.pools[i](x)
      else:
        supervisionInputs[i] = x
        
    for i in range(len(self.decoders) -1, -1, -1):
      decoder = self.decoders[i]
      encOut = encoder_outs[i]
      x = decoder(x, encOut)
      supervisionInputs[i] = x      
    
    
    outputFields = []
    for i in range(len(self.selfSupervisions) -1, -1, -1):
      decOut = supervisionInputs[i]
      selfSupervision = self.selfSupervisions[i]
      outputFields.append( selfSupervision(decOut) )
    
    x = torch.stack(outputFields)
    tmp = torch.sum(x, dim = 0)
    return tmp
  

  def reset_params(self):
    for i, m in enumerate(self.modules()):
      if isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight)
        init.constant_(m.bias, 0)
      elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight)
        init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
  