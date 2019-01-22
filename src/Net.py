import torch
import torch.nn as nn
from torch.nn import init

from EncoderBrick import EncoderBrick
from DecoderBrick import DecoderBrick
from SelfSupervisionBrick import SelfSupervisionBrick

class UNet(nn.Module):
  def __init__(self, in_channels=2, useBatchNorm=True, concatLayer=True, depth = 5, numberOfFiltersFirstLayer=32, useDeepSelfSupervision = False):
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
      self.encoders.append( EncoderBrick(outputFilterNumber, currentNumberOfInputChannels, self.useBatchNorm, self.concatLayer) )
      if ( i < self.depth-1):
        self.pools.append( nn.AvgPool3d(2, 2) )
        self.decoders.append( DecoderBrick(outputFilterNumber, outputFilterNumber*2, self.useBatchNorm, self.concatLayer) )
      if self.useDeepSelfSupervision:
        self.selfSupervisions.append(SelfSupervisionBrick(in_channels, outputFilterNumber, i))
    
    if not self.useDeepSelfSupervision:
      self.selfSupervisions.append(SelfSupervisionBrick(in_channels, numberOfFiltersFirstLayer, 0))
      
    self.encoders = nn.ModuleList(self.encoders)
    self.decoders = nn.ModuleList(self.decoders)
    self.pools = nn.ModuleList(self.pools)
    self.selfSupervisions = nn.ModuleList(self.selfSupervisions)
    
    self.reset_params()   
    

  def forward(self, x):
    encoder_outs = []
    supervisionInputs = range(len(self.encoders))
    for i, encoder in enumerate(self.encoders):
      x = encoder(x)
      encoder_outs.append(x)
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
  