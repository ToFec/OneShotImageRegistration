import torch.nn as nn
import torch

class DecoderBrick(nn.Module):
  def __init__(self ,num_filters, in_channels=2, useBatchNorm=True, concatLayer=True):
      super(DecoderBrick, self).__init__()
      
      self.useBatchNorm=useBatchNorm
      self.concatLayer=concatLayer
      self.padVals = (1, 1, 1, 1, 1, 1)
      self.in_channels = in_channels
      
      self.deconv0 = nn.ConvTranspose3d(in_channels, num_filters, 2, 2)
      if (self.concatLayer):
        self.conv0 = nn.Conv3d(2*num_filters, num_filters, 3)
      else:
        self.conv0 = nn.Conv3d(num_filters, num_filters, 3)
      self.conv1 = nn.Conv3d(num_filters, num_filters, 3)
      
      if (self.useBatchNorm):
        self.batch0 = nn.BatchNorm3d(num_filters)
        self.batch1 = nn.BatchNorm3d(num_filters)
      
  def forward(self, x, encodeTensor):
    decoder = self.deconv0(x)
    if (self.concatLayer):
      decoder = torch.cat((decoder,encodeTensor),1)
    else:
      decoder = torch.add(decoder,encodeTensor)
    
    decoder = nn.functional.pad(decoder, self.padVals, 'replicate')
    decoder = self.conv0(decoder)
    if (self.useBatchNorm):
      decoder = self.batch0(decoder)
      
    decoder = nn.functional.relu(decoder)
    
    decoder = nn.functional.pad(decoder, self.padVals, 'replicate')
    decoder = self.conv1(decoder)
    if (self.useBatchNorm):
      decoder = self.batch1(decoder)
    
    decoder = nn.functional.relu(decoder)
    
    return decoder
  
