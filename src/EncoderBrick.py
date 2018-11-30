import torch.nn as nn

class EncoderBrick(nn.Module):
  def __init__(self ,num_filters, in_channels=2, useBatchNorm=True, concatLayer=True):
      super(EncoderBrick, self).__init__()
      
      self.useBatchNorm=useBatchNorm
      self.concatLayer=concatLayer
      self.padVals = (1, 1, 1, 1, 1, 1)
      self.in_channels = in_channels
      
      self.conv0 = nn.Conv3d(in_channels, num_filters, 3)
      self.conv1 = nn.Conv3d(num_filters, num_filters, 3)
      if (self.useBatchNorm):
        self.batch0 = nn.BatchNorm3d(num_filters)
        self.batch1 = nn.BatchNorm3d(num_filters)
      
  def forward(self, x):
    encoder = nn.functional.pad(x, self.padVals, 'replicate')
    encoder = self.conv0(encoder)
    if (self.useBatchNorm):
      encoder = self.batch0(encoder)
    encoder = nn.functional.relu(encoder)
    
    encoder = nn.functional.pad(encoder, self.padVals, 'replicate')
    encoder = self.conv1(encoder)
    if (self.useBatchNorm):
      encoder = self.batch1(encoder)
    encoder = nn.functional.relu(encoder)
    
    return encoder
  
