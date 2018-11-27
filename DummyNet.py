import torch
import torch.nn as nn

class DummyNet(nn.Module):
  def __init__(self, useBatchNorm=True, concatLayer=True):
      super(DummyNet, self).__init__()
      
      self.useBatchNorm=useBatchNorm
      self.concatLayer=concatLayer
      self.padVals = (1, 1, 1, 1, 1, 1)

  def forward(self, x):
    # 256
    
    encoder0_pool, encoder0 = self.encoder_block(2, x, 32)
    # 128
    
    center = self.conv_block(32, encoder0_pool, 64)
    
    
    decoder4 = self.decoder_block(64, center, encoder0, 32)
    # 16
    
    outputs = nn.Conv3d(32, 3, 1)(decoder4)
      
    return outputs
  
  def conv_block(self, input_channels, input_tensor, num_filters):
    encoder = nn.functional.pad(input_tensor, self.padVals, 'replicate')
    encoder = nn.Conv3d(input_channels, num_filters, 3)(encoder)
    if (self.useBatchNorm):
      encoder = nn.BatchNorm3d(num_filters)(encoder)
    encoder = nn.ReLU(encoder)
    
    encoder = nn.functional.pad(encoder, self.padVals, 'replicate')
    encoder = nn.Conv3d(num_filters, num_filters, 3)(encoder)
    
    if (self.useBatchNorm):
      encoder = nn.BatchNorm3d(num_filters)(encoder)
    
    encoder = nn.ReLU(encoder)

    return encoder

  def encoder_block(self, input_channels, input_tensor, num_filters):
    encoder = self.conv_block(input_channels, input_tensor, num_filters)
    encoder_pool = nn.MaxPool3d(2, 2)(encoder)
    
    return encoder_pool, encoder
  
  def decoder_block(self, input_channels, input_tensor, concat_tensor, num_filters):
    p1d = (1, 1, 1, 1, 1, 1)
    decoder = nn.functional.pad(input_tensor, p1d, 'replicate')
    decoder = nn.ConvTranspose3d(input_channels, num_filters, 2, 2)(decoder)
    
    if (self.concatLayer):
      decoder = torch.cat((decoder,concat_tensor),1)
      input_channels = input_channels + concat_tensor.shape[1]
    else:
      decoder = torch.add(decoder,concat_tensor)
    
    
    if (self.useBatchNorm):
      decoder = nn.BatchNorm3d(num_filters)(decoder)
    
    decoder = nn.ReLU(decoder)
    decoder = nn.functional.pad(decoder, self.padVals, 'replicate')
    decoder = nn.Conv3d(num_filters, num_filters, 3)(decoder)
    
    if (self.useBatchNorm):
      decoder = nn.BatchNorm3d(num_filters)(decoder)
    
    decoder = nn.ReLU(decoder)
    decoder = nn.functional.pad(decoder, self.padVals, 'replicate')
    decoder = nn.Conv3d(num_filters, num_filters, 3)(decoder)
    
    if (self.useBatchNorm):
      decoder = nn.BatchNorm3d(num_filters)(decoder)
    
    decoder = nn.ReLU(decoder)
    
    return decoder