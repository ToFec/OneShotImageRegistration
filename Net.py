import torch
import torch.nn as nn

class Net(nn.Module):
  def __init__(self, useBatchNorm=True, concatLayer=True):
      super(Net, self).__init__()
      
      self.useBatchNorm=useBatchNorm
      self.concatLayer=concatLayer
      self.padVals = (1, 1, 1, 1, 1, 1)

  def forward(self, x):
    # 256
    
    encoder0_pool, encoder0 = self.encoder_block(2, x, 32)
    # 128
    
    encoder1_pool, encoder1 = self.encoder_block(32, encoder0_pool, 64)
    # 64
    
    encoder2_pool, encoder2 = self.encoder_block(64, encoder1_pool, 128)
    # 32
    
    encoder3_pool, encoder3 = self.encoder_block(128, encoder2_pool, 256)
    # 16
    
    encoder4_pool, encoder4 = self.encoder_block(256, encoder3_pool, 512)
    # 8
    
    center = self.conv_block(512, encoder4_pool, 1024)
    # center
    
    decoder4 = self.decoder_block(1024, center, encoder4, 512)
    # 16
    
    decoder3 = self.decoder_block(512, decoder4, encoder3, 256)
    # 32
    
    decoder2 = self.decoder_block(256, decoder3, encoder2, 128)
    # 64
    
    decoder1 = self.decoder_block(128, decoder2, encoder1, 64)
    # 128
    
    decoder0 = self.decoder_block(64, decoder1, encoder0, 32)
    # 256
    
    outputs = nn.Conv3d(32, 3, 1)(decoder0)
      
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