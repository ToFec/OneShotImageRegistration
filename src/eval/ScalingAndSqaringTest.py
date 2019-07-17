import getopt, sys, os
import SimpleITK as sitk
import torch
import numpy as np

sys.path.insert(0,os.path.realpath('.'))
import src.Utils

def main(argv):
  vecField = torch.zeros([1,3,10,10,10])
  vecField[:,0,3:7,3:7,3:7] = 1.0
  
#   vecField = vecField/(2**self.num_steps)
  for _ in range(3):
    vecDataDef = torch.empty(vecField.shape, device='cpu', requires_grad=False)
    for chanIdx in range(-1, (vecField.shape[1] /3) - 1):
      chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
      for channel in chanRange:
        imgToDef = vecField[:, None, channel, ]
        deformedTmp = src.Utils.deformImage(imgToDef, vecField[: , chanRange, ], 'cpu', False)
        vecDataDef[:, channel, ] = deformedTmp[:, 0, ]
    vecField.add_(vecDataDef)

  
if __name__ == "__main__":
  main(sys.argv[1:])    