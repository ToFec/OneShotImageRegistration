import SimpleITK as sitk
import LossFunctions as lf
import torch
import numpy as np
import Utils
import sys

class LossTests():
  def crossCorTest(self):
    fileName0 = '/home/fechter/workspace/TorchSandbox/resources/Data01/img1.nii.gz'
    fileName1 = '/home/fechter/workspace/TorchSandbox/resources/Data01/img0.nii.gz'
    img0 = Utils.loadImage(fileName0)
    img1 = Utils.loadImage(fileName1)
    
    cc0 = lf.normCrossCorr(img0, img0)
    cc1 = lf.normCrossCorr(img0, img1)
    if (np.isclose(cc0,-1.0)):
      return True
    else:
      return False
    
def main(argv):
  lossTests = LossTests()
  
  ccTest = lossTests.crossCorTest()
    
if __name__ == "__main__":
  main(sys.argv[1:]) 
    