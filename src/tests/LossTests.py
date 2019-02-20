import LossFunctions as lf
import numpy as np
import Utils
import sys
import Optimize as optim
import Options as userOpts
import torch
from Net import UNet

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
    
  def similarityLossTest(self):
    currDefField = torch.ones([1,3,16,16,16])*2
    defField = torch.ones([1,3,8,8,8])
    
    loss1 = lf.smoothBoundary(defField, currDefField, [8,8,8])
    
    loss2 = lf.smoothBoundary(defField, currDefField, [1,1,1])
    
    loss3 = lf.smoothBoundary(defField, currDefField, [0,0,0])
            
    if loss1 == 1.1250 and loss2 == 2.2500 and loss3 == 1.1250:
      return True
    else:
      return False

    
def main(argv):
  lossTests = LossTests()
  
  lossTests.similarityLossTest()
  
  ccTest = lossTests.crossCorTest()
    
if __name__ == "__main__":
  main(sys.argv[1:]) 
    