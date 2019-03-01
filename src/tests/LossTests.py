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
    
    lossFunction = lf.LossFunctions([1.0,1.0,1.0])
    
    cc0 = lossFunction.normCrossCorr(img0, img0)
    cc1 = lossFunction.normCrossCorr(img0, img1)
    if (np.isclose(cc0,0.0,rtol=1.e-5, atol=1.e-5)):
      return True
    else:
      return False
    
  def similarityLossTest(self):
    
    lossFunction = lf.LossFunctions([1.0,1.0,1.0])
    currDefField = torch.ones([1,3,16,16,16])*2
    defField = torch.ones([1,3,8,8,8])
    
    loss1 = lossFunction.smoothBoundary(defField, currDefField, [8,8,8], 'cpu')
    
    loss2 = lossFunction.smoothBoundary(defField, currDefField, [1,1,1], 'cpu')
    
    loss3 = lossFunction.smoothBoundary(defField, currDefField, [0,0,0], 'cpu')
            
    if np.isclose(loss1,3.3750,1e-4)  and np.isclose(loss2,6.750,1e-4) and np.isclose(loss3,3.3750,1e-4):
      return True
    else:
      return False

    
def main(argv):
  lossTests = LossTests()
  
  simTest = lossTests.similarityLossTest()
  
  ccTest = lossTests.crossCorTest()
  
  if ccTest and simTest:
    print("tests passed")
  else:
    print("tests failed")
    
if __name__ == "__main__":
  main(sys.argv[1:]) 
    