import src.LossFunctions as lf
import src.NetOptimizer as netOpt
import numpy as np
import torch
import sys
import SimpleITK as sitk
from numpy import dtype

class CycleLossTests():
  
  def loadDefField(self, fileName):
    defFieldITK = sitk.ReadImage(str(fileName))
    defField = sitk.GetArrayFromImage(defFieldITK)
    return defField
  
  def loadImage(self, fileName):
    defFieldITK = sitk.ReadImage(str(fileName))
    defField = sitk.GetArrayFromImage(defFieldITK)
    return defField  
  
  def cycleLossTest0(self):
    vf0FileName = '/home/fechter/workspace/TorchSandbox/resources/CycleLossFields/vf0.nrrd'
    vf1FileName = '/home/fechter/workspace/TorchSandbox/resources/CycleLossFields/vf1.nrrd'
    vf2FileName = '/home/fechter/workspace/TorchSandbox/resources/CycleLossFields/vf2.nrrd'
    vf3FileName = '/home/fechter/workspace/TorchSandbox/resources/CycleLossFields/vf3.nrrd'
    
    imgFileName0 = '/home/fechter/workspace/TorchSandbox/resources/CycleLossFields/img0.nrrd'
    imgFileName1 = '/home/fechter/workspace/TorchSandbox/resources/CycleLossFields/img1.nrrd'
    imgFileName2 = '/home/fechter/workspace/TorchSandbox/resources/CycleLossFields/img2.nrrd'
    imgFileName3 = '/home/fechter/workspace/TorchSandbox/resources/CycleLossFields/img3.nrrd'
    
    defFields = []
    
    vf0 = self.loadDefField(vf0FileName)
    vf1 = self.loadDefField(vf1FileName)
    vf2 = self.loadDefField(vf2FileName)
    vf3 = self.loadDefField(vf3FileName)
    
    mask0 = np.zeros(vf0.shape)
    mask0[27,30,32] = 1.0
    vf0 = vf0 * mask0
    
    mask1 = np.zeros(vf1.shape)
    mask1[28,30,34] = 1.0
    vf1 = vf1 * mask1
    
    mask2 = np.zeros(vf2.shape)
    mask2[28,32,34] = 1.0
    vf2 = vf2 * mask2
    
    mask3 = np.zeros(vf3.shape)
    mask3[27,32,32] = 1.0
    vf3 = vf3 * mask3
    
    defFields.append(vf0)
    defFields.append(vf1)
    defFields.append(vf2)
    defFields.append(vf3)
    defFields = np.concatenate(defFields,axis=-1).astype('float32')
    defFields = np.expand_dims(defFields, axis=0)
    defFields = np.moveaxis(defFields, -1, 1)
    defFields = torch.from_numpy(defFields)
    
    imgDataToWork = []
    imgDataToWork.append(self.loadImage(imgFileName0))
    imgDataToWork.append(self.loadImage(imgFileName1))
    imgDataToWork.append(self.loadImage(imgFileName2))
    imgDataToWork.append(self.loadImage(imgFileName3))
    imgDataToWork = np.stack(imgDataToWork).astype('float32')
    imgDataToWork = np.expand_dims(imgDataToWork, axis=0)
    imgDataToWork = torch.from_numpy(imgDataToWork)
    
    
    lossCalculator = lf.LossFunctions(imgDataToWork, defFields, defFields, (0.97,0.97,2.5))
    
    zeroIndices = torch.from_numpy( np.indices([imgDataToWork.shape[0],3,imgDataToWork.shape[2],imgDataToWork.shape[3],imgDataToWork.shape[4]],dtype=np.float32) )
    zeroIndices[1] -= 3.0 
    cycleImgData = torch.empty(defFields.shape, device=torch.device("cpu"))
    netOptim = netOpt.NetOptimizer(None, None, None, None)
    for chanIdx in range(-1, imgDataToWork.shape[1] - 1):
      chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
      
      netOptim.cycleLossCalculations(zeroIndices, cycleImgData, defFields, imgDataToWork.shape, chanRange)
      
       
#       cycleImgData[:,chanRange, ] = torch.nn.functional.grid_sample(defFields[:,chanRange, ], cycleIdxData, mode='bilinear', padding_mode='border')
                   
#       cycleIdxData[..., 0] = cycleIdxData[..., 0] + defFields[:, chanIdx * 3, ].detach() / ((imgToDef.shape[4]-1) / 2.0)
#       cycleIdxData[..., 1] = cycleIdxData[..., 1] + defFields[:, chanIdx * 3 + 1, ].detach() / ((imgToDef.shape[3]-1) / 2.0)
#       cycleIdxData[..., 2] = cycleIdxData[..., 2] + defFields[:, chanIdx * 3 + 2, ].detach() / ((imgToDef.shape[2]-1) / 2.0)
    
#     del cycleIdxData
    
    cycleLoss = lossCalculator.cycleLoss(cycleImgData, torch.device("cpu"))    

    return True
    
def main(argv):
  
  lossTests = CycleLossTests()
  
  cycleTest0 = lossTests.cycleLossTest0()
  
  if cycleTest0:
    print("tests passed")
  else:
    print("tests failed")
    
if __name__ == "__main__":
  main(sys.argv[1:]) 
    