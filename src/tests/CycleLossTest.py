import src.LossFunctions as lf
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
    fileName0 = '/home/fechter/workspace/TorchSandbox/results/DirLab08/deformationFieldDataset0image0channel0.nrrd'
    fileName1 = '/home/fechter/workspace/TorchSandbox/results/DirLab08/deformationFieldDataset0image0channel-1.nrrd'
    fileName2 = '/home/fechter/workspace/TorchSandbox/results/DirLab08/origImgDataset0image0channel0.nrrd'
    fileName3 = '/home/fechter/workspace/TorchSandbox/results/DirLab08/origImgDataset0image0channel-1.nrrd'
    
    defFields = []
    defFields.append(self.loadDefField(fileName0))
    defFields.append(self.loadDefField(fileName1))
    defFields = np.concatenate(defFields,axis=-1).astype('float32')
    defFields = np.expand_dims(defFields, axis=0)
    defFields = np.moveaxis(defFields, -1, 1)
    defFields = torch.from_numpy(defFields)
    
    imgDataToWork = []
    imgDataToWork.append(self.loadImage(fileName2))
    imgDataToWork.append(self.loadImage(fileName3))
    imgDataToWork = np.stack(imgDataToWork).astype('float32')
    imgDataToWork = np.expand_dims(imgDataToWork, axis=0)
    imgDataToWork = torch.from_numpy(imgDataToWork)
    
    
    lossCalculator = lf.LossFunctions(imgDataToWork, defFields, defFields, (0.97,0.97,2.5))
    
    zeroIndices = torch.from_numpy( np.indices([imgDataToWork.shape[0],3,imgDataToWork.shape[2],imgDataToWork.shape[3],imgDataToWork.shape[4]],dtype=np.float32) )
    zeroIndices[1] -= 3.0 
    cycleImgData = torch.empty(defFields.shape, device=torch.device("cpu"))
    for chanIdx in range(-1, imgDataToWork.shape[1] - 1):
      chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
      
      fieldsLow4 = zeroIndices[4].long()
      partHigh4 = zeroIndices[4] - fieldsLow4.float()
      partLow4 = 1.0 - partHigh4
      fieldsHigh4 = fieldsLow4 + 1
      fieldsHigh4[fieldsHigh4 > (imgDataToWork.shape[4] - 1)] = imgDataToWork.shape[4] - 1
      fieldsLow4[fieldsLow4 > (imgDataToWork.shape[4] - 1)] = imgDataToWork.shape[4] - 1      
      
      fieldsLow3 = zeroIndices[3].long()
      partHigh3 = zeroIndices[3] - fieldsLow3.float()
      partLow3 = 1.0 - partHigh3
      fieldsHigh3 = fieldsLow3 + 1
      fieldsHigh3[fieldsHigh3 > (imgDataToWork.shape[3] - 1)] = imgDataToWork.shape[3] - 1
      fieldsLow3[fieldsLow3 > (imgDataToWork.shape[3] - 1)] = imgDataToWork.shape[3] - 1  
      
      fieldsLow2 = zeroIndices[2].long()
      partHigh2 = zeroIndices[2] - fieldsLow2.float()
      partLow2 = 1.0 - partHigh2
      fieldsHigh2 = fieldsLow2 + 1
      fieldsHigh2[fieldsHigh2 > (imgDataToWork.shape[2] - 1)] = imgDataToWork.shape[2] - 1
      fieldsLow2[fieldsLow2 > (imgDataToWork.shape[2] - 1)] = imgDataToWork.shape[2] - 1  
      
      fields0 = zeroIndices[0].long()
      fields1 = zeroIndices[1].long()
      
      cycleImgData[:,chanRange, ] = partLow2 * partLow3 * partLow4 * defFields[fields0,fields1,fieldsLow2, fieldsLow3, fieldsLow4] + \
      partLow2 * partLow3 * partHigh4 * defFields[fields0,fields1,fieldsLow2, fieldsLow3, fieldsHigh4] + \
      partLow2 * partHigh3 * partLow4 * defFields[fields0,fields1,fieldsLow2, fieldsHigh3, fieldsLow4] + \
      partHigh2 * partLow3 * partLow4 * defFields[fields0,fields1,fieldsHigh2, fieldsLow3, fieldsLow4] + \
      partHigh2 * partHigh3 * partLow4 * defFields[fields0,fields1,fieldsHigh2, fieldsHigh3, fieldsLow4] + \
      partHigh2 * partLow3 * partHigh4 * defFields[fields0,fields1,fieldsHigh2, fieldsLow3, fieldsHigh4] + \
      partLow2 * partHigh3 * partHigh4 * defFields[fields0,fields1,fieldsLow2, fieldsHigh3, fieldsHigh4] + \
      partHigh2 * partHigh3 * partHigh4 * defFields[fields0,fields1,fieldsHigh2, fieldsHigh3, fieldsHigh4]
      
      zeroIndices[1] += 3.0
      
      ##take care of def vec order !!!
      tmpField = cycleImgData[:,None,chanRange[2],].detach()
      zeroIndices[2][:,None,0,] += tmpField
      zeroIndices[2][:,None,1,] += tmpField
      zeroIndices[2][:,None,2,] += tmpField
      
      tmpField = cycleImgData[:,None,chanRange[1],].detach()
      zeroIndices[3][:,None,0,] += tmpField
      zeroIndices[3][:,None,1,] += tmpField
      zeroIndices[3][:,None,2,] += tmpField
      
      tmpField = cycleImgData[:,None,chanRange[0],].detach()
      zeroIndices[4][:,None,0,] += tmpField
      zeroIndices[4][:,None,1,] += tmpField
      zeroIndices[4][:,None,2,] += tmpField      
       
#       cycleImgData[:,chanRange, ] = torch.nn.functional.grid_sample(defFields[:,chanRange, ], cycleIdxData, mode='bilinear', padding_mode='border')
                   
#       cycleIdxData[..., 0] = cycleIdxData[..., 0] + defFields[:, chanIdx * 3, ].detach() / ((imgToDef.shape[4]-1) / 2.0)
#       cycleIdxData[..., 1] = cycleIdxData[..., 1] + defFields[:, chanIdx * 3 + 1, ].detach() / ((imgToDef.shape[3]-1) / 2.0)
#       cycleIdxData[..., 2] = cycleIdxData[..., 2] + defFields[:, chanIdx * 3 + 2, ].detach() / ((imgToDef.shape[2]-1) / 2.0)
    
#     del cycleIdxData
    
    cycleLoss = lossCalculator.cycleLoss(cycleImgData, self.userOpts.device)    

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
    