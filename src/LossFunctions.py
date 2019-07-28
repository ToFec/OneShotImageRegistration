import torch
import numpy as np
import Utils
import GaussSmoothing as gs

class LossFunctions():
  
  def __init__(self, imgDataToWork, defFields, currDefFields, spacing):
    self.imgData = imgDataToWork
    self.defFields = defFields
    self.currDefFields = currDefFields
    self.dimWeight = spacing
    
    self.gaussSmothingKernels = []
    self.gaussSmothingKernels.append(gs.GaussianSmoothing(imgDataToWork.shape[1], 13, 4,3))
    self.gaussSmothingKernels.append(gs.GaussianSmoothing(imgDataToWork.shape[1], 25, 8,3))
    self.gaussSmothingKernels.append(gs.GaussianSmoothing(imgDataToWork.shape[1], 49, 16,3))
    self.diceKernelMapping = {}

  #only for 2 labels
  def dice_coeff(self, y_true, y_pred):
    smooth = 0.000001
    # Flatten
    y_true_f = torch.reshape(y_true, (-1,))
    y_pred_f = torch.reshape(y_pred, (-1,))
    intersection = torch.sum(y_true_f * y_pred_f)
    score = 2. * intersection / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score


  def dice_loss(self, y_true, y_pred):
      loss = 1 - self.dice_coeff(y_true, y_pred)
      return loss
  
  
  
#Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations  
  def multiLabelDiceLoss(self, y_true, y_pred, multiScale = False):
    smooth = 0.000001
    uniqueVals = torch.unique(y_true, sorted=True)
    
    denominator = 0.0
    numerator = 0.0

    for label in uniqueVals[1:]:
      trueSmooth = torch.zeros_like(y_true) - 1.0
      trueSmooth[y_true == label ] = label
      
#       predSmooth = torch.zeros_like(y_pred)
#       predSmooth[y_pred == label] = y_pred[y_pred == label]
#       predSmooth[predSmooth > 0] = 1.0

      intersection = torch.sum(y_pred[y_pred == trueSmooth], dtype=torch.float32) / label
      
#       intersection = torch.sum(trueSmooth * predSmooth, dtype=torch.float32)
      labelSum = (torch.sum(y_pred[y_pred == label], dtype=torch.float32) + torch.sum(y_true[y_true == label], dtype=torch.float32)) / label
      denominator = denominator + labelSum
      numerator = numerator + intersection
      
    dice = 2. * numerator / (denominator + smooth)

    loss =  1 - dice
    if multiScale:
      for gaussKernel in self.gaussSmothingKernels:
        denominator = 0.0
        numerator = 0.0
        for label in uniqueVals[1:]:
          if self.diceKernelMapping.has_key(gaussKernel) and self.diceKernelMapping[gaussKernel].has_key(label):
            trueSmooth = self.diceKernelMapping[gaussKernel][label]
          else:
            trueSmooth = gaussKernel(y_true * (y_true==label).to(torch.float32))
            if self.diceKernelMapping.has_key(gaussKernel):
              self.diceKernelMapping[gaussKernel][label] = trueSmooth
            else:
              self.diceKernelMapping[gaussKernel] = {label: trueSmooth}
              
          predSmooth = y_pred * (y_pred == label).to(torch.float32)
          predSmooth = gaussKernel(predSmooth)          
          intersection = torch.sum(trueSmooth * predSmooth, dtype=torch.float32) / label
          labelSum = torch.sum(trueSmooth, dtype=torch.float32) + torch.sum(predSmooth,dtype=torch.float32) / label
          denominator = denominator + labelSum
          numerator = numerator + intersection
        dice = 2. * numerator / (denominator + smooth)
        loss = loss + (1 - dice)
      return loss / (len(self.gaussSmothingKernels) + 1.0)
    else:
      return loss
  
    
#     Weakly-supervised convolutional neural networks for multimodal image registration
  def multiScaleDiceLoss(self, y_true, y_pred):
    currDscLoss = self.dice_loss(y_true, y_pred)
    for gaussKernel in self.gaussSmothingKernels:
      if self.diceKernelMapping.has_key(gaussKernel):
        trueSmooth = self.diceKernelMapping[gaussKernel]
      else:
        trueSmooth = gaussKernel(y_true)
        self.diceKernelMapping[gaussKernel] = trueSmooth
      predSmooth = gaussKernel(y_pred)
      currDscLoss = currDscLoss + self.dice_loss(trueSmooth, predSmooth)
      
    return currDscLoss / (len(self.gaussSmothingKernels) + 1.0)
      

  #TODO:   
  def cycleLoss(self, vecFields,outOfBoundsTensor, device0):
    loss = torch.empty(vecFields.shape[0], device=device0)
    for imgIdx in range(vecFields.shape[0]):
      vecField = vecFields[imgIdx]
      oOBT = outOfBoundsTensor[imgIdx]
       
      dir0Idx = range(0,vecField.shape[0], 3)
      dir1Idx = range(1,vecField.shape[0], 3)
      dir2Idx = range(2,vecField.shape[0], 3)
       
      dir0Sum = torch.sum(vecField[dir0Idx,],dim=0)
      dir0Sum[oOBT[0,:]] = 0
      dir1Sum = torch.sum(vecField[dir1Idx,],dim=0)
      dir1Sum[oOBT[1,:]] = 0
      dir2Sum = torch.sum(vecField[dir2Idx,],dim=0)
      dir2Sum[oOBT[2,:]] = 0
       
      dir0Pow = torch.pow(dir0Sum, 2)
      dir1Pow = torch.pow(dir1Sum, 2)
      dir2Pow = torch.pow(dir2Sum, 2)
       
      loss[imgIdx] = torch.mean(dir0Pow + dir1Pow + dir2Pow)
    return loss.sum() / vecFields.shape[0]
  

  def smoothBoundary(self, idx, device0):
    loss00 = torch.tensor(0.0, device=device0)
    loss01 =torch.tensor(0.0, device=device0)
    loss11= torch.tensor(0.0, device=device0)
    loss10= torch.tensor(0.0, device=device0)
    loss20= torch.tensor(0.0, device=device0)
    loss21 = torch.tensor(0.0, device=device0)
    currDefFields = self.currDefFields
    defFields = self.defFields
    if idx[0] > 0:
      loss000 = currDefFields[:,:,idx[0]-1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,0,:,:]
      loss001 = (currDefFields[:,:,idx[0]-1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,1,:,:]) * 0.8
      loss002 = (currDefFields[:,:,idx[0]-1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,2,:,:]) * 0.6
      loss003 = (currDefFields[:,:,idx[0]-1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,3,:,:]) * 0.4
      loss004 = (currDefFields[:,:,idx[0]-1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,4,:,:]) * 0.2
      loss00 = torch.pow(loss000 + loss001 + loss002 + loss003 + loss004,2)
    if idx[0] < currDefFields.shape[2] - defFields.shape[2]:
      loss010 = (currDefFields[:,:,idx[0]+defFields.shape[2]+1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,-1,:,:])
      loss011 = (currDefFields[:,:,idx[0]+defFields.shape[2]+1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,-2,:,:]) * 0.8
      loss012 = (currDefFields[:,:,idx[0]+defFields.shape[2]+1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,-3,:,:]) * 0.6
      loss013 = (currDefFields[:,:,idx[0]+defFields.shape[2]+1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,-4,:,:]) * 0.4
      loss014 = (currDefFields[:,:,idx[0]+defFields.shape[2]+1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,-5,:,:]) * 0.2
      loss01 = torch.pow(loss010 + loss011 + loss012 + loss013 + loss014,2)
    loss0 = torch.sum(loss00 + loss01) / self.dimWeight[0]
    
    if idx[1] > 0:
      loss100 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]-1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,0,:])
      loss101 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]-1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,1,:]) * 0.8
      loss102 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]-1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,2,:]) * 0.6
      loss103 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]-1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,3,:]) * 0.4
      loss104 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]-1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,4,:]) * 0.2
      loss10 = torch.pow(loss100 + loss101 + loss102 + loss103 + loss104,2)
    if idx[1] < currDefFields.shape[3] - defFields.shape[3]:
      loss110 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]+defFields.shape[3]+1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,-1,:])
      loss111 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]+defFields.shape[3]+1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,-2,:]) * 0.8
      loss112 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]+defFields.shape[3]+1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,-3,:]) * 0.6
      loss113 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]+defFields.shape[3]+1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,-4,:]) * 0.4
      loss114 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]+defFields.shape[3]+1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,-5,:]) * 0.2
      loss11 = torch.pow(loss110 + loss111 + loss112 + loss113 + loss114,2)
    loss1 = torch.sum(loss10 + loss11) / self.dimWeight[1]
      
    if idx[2] > 0:
      loss200 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]-1] - defFields[:,:,:,:,0])
      loss201 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]-1] - defFields[:,:,:,:,1]) * 0.8
      loss202 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]-1] - defFields[:,:,:,:,2]) * 0.6
      loss203 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]-1] - defFields[:,:,:,:,3]) * 0.4
      loss204 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]-1] - defFields[:,:,:,:,4]) * 0.2
      loss20 = torch.pow(loss200 + loss201 + loss202 + loss203 + loss204,2)
    if idx[2] < currDefFields.shape[4] - defFields.shape[4]:
      loss210 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]+defFields.shape[4]+1] - defFields[:,:,:,:,-1])
      loss211 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]+defFields.shape[4]+1] - defFields[:,:,:,:,-2]) * 0.8
      loss212 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]+defFields.shape[4]+1] - defFields[:,:,:,:,-3]) * 0.6
      loss213 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]+defFields.shape[4]+1] - defFields[:,:,:,:,-4]) * 0.4
      loss214 = (currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]+defFields.shape[4]+1] - defFields[:,:,:,:,-5]) * 0.2
      loss21 = torch.pow(loss210 + loss211 + loss212 + loss213 + loss214,2)
    loss2 = torch.sum(loss20 + loss21) / self.dimWeight[2]
      
    loss = torch.sum(loss0 + loss1 + loss2) / (defFields.shape[2]*defFields.shape[3]*defFields.shape[4])
    return loss / defFields.shape[0]

  def getSmoothnessForDir1(self, imgidx, device):
    
    vecField = self.defFields[imgidx]
    idx = np.arange(0,vecField.shape[1]-1)
    
    t = vecField[:,idx+1,:,:].detach()
    loss10 = Utils.getLoss10(vecField.shape, device)
    loss10[:,idx,:,:] = t -vecField[:,idx,:,:]# * weights
    
    t = vecField[:,idx,:,:].detach()
    loss11 = Utils.getLoss11(vecField.shape, device)
    loss11[:,idx+1,:,:] = t - vecField[:,idx+1,:,:]# * weights
    loss1 = (loss10 + loss11) / self.dimWeight[0]
     
    return loss1
  
  def getSmoothnessForDir2(self, imgidx, device):
    vecField = self.defFields[imgidx]
    idx = np.arange(0,vecField.shape[2]-1)
    
    t = vecField[:,:,idx+1,:].detach()
    loss20 = Utils.getLoss20(vecField.shape, device)
    loss20[:,:,idx,:] = t - vecField[:,:,idx,:]# * weights
    
    t = vecField[:,:,idx,:].detach()
    loss21 = Utils.getLoss21(vecField.shape, device)
    loss21[:,:,idx+1,:] = t - vecField[:,:,idx+1,:]# * weights
    loss2 = (loss20 + loss21) / self.dimWeight[1]
    return loss2
  
  def getSmoothnessForDir3(self, imgidx, device):
    vecField = self.defFields[imgidx]
    idx = np.arange(0,vecField.shape[3]-1)
    
    t = vecField[:,:,:,idx+1].detach()
    loss30 = Utils.getLoss30(vecField.shape, device)
    loss30[:,:,:,idx] = t - vecField[:,:,:,idx]# * weights
    
    t = vecField[:,:,:,idx].detach()
    loss31 = Utils.getLoss31(vecField.shape, device)
    loss31[:,:,:,idx+1] = t - vecField[:,:,:,idx+1]# * weights
    loss3 = (loss30 + loss31) / self.dimWeight[2]
    return loss3
  
  def smoothnessVecField(self, device):
    loss = torch.empty(self.defFields.shape[0], device=device)
    for imgIdx in range(self.defFields.shape[0]):
      vecField = self.defFields[imgIdx]
      
      loss1 = self.getSmoothnessForDir1(imgIdx, device)
      
      loss2 = self.getSmoothnessForDir2(imgIdx, device)
#        
      loss3 = self.getSmoothnessForDir3(imgIdx, device)
      
      loss[imgIdx] = torch.sum(torch.pow(loss1 + loss2 + loss3,2)) / (vecField.shape[1]*vecField.shape[2]*vecField.shape[3])
    return loss.sum() / self.defFields.shape[0]

  def smoothnessVecFieldT(self, device):
    loss = torch.empty(self.defFields.shape[0], device=device)
    for imgIdx in range(self.defFields.shape[0]):
      vecField = self.defFields[imgIdx]
  
      #idx = np.roll(range(0,vecField.shape[0]),-3)
      #t = vecField[idx,:,:,:].detach()
      #loss0 = torch.abs(t - vecField)
      
      loss1 = self.getSmoothnessForDir1(imgIdx, device)
      
      loss2 = self.getSmoothnessForDir2(imgIdx, device)
      
      loss3 = self.getSmoothnessForDir3(imgIdx, device)
      
#       loss[imgIdx] = torch.sum(loss0 + loss1 + loss2 + loss3) / (vecField.shape[1]*vecField.shape[2]*vecField.shape[3]*((vecField.shape[0]/3)-1))
      loss[imgIdx] = torch.sum(torch.pow(loss1 + loss2 + loss3,2)) / (vecField.shape[1]*vecField.shape[2]*vecField.shape[3]*((vecField.shape[0]/3)-1))
     
    return loss.sum() / self.defFields.shape[0]
  
  ## images must have the same shape
  def normCrossCorr(self, defImg, device0):
    results = torch.empty(self.imgData.shape[0]*self.imgData.shape[1], device=device0)
    for imgIdx in range(self.imgData.shape[0]):
      for chanIdx in range(self.imgData.shape[1]):
        x = self.imgData[imgIdx, chanIdx,]
        y = defImg[imgIdx, chanIdx,]
        x = torch.reshape(x, (-1,))
        y = torch.reshape(y, (-1,))
        x = torch.nn.functional.normalize(x,2,-1)
        y = torch.nn.functional.normalize(y,2,-1)
        dotProd = torch.dot(x,y) + 1
        results[imgIdx * self.imgData.shape[0] + chanIdx] = dotProd
    return 1 - (torch.sum(results) / (2 * self.imgData.shape[0] * self.imgData.shape[1]))

  def vecLength(self, defField):
    tmp0 = defField[:,range(0,defField.shape[1],3),] * defField[:,range(0,defField.shape[1],3),]
    tmp1 = defField[:,range(1,defField.shape[1],3),] * defField[:,range(1,defField.shape[1],3),]
    tmp2 = defField[:,range(2,defField.shape[1],3),] * defField[:,range(2,defField.shape[1],3),]
    tmpSum = tmp0 + tmp1 + tmp2
    del tmp0, tmp1, tmp2
    tmpSqrt = torch.sqrt(tmpSum)
    return tmpSqrt.mean()
  
  