import torch
import numpy as np
import Utils

class LossFunctions():
  
  def __init__(self, imgDataToWork, defFields, currDefFields, spacing):
    self.imgData = imgDataToWork
    self.defFields = defFields
    self.currDefFields = currDefFields
    self.dimWeight = spacing

  def dice_coeff(self, y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = torch.reshape(y_true, (-1,))
    y_pred_f = torch.reshape(y_pred, (-1,))
    intersection = torch.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score


  def dice_loss(self, y_true, y_pred):
      loss = 1 - self.dice_coeff(y_true, y_pred)
      return loss

  #TODO:   
  def cycleLoss(self, vecFields, device0):
    loss = torch.empty(vecFields.shape[0], device=device0)
    for imgIdx in range(vecFields.shape[0]):
      vecField = vecFields[imgIdx]
       
      dir0Idx = range(0,vecField.shape[0], 3)
      dir1Idx = range(1,vecField.shape[0], 3)
      dir2Idx = range(2,vecField.shape[0], 3)
       
      dir0Sum = torch.abs(torch.sum(vecField[dir0Idx,],dim=0))
      dir1Sum = torch.abs(torch.sum(vecField[dir1Idx,],dim=0))
      dir2Sum = torch.abs(torch.sum(vecField[dir2Idx,],dim=0))
       
      loss[imgIdx] = torch.mean(dir0Sum + dir1Sum + dir2Sum)
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
      loss000 = torch.abs(currDefFields[:,:,idx[0]-1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,0,:,:])
      loss001 = torch.abs(currDefFields[:,:,idx[0]-1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,1,:,:]) * 0.8
      loss002 = torch.abs(currDefFields[:,:,idx[0]-1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,2,:,:]) * 0.6
      loss003 = torch.abs(currDefFields[:,:,idx[0]-1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,3,:,:]) * 0.4
      loss004 = torch.abs(currDefFields[:,:,idx[0]-1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,4,:,:]) * 0.2
      loss00 = loss000 + loss001 + loss002 + loss003 + loss004
    if idx[0] < currDefFields.shape[2] - defFields.shape[2]:
      loss010 = torch.abs(currDefFields[:,:,idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,-1,:,:])
      loss011 = torch.abs(currDefFields[:,:,idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,-2,:,:]) * 0.8
      loss012 = torch.abs(currDefFields[:,:,idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,-3,:,:]) * 0.6
      loss013 = torch.abs(currDefFields[:,:,idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,-4,:,:]) * 0.4
      loss014 = torch.abs(currDefFields[:,:,idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,-5,:,:]) * 0.2
      loss01 = loss010 + loss011 + loss012 + loss013 + loss014
    loss0 = torch.sum(loss00 + loss01) / self.dimWeight[0]
    
    if idx[1] > 0:
      loss100 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]-1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,0,:])
      loss101 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]-1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,1,:]) * 0.8
      loss102 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]-1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,2,:]) * 0.6
      loss103 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]-1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,3,:]) * 0.4
      loss104 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]-1,idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,4,:]) * 0.2
      loss10 = loss100 + loss101 + loss102 + loss103 + loss104
    if idx[1] < currDefFields.shape[3] - defFields.shape[3]:
      loss110 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,-1,:])
      loss111 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,-2,:]) * 0.8
      loss112 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,-3,:]) * 0.6
      loss113 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,-4,:]) * 0.4
      loss114 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]] - defFields[:,:,:,-5,:]) * 0.2
      loss11 = loss110 + loss111 + loss112 + loss113 + loss114
    loss1 = torch.sum(loss10 + loss11) / self.dimWeight[1]
      
    if idx[2] > 0:
      loss200 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]-1] - defFields[:,:,:,:,0])
      loss201 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]-1] - defFields[:,:,:,:,1]) * 0.8
      loss202 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]-1] - defFields[:,:,:,:,2]) * 0.6
      loss203 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]-1] - defFields[:,:,:,:,3]) * 0.4
      loss204 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]-1] - defFields[:,:,:,:,4]) * 0.2
      loss20 = loss200 + loss201 + loss202 + loss203 + loss204
    if idx[2] < currDefFields.shape[4] - defFields.shape[4]:
      loss210 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]+defFields.shape[4]] - defFields[:,:,:,:,-1])
      loss211 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]+defFields.shape[4]] - defFields[:,:,:,:,-2]) * 0.8
      loss212 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]+defFields.shape[4]] - defFields[:,:,:,:,-3]) * 0.6
      loss213 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]+defFields.shape[4]] - defFields[:,:,:,:,-4]) * 0.4
      loss214 = torch.abs(currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]+defFields.shape[4]] - defFields[:,:,:,:,-5]) * 0.2
      loss21 = loss210 + loss211 + loss212 + loss213 + loss214
    loss2 = torch.sum(loss20 + loss21) / self.dimWeight[2]
      
    loss = torch.sum(loss0 + loss1 + loss2) / (defFields.shape[2]*defFields.shape[3]*defFields.shape[4])
    return loss / defFields.shape[0]

  def getSmoothnessForDir1(self, imgidx, device):
    
    vecField = self.defFields[imgidx]
    idx = np.arange(0,vecField.shape[1]-1)
    
    t = vecField[:,idx+1,:,:].detach()
    loss10 = Utils.getLoss10(vecField.shape, device)
    loss10[:,idx,:,:] = torch.abs(t -vecField[:,idx,:,:])# * weights
    
    t = vecField[:,idx,:,:].detach()
    loss11 = Utils.getLoss11(vecField.shape, device)
    loss11[:,idx+1,:,:] = torch.abs(t - vecField[:,idx+1,:,:])# * weights
    loss1 = (loss10 + loss11) / self.dimWeight[0]
    
    return loss1
  
  def getSmoothnessForDir2(self, imgidx, device):
    vecField = self.defFields[imgidx]
    idx = np.arange(0,vecField.shape[2]-1)
    
    t = vecField[:,:,idx+1,:].detach()
    loss20 = Utils.getLoss20(vecField.shape, device)
    loss20[:,:,idx,:] = torch.abs(t - vecField[:,:,idx,:])# * weights
    
    t = vecField[:,:,idx,:].detach()
    loss21 = Utils.getLoss21(vecField.shape, device)
    loss21[:,:,idx+1,:] = torch.abs(t - vecField[:,:,idx+1,:])# * weights
    loss2 = (loss20 + loss21) / self.dimWeight[1]
    return loss2
  
  def getSmoothnessForDir3(self, imgidx, device):
    vecField = self.defFields[imgidx]
    idx = np.arange(0,vecField.shape[3]-1)
    
    t = vecField[:,:,:,idx+1].detach()
    loss30 = Utils.getLoss30(vecField.shape, device)
    loss30[:,:,:,idx] = torch.abs(t - vecField[:,:,:,idx])# * weights
    
    t = vecField[:,:,:,idx].detach()
    loss31 = Utils.getLoss31(vecField.shape, device)
    loss31[:,:,:,idx+1] = torch.abs(t - vecField[:,:,:,idx+1])# * weights
    loss3 = (loss30 + loss31) / self.dimWeight[2]
    return loss3
  
  def smoothnessVecField(self, device):
    loss = torch.empty(self.defFields.shape[0], device=device)
    for imgIdx in range(self.defFields.shape[0]):
      vecField = self.defFields[imgIdx]
      
      loss1 = self.getSmoothnessForDir1(imgIdx, device)
      
      loss2 = self.getSmoothnessForDir2(imgIdx, device)
      
      loss3 = self.getSmoothnessForDir3(imgIdx, device)
      
      loss[imgIdx] = torch.sum(loss1 + loss2 + loss3) / (vecField.shape[1]*vecField.shape[2]*vecField.shape[3])
    return loss.sum() / self.defFields.shape[0]

  def smoothnessVecFieldT(self, device):
    loss = torch.empty(self.defFields.shape[0], device=device)
    for imgIdx in range(self.defFields.shape[0]):
      vecField = self.defFields[imgIdx]
  
      idx = np.roll(range(0,vecField.shape[0]),-3)
      t = vecField[idx,:,:,:].detach()
      loss0 = torch.abs(t - vecField)
      
      loss1 = self.getSmoothnessForDir1(imgIdx, device)
      
      loss2 = self.getSmoothnessForDir2(imgIdx, device)
      
      loss3 = self.getSmoothnessForDir3(imgIdx, device)
      
      loss[imgIdx] = torch.sum(loss0 + loss1 + loss2 + loss3) / (vecField.shape[1]*vecField.shape[2]*vecField.shape[3])
      
     
    return loss.sum() / self.defFields.shape[0]
  
  ## images must have the same shape
  def normCrossCorr(self, defImg):
    result = 0
    for imgIdx in range(self.imgData.shape[0]):
      for chanIdx in range(self.imgData.shape[1]):
        x = self.imgData[imgIdx, chanIdx,]
        y = defImg[imgIdx, chanIdx,]
        x = torch.reshape(x, (-1,))
        y = torch.reshape(y, (-1,))
        x = torch.nn.functional.normalize(x,2,-1)
        y = torch.nn.functional.normalize(y,2,-1)
        dotProd = torch.dot(x,y) + 1
        result = result + dotProd
    return 1 - (result / (2 * self.imgData.shape[0] * self.imgData.shape[1]))

  def vecLength(self, defField):
    tmp0 = defField[:,range(0,defField.shape[1],3),] * defField[:,range(0,defField.shape[1],3),]
    tmp1 = defField[:,range(1,defField.shape[1],3),] * defField[:,range(1,defField.shape[1],3),]
    tmp2 = defField[:,range(2,defField.shape[1],3),] * defField[:,range(2,defField.shape[1],3),]
    tmpSum = tmp0 + tmp1 + tmp2
    del tmp0, tmp1, tmp2
    tmpSqrt = torch.sqrt(tmpSum)
    return tmpSqrt.mean()
  
  