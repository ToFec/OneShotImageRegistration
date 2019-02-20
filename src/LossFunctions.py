import torch
import numpy as np

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = torch.reshape(y_true, (-1,))
    y_pred_f = torch.reshape(y_pred, (-1,))
    intersection = torch.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

#TODO:   
def cycleLoss(vecFields, device0):
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
  

def smoothBoundary(defFields, currDefFields, idx, device0):
  loss00 = torch.tensor(0.0, device=device0)
  loss01 =torch.tensor(0.0, device=device0)
  loss11= torch.tensor(0.0, device=device0)
  loss10= torch.tensor(0.0, device=device0)
  loss20= torch.tensor(0.0, device=device0)
  loss21 = torch.tensor(0.0, device=device0)
  if idx[0] > 0:
    loss00 = torch.abs(defFields[:,:,0,:,:] - currDefFields[:,:,idx[0]-1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]])
  if idx[0] < currDefFields.shape[2] - defFields.shape[2]:
    loss01 = torch.abs(defFields[:,:,-1,:,:] - currDefFields[:,:,idx[0]+defFields.shape[2]+1,idx[1]:idx[1]+defFields.shape[3],idx[2]:idx[2]+defFields.shape[4]])
  loss0 = torch.sum(loss00 + loss01)
  
  if idx[1] > 0:
    loss10 = torch.abs(defFields[:,:,:,0,:] - currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]-1,idx[2]:idx[2]+defFields.shape[4]])
  if idx[1] < currDefFields.shape[3] - defFields.shape[3]:
    loss11 = torch.abs(defFields[:,:,:,-1,:] - currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]+defFields.shape[3]+1,idx[2]:idx[2]+defFields.shape[4]])
  loss1 = torch.sum(loss10 + loss11)
    
  if idx[2] > 0:
    loss20 = torch.abs(defFields[:,:,:,:,0] - currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]-1])
  if idx[2] < currDefFields.shape[4] - defFields.shape[4]:
    loss21 = torch.abs(defFields[:,:,:,:,-1] - currDefFields[:,:,idx[0]:idx[0]+defFields.shape[2],idx[1]:idx[1]+defFields.shape[3],idx[2]+defFields.shape[4]+1])
  loss2 = torch.sum(loss20 + loss21)   
    
  loss = torch.sum(loss0 + loss1 + loss2) / (defFields.shape[2]*defFields.shape[3]*defFields.shape[4])
  return loss / defFields.shape[0]

def smoothnessVecField(vecFields, device):
  loss = torch.empty(vecFields.shape[0], device=device)
  for imgIdx in range(vecFields.shape[0]):
    vecField = vecFields[imgIdx]
    
    idx = np.arange(0,vecField.shape[1]-1)
    t = vecField[:,idx+1,:,:].detach()
    loss10 = torch.zeros(vecField.shape, device=device, requires_grad=False)
    loss10[:,idx,:,:] = torch.abs(t - vecField[:,idx,:,:])
    
    t = vecField[:,idx,:,:].detach()
    loss11 = torch.zeros(vecField.shape, device=device, requires_grad=False)
    loss11[:,idx+1,:,:] = torch.abs(t - vecField[:,idx+1,:,:])
    loss1 = loss10 + loss11
    
    idx = np.arange(0,vecField.shape[2]-1)
    t = vecField[:,:,idx+1,:].detach()
    loss20 = torch.zeros(vecField.shape, device=device, requires_grad=False)
    loss20[:,:,idx,:] = torch.abs(t - vecField[:,:,idx,:])
    
    t = vecField[:,:,idx,:].detach()
    loss21 = torch.zeros(vecField.shape, device=device, requires_grad=False)
    loss21[:,:,idx+1,:] = torch.abs(t - vecField[:,:,idx+1,:])
    loss2 = loss20 + loss21
    
    idx = np.arange(0,vecField.shape[3]-1)
    t = vecField[:,:,:,idx+1].detach()
    loss30 = torch.zeros(vecField.shape, device=device, requires_grad=False)
    loss30[:,:,:,idx] = torch.abs(t - vecField[:,:,:,idx])
    
    t = vecField[:,:,:,idx].detach()
    loss31 = torch.zeros(vecField.shape, device=device, requires_grad=False)
    loss31[:,:,:,idx+1] = torch.abs(t - vecField[:,:,:,idx+1])
    loss3 = loss30 + loss31
    
    loss[imgIdx] = torch.sum(loss1 + loss2 + loss3) / (vecField.shape[1]*vecField.shape[2]*vecField.shape[3])
  return loss.sum() / vecFields.shape[0]

def smoothnessVecFieldT(vecFields, device):
  loss = torch.empty(vecFields.shape[0], device=device)
  for imgIdx in range(vecFields.shape[0]):
    vecField = vecFields[imgIdx]

    idx = np.roll(range(0,vecField.shape[0]),-3)
    t = vecField[idx,:,:,:].detach()
    loss0 = torch.abs(t - vecField)
    
    idx = np.arange(0,vecField.shape[1]-1)
    t = vecField[:,idx+1,:,:].detach()
    loss10 = torch.zeros(vecField.shape, device=device, requires_grad=False)
    loss10[:,idx,:,:] = torch.abs(t - vecField[:,idx,:,:])
    
    t = vecField[:,idx,:,:].detach()
    loss11 = torch.zeros(vecField.shape, device=device, requires_grad=False)
    loss11[:,idx+1,:,:] = torch.abs(t - vecField[:,idx+1,:,:])
    loss1 = loss10 + loss11
    
    idx = np.arange(0,vecField.shape[2]-1)
    t = vecField[:,:,idx+1,:].detach()
    loss20 = torch.zeros(vecField.shape, device=device, requires_grad=False)
    loss20[:,:,idx,:] = torch.abs(t - vecField[:,:,idx,:])
    
    t = vecField[:,:,idx,:].detach()
    loss21 = torch.zeros(vecField.shape, device=device, requires_grad=False)
    loss21[:,:,idx+1,:] = torch.abs(t - vecField[:,:,idx+1,:])
    loss2 = loss20 + loss21
    
    idx = np.arange(0,vecField.shape[3]-1)
    t = vecField[:,:,:,idx+1].detach()
    loss30 = torch.zeros(vecField.shape, device=device, requires_grad=False)
    loss30[:,:,:,idx] = torch.abs(t - vecField[:,:,:,idx])
    
    t = vecField[:,:,:,idx].detach()
    loss31 = torch.zeros(vecField.shape, device=device, requires_grad=False)
    loss31[:,:,:,idx+1] = torch.abs(t - vecField[:,:,:,idx+1])
    loss3 = loss30 + loss31
    
    loss[imgIdx] = torch.sum(loss0 + loss1 + loss2 + loss3) / (vecField.shape[1]*vecField.shape[2]*vecField.shape[3])
  return loss.sum() / vecFields.shape[0]
  
  ## img0 and img1 must have the same shape
def normCrossCorr(img0, img1):
  result = 0
  for imgIdx in range(img0.shape[0]):
    for chanIdx in range(img0.shape[1]):
      x = img0[imgIdx, chanIdx,]
      y = img1[imgIdx, chanIdx,]
      x = torch.reshape(x, (-1,))
      y = torch.reshape(y, (-1,))
      x = torch.nn.functional.normalize(x,2,-1)
      y = torch.nn.functional.normalize(y,2,-1)
      dotProd = torch.dot(x,y) + 1
      result = result + dotProd
  return 1 - (result / (2 * img0.shape[0] * img0.shape[1]))

def vecLength(defField):
  tmp0 = defField[:,range(0,defField.shape[1],3),] * defField[:,range(0,defField.shape[1],3),]
  tmp1 = defField[:,range(1,defField.shape[1],3),] * defField[:,range(1,defField.shape[1],3),]
  tmp2 = defField[:,range(2,defField.shape[1],3),] * defField[:,range(2,defField.shape[1],3),]
  tmpSum = tmp0 + tmp1 + tmp2
  del tmp0, tmp1, tmp2
  tmpSqrt = torch.sqrt(tmpSum)
  return tmpSqrt.mean()
  
  