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
def cycleLoss(vecFields, device):
  loss = torch.empty(vecFields.shape[0], device=device)
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
  
def smoothnessVecField(vecFields, device):
  loss = torch.empty(vecFields.shape[0], device=device)
  for imgIdx in range(vecFields.shape[0]):
    vecField = vecFields[imgIdx]

    idx = np.roll(range(0,vecField.shape[1]),-1)
    t = vecField[:,idx,:,:].detach()
    loss1 = torch.abs(t - vecField)
    
    idx = np.roll(range(0,vecField.shape[2]),-1)
    t = vecField[:,:,idx,:].detach()
    loss2 = torch.abs(t - vecField)
    
    idx = np.roll(range(0,vecField.shape[3]),-1)
    t = vecField[:,:,:,idx].detach()
    loss3 = torch.abs(t - vecField)
    
    loss[imgIdx] = torch.sum(loss1 + loss2 + loss3) / (vecField.shape[1]*vecField.shape[2]*vecField.shape[3])
  return loss.sum() / vecFields.shape[0]

def smoothnessVecFieldT(vecFields, device):
  loss = torch.empty(vecFields.shape[0], device=device)
  for imgIdx in range(vecFields.shape[0]):
    vecField = vecFields[imgIdx]

    idx = np.roll(range(0,vecField.shape[0]),-3)
    t = vecField[idx,:,:,:].detach()
    loss0 = torch.abs(t - vecField)
    
    idx = np.roll(range(0,vecField.shape[1]),-1)
    t = vecField[:,idx,:,:].detach()
    loss1 = torch.abs(t - vecField)
    
    idx = np.roll(range(0,vecField.shape[2]),-1)
    t = vecField[:,:,idx,:].detach()
    loss2 = torch.abs(t - vecField)
    
    idx = np.roll(range(0,vecField.shape[3]),-1)
    t = vecField[:,:,:,idx].detach()
    loss3 = torch.abs(t - vecField)
    
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
      #dotProd = dotProd * dotProd
      result = result + dotProd
  return 1 - (result / (2 * img0.shape[0] * img0.shape[1]))
  
  