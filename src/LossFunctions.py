import torch

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
  
def smoothnessVecField(vecFields):
  criterion = torch.nn.SmoothL1Loss()
  loss = torch.tensor([0.])
  for imgIdx in range(vecFields.shape[0]):
    vecField = vecFields[imgIdx]
    idx = range(-1,vecField.shape[0]-1)
    
    t = vecField[idx,:,:,:].detach()
    loss0 = criterion(vecField[:,:,:,:], t)
    
    t = vecField[:,1:,:,:].detach()
    loss1 = criterion(vecField[:,:-1,:,:], t)
    
    t = vecField[:,:,1:,:].detach()
    loss2 = criterion(vecField[:,:,:-1,:], t)
    
    t = vecField[:,:,:,1:].detach()
    loss3 = criterion(vecField[:,:,:,:-1], t)
    
    loss = loss + loss0 + loss1 + loss2 + loss3
  return loss
  
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
  
  