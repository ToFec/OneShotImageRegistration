
import Utils
import torch
import LossFunctions as lf
import numpy as np

class NetOptimizer(object):

  def __init__(self, net, spacing, optimizer,options):
    self.spacing = spacing
    self.optimizer = optimizer
    self.net = net
    self.userOpts = options
        
  def normalizeWeights(self, ccW, sW, cyW):
    weightSum = ccW + sW + cyW
    return [ccW  / weightSum, sW  / weightSum, cyW  / weightSum]
            
  def optimizeNet(self, imgDataToWork, labelToWork, lastDefField = None, currDefFields = None, idx=None, itIdx=0):
    
    # zero the parameter gradients
    self.optimizer.zero_grad()
        
    defFields = self.net(imgDataToWork)
    
    if (lastDefField is not None) and (idx is not None):
      addedField = lastDefField[:, :, idx[0]:idx[0]+defFields.shape[2], idx[1]:idx[1]+defFields.shape[3], idx[2]:idx[2]+defFields.shape[4]]+ defFields
    else:
      addedField = defFields
      
    if currDefFields is not None:
      currDefFields[:, :, idx[0]:idx[0]+defFields.shape[2], idx[1]:idx[1]+defFields.shape[3], idx[2]:idx[2]+defFields.shape[4]] = addedField.detach()
    
    cropStart0 = (imgDataToWork.shape[2]-defFields.shape[2])/2
    cropStart1 = (imgDataToWork.shape[3]-defFields.shape[3])/2
    cropStart2 = (imgDataToWork.shape[4]-defFields.shape[4])/2
    imgDataToWork = imgDataToWork[:,:,cropStart0:cropStart0+defFields.shape[2], cropStart1:cropStart1+defFields.shape[3], cropStart2:cropStart2+defFields.shape[4]]
    
    lossCalculator = lf.LossFunctions(imgDataToWork, addedField, currDefFields, self.spacing)
    
    boundaryLoss = 0.0
    smoothnessLoss = 0.0
    
    smoothNessWeight = self.userOpts.smoothW[itIdx]
    crossCorrWeight = self.userOpts.ccW
    cyclicWeight = self.userOpts.cycleW
    crossCorrWeight,smoothNessWeight, cyclicWeight = self.normalizeWeights(crossCorrWeight, smoothNessWeight, cyclicWeight)
    
    if (currDefFields is not None) and (idx is not None):
      boundaryLoss = lossCalculator.smoothBoundary(idx, self.userOpts.device)
        
    if imgDataToWork.shape[1] > 3:
      smoothnessLoss =lossCalculator.smoothnessVecFieldT(self.userOpts.device)
    else:
      smoothnessLoss = lossCalculator.smoothnessVecField(self.userOpts.device)
    smoothnessDF = smoothnessLoss + boundaryLoss * self.userOpts.boundarySmoothnessW[itIdx]
    
    
    #zeroDefField = Utils.getZeroDefField(imgDataToWork.shape, self.userOpts.device)
    
    imgDataDef = torch.empty(imgDataToWork.shape, device=self.userOpts.device, requires_grad=False)
    cycleImgData = torch.empty(defFields.shape, device=self.userOpts.device)
    
#     cycleIdxData = zeroDefField.clone()
    
    zeroIndices = torch.from_numpy( np.indices([imgDataToWork.shape[0],3,imgDataToWork.shape[2],imgDataToWork.shape[3],imgDataToWork.shape[4]]) )
    for chanIdx in range(-1, imgDataToWork.shape[1] - 1):
      imgToDef = imgDataToWork[:, None, chanIdx, ]
      chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
      deformedTmp = Utils.deformImage(imgToDef, addedField[: , chanRange, ], self.userOpts.device, False)
      imgDataDef[:, chanIdx + 1, ] = deformedTmp[:, 0, ]
      
      cycleImgData[:,chanRange, ] = defFields[zeroIndices[0],zeroIndices[1],zeroIndices[2], zeroIndices[3], zeroIndices[4]]
      zeroIndices[1] += 3
      
      ##take care of def vec order !!!
#       zeroIndices[2] += defFields[:,0,].detach()
#       zeroIndices[2][zeroIndices[2] > (imgDataToWork.shape[2] - 1)] = imgDataToWork.shape[2] - 1
#       
#       zeroIndices[3] += defFields[:,1,].detach()
#       zeroIndices[3][zeroIndices[3] > (imgDataToWork.shape[3] - 1)] = imgDataToWork.shape[3] - 1
#       
#       zeroIndices[4] += defFields[:,3,].detach()
#       zeroIndices[4][zeroIndices[4] > (imgDataToWork.shape[4] - 1)] = imgDataToWork.shape[4] - 1  
       
#       cycleImgData[:,chanRange, ] = torch.nn.functional.grid_sample(defFields[:,chanRange, ], cycleIdxData, mode='bilinear', padding_mode='border')
                   
#       cycleIdxData[..., 0] = cycleIdxData[..., 0] + defFields[:, chanIdx * 3, ].detach() / ((imgToDef.shape[4]-1) / 2.0)
#       cycleIdxData[..., 1] = cycleIdxData[..., 1] + defFields[:, chanIdx * 3 + 1, ].detach() / ((imgToDef.shape[3]-1) / 2.0)
#       cycleIdxData[..., 2] = cycleIdxData[..., 2] + defFields[:, chanIdx * 3 + 2, ].detach() / ((imgToDef.shape[2]-1) / 2.0)
    
#     del cycleIdxData
    
    crossCorr = lossCalculator.normCrossCorr(imgDataDef)
    cycleLoss = lossCalculator.cycleLoss(cycleImgData, self.userOpts.device)
    
    print('cycleLoss', np.float64(cycleLoss))
    defFields.register_hook(Utils.save_grad('defFields'))
    cycleImgData.register_hook(Utils.save_grad('cycleImgData'))
    
    loss = cycleLoss
    #loss = crossCorrWeight * crossCorr + smoothNessWeight * smoothnessDF + self.userOpts.cycleW * cycleLoss
#     print('cc: %.5f smmothness: %.5f cycleLoss: %.5f' % (crossCorr, smoothnessDF, cycleLoss))
#     print('weighted cc: %.5f smmothness: %.5f cycleLoss: %.5f' % (crossCorrWeight * crossCorr, smoothNessWeight * smoothnessDF, self.userOpts.cycleW * cycleLoss))
      
    loss.backward()
    self.optimizer.step()
    return loss, defFields.detach()        