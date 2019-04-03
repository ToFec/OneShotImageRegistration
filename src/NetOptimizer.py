
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


  def cycleLossCalculationsNearestNeighbor(self, zeroIndices, cycleImgData, defFields, imgDataShape, chanRange):
    
    fieldsIdxs4 = zeroIndices[4].round().long()
    fieldsIdxs4[fieldsIdxs4 > (imgDataShape[4] - 1)] = imgDataShape[4] - 1
    
    fieldsIdxs3 = zeroIndices[3].round().long()
    fieldsIdxs3[fieldsIdxs3 > (imgDataShape[3] - 1)] = imgDataShape[3] - 1
    
    fieldsIdxs2 = zeroIndices[2].round().long()
    fieldsIdxs2[fieldsIdxs2 > (imgDataShape[2] - 1)] = imgDataShape[2] - 1
    
    fields0 = zeroIndices[0]
    fields1 = zeroIndices[1]
    
    cycleImgData[:,chanRange, ] = defFields[fields0,fields1,fieldsIdxs2, fieldsIdxs3, fieldsIdxs4]
    
    zeroIndices[1] += 3
    
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
            

  def cycleLossCalculations(self, zeroIndices, cycleImgData, defFields, imgDataShape, chanRange):
    
    fieldsLow4 = zeroIndices[4].trunc()
    partHigh4 = zeroIndices[4] - fieldsLow4
    partLow4 = 1.0 - partHigh4
    fieldsLow4 = fieldsLow4.long()
    fieldsHigh4 = fieldsLow4 + 1
    fieldsHigh4[fieldsHigh4 > (imgDataShape[4] - 1)] = imgDataShape[4] - 1
    fieldsLow4[fieldsLow4 > (imgDataShape[4] - 1)] = imgDataShape[4] - 1      
    
    fieldsLow3 = zeroIndices[3].trunc()
    partHigh3 = zeroIndices[3] - fieldsLow3
    partLow3 = 1.0 - partHigh3
    fieldsLow3 = fieldsLow3.long()
    fieldsHigh3 = fieldsLow3 + 1
    fieldsHigh3[fieldsHigh3 > (imgDataShape[3] - 1)] = imgDataShape[3] - 1
    fieldsLow3[fieldsLow3 > (imgDataShape[3] - 1)] = imgDataShape[3] - 1  
    
    fieldsLow2 = zeroIndices[2].trunc()
    partHigh2 = zeroIndices[2] - fieldsLow2
    partLow2 = 1.0 - partHigh2
    fieldsLow2 = fieldsLow2.long()
    fieldsHigh2 = fieldsLow2 + 1
    fieldsHigh2[fieldsHigh2 > (imgDataShape[2] - 1)] = imgDataShape[2] - 1
    fieldsLow2[fieldsLow2 > (imgDataShape[2] - 1)] = imgDataShape[2] - 1  
    
    fields0 = zeroIndices[0]
    fields1 = zeroIndices[1]
    
    cycleImgData[:,chanRange, ] = partLow2 * partLow3 * partLow4 * defFields[fields0,fields1,fieldsLow2, fieldsLow3, fieldsLow4] + \
    partLow2 * partLow3 * partHigh4 * defFields[fields0,fields1,fieldsLow2, fieldsLow3, fieldsHigh4] + \
    partLow2 * partHigh3 * partLow4 * defFields[fields0,fields1,fieldsLow2, fieldsHigh3, fieldsLow4] + \
    partHigh2 * partLow3 * partLow4 * defFields[fields0,fields1,fieldsHigh2, fieldsLow3, fieldsLow4] + \
    partHigh2 * partHigh3 * partLow4 * defFields[fields0,fields1,fieldsHigh2, fieldsHigh3, fieldsLow4] + \
    partHigh2 * partLow3 * partHigh4 * defFields[fields0,fields1,fieldsHigh2, fieldsLow3, fieldsHigh4] + \
    partLow2 * partHigh3 * partHigh4 * defFields[fields0,fields1,fieldsLow2, fieldsHigh3, fieldsHigh4] + \
    partHigh2 * partHigh3 * partHigh4 * defFields[fields0,fields1,fieldsHigh2, fieldsHigh3, fieldsHigh4]
    
    zeroIndices[1] += 3
    
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
    
    
    imgDataDef = Utils.getImgDataDef(imgDataToWork.shape, self.userOpts.device)#torch.empty(imgDataToWork.shape, device=self.userOpts.device, requires_grad=False)#
    cycleImgData = Utils.getCycleImgData(defFields.shape, self.userOpts.device)#torch.empty(defFields.shape, device=self.userOpts.device)
    
    zeroIndices = Utils.getZeroIdxField(imgDataToWork.shape, self.userOpts.device)
    
    for chanIdx in range(-1, imgDataToWork.shape[1] - 1):
      imgToDef = imgDataToWork[:, None, chanIdx, ]
      chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
      deformedTmp = Utils.deformImage(imgToDef, addedField[: , chanRange, ], self.userOpts.device, False)
      imgDataDef[:, chanIdx + 1, ] = deformedTmp[:, 0, ]
      
      self.cycleLossCalculationsNearestNeighbor(zeroIndices, cycleImgData, defFields, imgDataToWork.shape, chanRange)
      
    crossCorr = lossCalculator.normCrossCorr(imgDataDef)
    cycleLoss = lossCalculator.cycleLoss(cycleImgData, self.userOpts.device)
    
    loss = crossCorrWeight * crossCorr + smoothNessWeight * smoothnessDF + self.userOpts.cycleW * cycleLoss
#     print('cc: %.5f smmothness: %.5f cycleLoss: %.5f' % (crossCorr, smoothnessDF, cycleLoss))
#     print('weighted cc: %.5f smmothness: %.5f cycleLoss: %.5f' % (crossCorrWeight * crossCorr, smoothNessWeight * smoothnessDF, self.userOpts.cycleW * cycleLoss))
      
    loss.backward()
    self.optimizer.step()
    return loss, defFields.detach()        