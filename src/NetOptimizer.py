
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
    if self.userOpts.ccCalcNN:
      self.cycleLossCalculationMethod = self.cycleLossCalculationsNearestNeighbor
    else:
      self.cycleLossCalculationMethod = self.cycleLossCalculations

  def setOptimizer(self, optimizer):
    self.optimizer = optimizer
    
  def setUserOpts(self, options):
    self.userOpts = options
        
  def normalizeWeights(self, ccW, sW, cyW):
    weightSum = ccW + sW + cyW
    return [ccW  / weightSum, sW  / weightSum, cyW  / weightSum]
  
  def setSpacing(self, spacing):
    self.spacing = spacing

  def cycleLossCalculationsNearestNeighbor(self, zeroIndices, cycleImgData, defFields, chanRange, currDefFields, idx, borderCrossingArry=None):
    fieldsIdxs4 = zeroIndices[4].round().long()
    fieldsIdxs3 = zeroIndices[3].round().long()
    fieldsIdxs2 = zeroIndices[2].round().long()
    
    if currDefFields is not None:
      currentAndActualField = currDefFields.clone()
      currentAndActualField[:, :, idx[0]:idx[0]+defFields.shape[2], idx[1]:idx[1]+defFields.shape[3], idx[2]:idx[2]+defFields.shape[4]] = defFields
      fieldsIdxs4 += idx[2]
      fieldsIdxs3 += idx[1]
      fieldsIdxs2 += idx[0]
    else:
      currentAndActualField = defFields
    
    boolMask = (((fieldsIdxs4 > (currentAndActualField.shape[4] - 1)) | (fieldsIdxs4 < 0)) | ((fieldsIdxs3 > (currentAndActualField.shape[3] - 1)) | (fieldsIdxs3 < 0)) | ((fieldsIdxs2 > (currentAndActualField.shape[2] - 1)) | (fieldsIdxs2 < 0)))
    borderCrossingArry[boolMask] = 1
    
    fieldsIdxs4[fieldsIdxs4 > (currentAndActualField.shape[4] - 1)] = currentAndActualField.shape[4] - 1
    fieldsIdxs4[fieldsIdxs4 < 0] = 0
    
    fieldsIdxs3[fieldsIdxs3 > (currentAndActualField.shape[3] - 1)] = currentAndActualField.shape[3] - 1
    fieldsIdxs3[fieldsIdxs3 < 0] = 0
    
    fieldsIdxs2[fieldsIdxs2 > (currentAndActualField.shape[2] - 1)] = currentAndActualField.shape[2] - 1
    fieldsIdxs2[fieldsIdxs2 < 0] = 0
    
    fields0 = zeroIndices[0]
    fields1 = zeroIndices[1]
    
    cycleImgData[:,chanRange, ] = currentAndActualField[fields0,fields1,fieldsIdxs2, fieldsIdxs3, fieldsIdxs4]
    
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

#
###deprecated
#
  def cycleLossCalculations(self, zeroIndices, cycleImgData, defFields, chanRange, currDefFields, idx):
    
    fieldsLow4 = zeroIndices[4].trunc()
    partHigh4 = zeroIndices[4] - fieldsLow4
    fieldsLow3 = zeroIndices[3].trunc()
    partHigh3 = zeroIndices[3] - fieldsLow3
    fieldsLow2 = zeroIndices[2].trunc()
    partHigh2 = zeroIndices[2] - fieldsLow2
        
    if currDefFields is not None:
      currentAndActualField = currDefFields.clone()
      currentAndActualField[:, :, idx[0]:idx[0]+defFields.shape[2], idx[1]:idx[1]+defFields.shape[3], idx[2]:idx[2]+defFields.shape[4]] = defFields
      fieldsLow4 += idx[2]
      fieldsLow3 += idx[1]
      fieldsLow2 += idx[0]
    else:
      currentAndActualField = defFields
    
    partLow4 = 1.0 - partHigh4
    fieldsLow4 = fieldsLow4.long()
    fieldsLow4[fieldsLow4 < 0] = 0 
    fieldsHigh4 = fieldsLow4 + 1
    fieldsHigh4[fieldsHigh4 > (currentAndActualField.shape[4] - 1)] = currentAndActualField.shape[4] - 1
    fieldsLow4[fieldsLow4 > (currentAndActualField.shape[4] - 1)] = currentAndActualField.shape[4] - 1
         
    partLow3 = 1.0 - partHigh3
    fieldsLow3 = fieldsLow3.long()
    fieldsLow3[fieldsLow3 < 0] = 0
    fieldsHigh3 = fieldsLow3 + 1
    fieldsHigh3[fieldsHigh3 > (currentAndActualField.shape[3] - 1)] = currentAndActualField.shape[3] - 1
    fieldsLow3[fieldsLow3 > (currentAndActualField.shape[3] - 1)] = currentAndActualField.shape[3] - 1  
    
    partLow2 = 1.0 - partHigh2
    fieldsLow2 = fieldsLow2.long()
    fieldsLow2[fieldsLow2 < 0] = 0
    fieldsHigh2 = fieldsLow2 + 1
    fieldsHigh2[fieldsHigh2 > (currentAndActualField.shape[2] - 1)] = currentAndActualField.shape[2] - 1
    fieldsLow2[fieldsLow2 > (currentAndActualField.shape[2] - 1)] = currentAndActualField.shape[2] - 1  
    
    fields0 = zeroIndices[0]
    fields1 = zeroIndices[1]
    
    cycleImgData[:,chanRange, ] = partLow2 * partLow3 * partLow4 * currentAndActualField[fields0,fields1,fieldsLow2, fieldsLow3, fieldsLow4] + \
    partLow2 * partLow3 * partHigh4 * currentAndActualField[fields0,fields1,fieldsLow2, fieldsLow3, fieldsHigh4] + \
    partLow2 * partHigh3 * partLow4 * currentAndActualField[fields0,fields1,fieldsLow2, fieldsHigh3, fieldsLow4] + \
    partHigh2 * partLow3 * partLow4 * currentAndActualField[fields0,fields1,fieldsHigh2, fieldsLow3, fieldsLow4] + \
    partHigh2 * partHigh3 * partLow4 * currentAndActualField[fields0,fields1,fieldsHigh2, fieldsHigh3, fieldsLow4] + \
    partHigh2 * partLow3 * partHigh4 * currentAndActualField[fields0,fields1,fieldsHigh2, fieldsLow3, fieldsHigh4] + \
    partLow2 * partHigh3 * partHigh4 * currentAndActualField[fields0,fields1,fieldsLow2, fieldsHigh3, fieldsHigh4] + \
    partHigh2 * partHigh3 * partHigh4 * currentAndActualField[fields0,fields1,fieldsHigh2, fieldsHigh3, fieldsHigh4]
    
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
            
            
  def optimizeNetTrain(self, imgDataToWork, groundTruthData, itIdx=0):
    self.optimizer.zero_grad()
    netInput = torch.cat([imgDataToWork, groundTruthData], dim=1)
    defFields = self.net(netInput)
    
    loss = self.calculateLoss(imgDataToWork, defFields, itIdx, (0,0,0, defFields.shape[2],defFields.shape[3],defFields.shape[4]))    
    torch.cuda.empty_cache()
    loss.backward()
    self.optimizer.step()
    return loss    
  
  def calculateLoss(self, imgDataToWork, defField, itIdx, idx):
    lossCalculator = lf.LossFunctions(imgDataToWork, defField, None, self.spacing)
    
    smoothnessDF = 0.0
    smoothNessWeight = self.userOpts.smoothW[itIdx]
    crossCorrWeight = self.userOpts.ccW
    cyclicWeight = self.userOpts.cycleW
    crossCorrWeight,smoothNessWeight, cyclicWeight = self.normalizeWeights(crossCorrWeight, smoothNessWeight, cyclicWeight)
    
    if imgDataToWork.shape[1] > 2:
      smoothnessDF =lossCalculator.smoothnessVecFieldT(self.userOpts.device)
    else:
      smoothnessDF = lossCalculator.smoothnessVecField(self.userOpts.device)
     
    imgDataDef = Utils.getImgDataDef(imgDataToWork.shape, self.userOpts.device)
    for chanIdx in range(-1, imgDataToWork.shape[1] - 1):
      imgToDef = imgDataToWork[:, None, chanIdx, ]
      chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
      deformedTmp = Utils.deformImage(imgToDef, defField[: , chanRange, ], self.userOpts.device, False)
      imgDataDef[:, chanIdx + 1, ] = deformedTmp[:, 0, ]

    cycleImgData = Utils.getCycleImgData(defField.shape, self.userOpts.device)#torch.empty(defFields.shape, device=self.userOpts.device)
    zeroIndices = Utils.getZeroIdxField(defField.shape, self.userOpts.device)
    outOfBoundsTensor = torch.zeros(zeroIndices[0].shape,dtype=torch.uint8, device=self.userOpts.device)
    if self.userOpts.cycleW > 0.0:
      for chanIdx in range(imgDataToWork.shape[1] - 1, -1, -1):
        chanRange = range(chanIdx * 3, chanIdx * 3 + 3)   
        self.cycleLossCalculationMethod(zeroIndices, cycleImgData, defField, chanRange, None, idx, outOfBoundsTensor)
   
    crossCorr = lossCalculator.normCrossCorr(imgDataDef, self.userOpts.device)
    cycleLoss = lossCalculator.cycleLoss(cycleImgData,outOfBoundsTensor, self.userOpts.device)       
    loss = crossCorrWeight * crossCorr + smoothNessWeight * smoothnessDF + self.userOpts.cycleW * cycleLoss
    return loss    
            
  def optimizeNetOneShot(self, imgDataToWork, groundTruthData, labelToWork, lastDefField = None, currDefFields = None, idx=None, itIdx=0, printLoss = False):
    # zero the parameter gradients
    self.optimizer.zero_grad()
    
    netInput = torch.cat([imgDataToWork, groundTruthData], dim=1) 
    defFields = self.net(netInput)
    
    tmpZeroField = torch.zeros_like(lastDefField)
    tmpZeroField = tmpZeroField[:, :, idx[0]:idx[0]+defFields.shape[2], idx[1]:idx[1]+defFields.shape[3], idx[2]:idx[2]+defFields.shape[4]]+ defFields
    
    tmpZeroField = Utils.combineDeformationFields(tmpZeroField, lastDefField, True) 
      
    addedField = tmpZeroField[:, :, idx[0]:idx[0]+defFields.shape[2], idx[1]:idx[1]+defFields.shape[3], idx[2]:idx[2]+defFields.shape[4]]
      
    currDefFields[:, :, idx[0]:idx[0]+defFields.shape[2], idx[1]:idx[1]+defFields.shape[3], idx[2]:idx[2]+defFields.shape[4]] = addedField.detach()

    cropStart0 = int((imgDataToWork.shape[2]-defFields.shape[2])/2)
    cropStart1 = int((imgDataToWork.shape[3]-defFields.shape[3])/2)
    cropStart2 = int((imgDataToWork.shape[4]-defFields.shape[4])/2)
    imgDataToWork = imgDataToWork[:,:,cropStart0:cropStart0+defFields.shape[2], cropStart1:cropStart1+defFields.shape[3], cropStart2:cropStart2+defFields.shape[4]]

    lossCalculator = lf.LossFunctions(imgDataToWork, addedField, currDefFields, self.spacing)
     
    boundaryLoss = 0.0
    smoothnessLoss = 0.0
    
    smoothNessWeight = self.userOpts.smoothW[itIdx]
    crossCorrWeight = self.userOpts.ccW
    cyclicWeight = self.userOpts.cycleW
    crossCorrWeight,smoothNessWeight, cyclicWeight = self.normalizeWeights(crossCorrWeight, smoothNessWeight, cyclicWeight)
    
    if self.userOpts.boundarySmoothnessW[itIdx] > 0.0:
      boundaryLoss = lossCalculator.smoothBoundary(idx, self.userOpts.device)
           
    if imgDataToWork.shape[1] > 2:
      smoothnessLoss =lossCalculator.smoothnessVecFieldT(self.userOpts.device)
    else:
      smoothnessLoss = lossCalculator.smoothnessVecField(self.userOpts.device)
       
    smoothnessDF = smoothnessLoss + boundaryLoss * self.userOpts.boundarySmoothnessW[itIdx]
    
    
    imgDataDef = Utils.getImgDataDef(imgDataToWork.shape, self.userOpts.device)#torch.empty(imgDataToWork.shape, device=self.userOpts.device, requires_grad=False)#
    for chanIdx in range(-1, imgDataToWork.shape[1] - 1):
      imgToDef = imgDataToWork[:, None, chanIdx, ]
      chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
      deformedTmp = Utils.deformImage(imgToDef, addedField[: , chanRange, ], self.userOpts.device, False)
      imgDataDef[:, chanIdx + 1, ] = deformedTmp[:, 0, ]

    cycleImgData = Utils.getCycleImgData(defFields.shape, self.userOpts.device)#torch.empty(defFields.shape, device=self.userOpts.device)
    zeroIndices = Utils.getZeroIdxField(defFields.shape, self.userOpts.device)
    outOfBoundsTensor = torch.zeros(zeroIndices[0].shape,dtype=torch.uint8, device=self.userOpts.device)
    if self.userOpts.cycleW > 0.0:
      for chanIdx in range(imgDataToWork.shape[1] - 1, -1, -1):
        chanRange = range(chanIdx * 3, chanIdx * 3 + 3)   
        self.cycleLossCalculationMethod(zeroIndices, cycleImgData, addedField, chanRange, None, idx, outOfBoundsTensor)
   
    crossCorr = lossCalculator.normCrossCorr(imgDataDef, self.userOpts.device)
    cycleLoss = lossCalculator.cycleLoss(cycleImgData,outOfBoundsTensor, self.userOpts.device)
    
#     loss = crossCorrWeight * crossCorr + self.userOpts.cycleW * cycleLoss
    loss = crossCorrWeight * crossCorr + smoothNessWeight * smoothnessDF + self.userOpts.cycleW * cycleLoss    
    if printLoss:
      print('%.5f; %.5f; %5f; %5f; %.5f' % (loss, crossCorr, smoothnessLoss, boundaryLoss, cycleLoss))
#     print('%.5f; %5f; %5f; %.5f' % (crossCorr, smoothnessLoss, boundaryLoss, cycleLoss))
#     print('cc: %.5f smmothness: %.5f cycleLoss: %.5f' % (crossCorr, smoothnessDF, cycleLoss))
#     print('weighted cc: %.5f smmothness: %.5f cycleLoss: %.5f' % (crossCorrWeight * crossCorr, smoothNessWeight * smoothnessDF, self.userOpts.cycleW * cycleLoss))
    torch.cuda.empty_cache()
#     print(torch.cuda.memory_allocated() / 1048576.0) 
          
    loss.backward()
    self.optimizer.step()
    return loss        