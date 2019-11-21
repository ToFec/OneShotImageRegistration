
import torch
import Utils
import ScalingAndSquaring as sas
import GaussSmoothing as gs

class NetOptimizer(object):

  def __init__(self, net, channels, optimizer,options):
    self.optimizer = optimizer
    self.net = net
    self.userOpts = options
    self.scalingSquaring = sas.ScalingAndSquaring(options.sasSteps)
    self.smoother = gs.GaussianSmoothing(channels, options.finalGaussKernelSize, options.finalGaussKernelStd,3,options.device)

  def setOptimizer(self, optimizer):
    self.optimizer = optimizer
    
  def setUserOpts(self, options):
    self.userOpts = options
        
  def normalizeWeights(self, ccW, sW, cyW, dscWeight):
    weightSum = ccW + sW + cyW + dscWeight
    return [ccW  / weightSum, sW  / weightSum, cyW  / weightSum, dscWeight / weightSum]


  def cycleLossCalculationsNearestNeighbor(self, zeroIndices, defFields, currDefFields, idx, borderCrossingArry=None):
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
    
    cycleImgData = currentAndActualField[fields0,fields1,fieldsIdxs2, fieldsIdxs3, fieldsIdxs4]
    
    zeroIndices[1] = zeroIndices[1] + 3
    
    ##take care of def vec order !!!
    tmpField = cycleImgData[:,None,2,].detach()
    zeroIndices[2][:,None,0,] += tmpField
    zeroIndices[2][:,None,1,] += tmpField
    zeroIndices[2][:,None,2,] += tmpField
     
    tmpField = cycleImgData[:,None,1,].detach()
    zeroIndices[3][:,None,0,] += tmpField
    zeroIndices[3][:,None,1,] += tmpField
    zeroIndices[3][:,None,2,] += tmpField
     
    tmpField = cycleImgData[:,None,0,].detach()
    zeroIndices[4][:,None,0,] += tmpField
    zeroIndices[4][:,None,1,] += tmpField
    zeroIndices[4][:,None,2,] += tmpField
    
    return cycleImgData


  def getCycleImageData(self, addedField):
    cycleImgData = Utils.getCycleImgData(addedField.shape, self.userOpts.device)#torch.empty(defFields.shape, device=self.userOpts.device)
    zeroIndices = Utils.getZeroIdxField(addedField.shape, self.userOpts.device)
    outOfBoundsTensor = torch.zeros(zeroIndices[0].shape,dtype=torch.uint8, device=self.userOpts.device)
    if self.userOpts.cycleW > 0.0:
      for chanIdx in range(int(addedField.shape[1] /3.0 ) - 1, -1, -1):
        chanRange = range(chanIdx * 3, chanIdx * 3 + 3)   
        cycleImgData[:,chanRange, ] = self.cycleLossCalculationsNearestNeighbor(zeroIndices, addedField, None, None, outOfBoundsTensor)
#         tmp = self.cycleLossCalculationsNearestNeighbor(zeroIndices, addedField, None, None, outOfBoundsTensor)
#         cycleImageDataList[chanIdx] = tmp
        
    return cycleImgData, outOfBoundsTensor
  
  def getCycleImageDataNew(self, addedField):
    zeroIndices = Utils.getZeroIdxField(addedField.shape, self.userOpts.device)
    outOfBoundsTensor = torch.zeros(zeroIndices[0].shape,dtype=torch.uint8, device=self.userOpts.device)
    cycleImageDataList = [None]*(addedField.shape[1] /3)
    if self.userOpts.cycleW > 0.0:
      for chanIdx in range((addedField.shape[1] /3 ) - 1, -1, -1):
#         chanRange = range(chanIdx * 3, chanIdx * 3 + 3)   
#         cycleImgData[:,chanRange, ] = self.cycleLossCalculationsNearestNeighbor(zeroIndices, addedField, None, None, outOfBoundsTensor)
        tmp = self.cycleLossCalculationsNearestNeighbor(zeroIndices, addedField, None, None, outOfBoundsTensor)
        cycleImageDataList[chanIdx] = tmp
        
    return torch.cat(cycleImageDataList, dim=1), outOfBoundsTensor  
  
  def optimizeNet(self, imgDataToWork, lossCalculator, labelToWork, lastVecField = None, currVecFields = None, idx=None, itIdx=0, printLoss = False):
    # zero the parameter gradients
    self.optimizer.zero_grad()
     
    vecFields = self.net(imgDataToWork)
    if self.userOpts.diffeomorphicRegistration:
      vecFields = self.smoother(vecFields)    

    cropStart0 = int((imgDataToWork.shape[2]-vecFields.shape[2])/2)
    cropStart1 = int((imgDataToWork.shape[3]-vecFields.shape[3])/2)
    cropStart2 = int((imgDataToWork.shape[4]-vecFields.shape[4])/2)
      
    addedField = lastVecField[:, :, idx[0]+cropStart0:idx[0]+cropStart0+vecFields.shape[2],
                               idx[1]+cropStart1:idx[1]+cropStart1+vecFields.shape[3], 
                               idx[2]+cropStart2:idx[2]+cropStart2+vecFields.shape[4]]+ vecFields
      
    currVecFields[:, :, idx[0]+cropStart0:idx[0]+cropStart0+vecFields.shape[2],
                   idx[1]+cropStart1:idx[1]+cropStart1+vecFields.shape[3], 
                   idx[2]+cropStart2:idx[2]+cropStart2+vecFields.shape[4]] = addedField.detach()

    imgDataToWork = imgDataToWork[:,:,cropStart0:cropStart0+vecFields.shape[2], cropStart1:cropStart1+vecFields.shape[3], cropStart2:cropStart2+vecFields.shape[4]]
    
    lossCalculator.update(imgDataToWork, vecFields, currVecFields)
    
    smoothNessWeight = self.userOpts.smoothW[itIdx]
    crossCorrWeight = self.userOpts.ccW
    cyclicWeight = self.userOpts.cycleW
    dscWeight = self.userOpts.dscWeight
    crossCorrWeight,smoothNessWeight, cyclicWeight, dscWeight = self.normalizeWeights(crossCorrWeight, smoothNessWeight, cyclicWeight, dscWeight)
    
    boundaryLoss = torch.tensor(0.0,device=self.userOpts.device)
    if self.userOpts.boundarySmoothnessW[itIdx] > 0.0:
      boundaryLoss = lossCalculator.smoothBoundary(idx, self.userOpts.device, addedField)
    
    smoothnessLoss = torch.tensor(0.0,device=self.userOpts.device)
    if self.userOpts.smoothVF:
      smoothnessLoss = lossCalculator.smoothnessVecField(self.userOpts.device)
       
    smoothnessDF = smoothnessLoss + boundaryLoss * self.userOpts.boundarySmoothnessW[itIdx]

    if self.userOpts.diffeomorphicRegistration:
      deformationField = self.scalingSquaring(addedField)
    else:
      deformationField = addedField
    
    imgDataDef = Utils.deformWholeImage(imgDataToWork, deformationField)
    cycleImgData, outOfBoundsTensor = self.getCycleImageData(deformationField)
   
    crossCorr = lossCalculator.normCrossCorr(imgDataDef, self.userOpts.device)
    cycleLoss = lossCalculator.cycleLoss(cycleImgData,outOfBoundsTensor, self.userOpts.device)
    
    diceLoss = torch.tensor(0.0,device=self.userOpts.device)
    if labelToWork is not None and dscWeight > 0.0:
      labelToWork = labelToWork[:,:,cropStart0:cropStart0+vecFields.shape[2], cropStart1:cropStart1+vecFields.shape[3], cropStart2:cropStart2+vecFields.shape[4]]
      diceLoss = lossCalculator.multiLabelDiceLoss(labelToWork, deformationField, False)

    loss = crossCorrWeight * crossCorr + dscWeight * diceLoss + smoothNessWeight * smoothnessDF + self.userOpts.cycleW * cycleLoss    
    if printLoss:
      print('%.5f; %.5f; %5f; %5f; %.5f' % (loss, crossCorr, smoothnessDF, cycleLoss, diceLoss))
    torch.cuda.empty_cache()
#     print(torch.cuda.memory_allocated() / 1048576.0) 
          
    loss.backward()
    self.optimizer.step()
    return [loss, crossCorr, diceLoss, smoothnessDF, cycleLoss]   
