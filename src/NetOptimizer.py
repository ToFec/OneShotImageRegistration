
import Utils
import torch
import LossFunctions as lf

class NetOptimizer(object):

  def __init__(self, net, spacing, optimizer,options):
    self.spacing = spacing
    self.optimizer = optimizer
    self.net = net
    self.userOpts = options
        
  def normalizeWeights(self, ccW, sW, cyW):
    weightSum = ccW + sW + cyW
    return [ccW  / weightSum, sW  / weightSum, cyW  / weightSum]
            
  def optimizeNet(self, imgDataToWork, labelToWork, currDefFields = None, idx=None, itIdx=0):
    
    # zero the parameter gradients
    self.optimizer.zero_grad()
        
    defFields = self.net(imgDataToWork)
    
    cropStart0 = (imgDataToWork.shape[2]-defFields.shape[2])/2
    cropStart1 = (imgDataToWork.shape[3]-defFields.shape[3])/2
    cropStart2 = (imgDataToWork.shape[4]-defFields.shape[4])/2
    imgDataToWork = imgDataToWork[:,:,cropStart0:cropStart0+defFields.shape[2], cropStart1:cropStart1+defFields.shape[3], cropStart2:cropStart2+defFields.shape[4]]
    
    lossCalculator = lf.LossFunctions(imgDataToWork, defFields, currDefFields, self.spacing)
    
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
    
    
#     zeroDefField = getZeroDefField(imgDataToWork.shape, self.userOpts.device)
    
    
    imgDataDef = torch.empty(imgDataToWork.shape, device=self.userOpts.device, requires_grad=False)
    
#     cycleImgData = torch.empty(defFields.shape, device=self.userOpts.device)
    
    #         cycleIdxData = torch.empty((imgData.shape[0:2]) + zeroDefField.shape[1:], device=device)
   # cycleIdxData = zeroDefField.clone()
    
    for chanIdx in range(-1, imgDataToWork.shape[1] - 1):
      imgToDef = imgDataToWork[:, None, chanIdx, ]
      chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
      deformedTmp = Utils.deformImage(imgToDef, defFields[: , chanRange, ], self.userOpts.device, False)
      imgDataDef[:, chanIdx + 1, ] = deformedTmp[:, 0, ]
      
#       cycleImgData[:, chanRange, ] = torch.nn.functional.grid_sample(defFields[:, chanRange, ], cycleIdxData.clone(), mode='bilinear', padding_mode='border')
#                   
#       cycleIdxData[..., 0] = cycleIdxData[..., 0] + defFields[:, chanIdx * 3, ].detach() / (imgToDef.shape[2] / 2)
#       cycleIdxData[..., 1] = cycleIdxData[..., 1] + defFields[:, chanIdx * 3 + 1, ].detach() / (imgToDef.shape[3] / 2)
#       cycleIdxData[..., 2] = cycleIdxData[..., 2] + defFields[:, chanIdx * 3 + 2, ].detach() / (imgToDef.shape[4] / 2)
#     
#     del zeroDefField, cycleIdxData
          
    crossCorr = lossCalculator.normCrossCorr(imgDataDef)
#     
#     cycleLoss = lossCalculator.cycleLoss(cycleImgData, self.userOpts.device)
#     loss = self.userOpts.ccW * crossCorr + self.userOpts.smoothW * smoothnessDF + self.userOpts.cycleW * cycleLoss
    loss = crossCorrWeight * crossCorr + smoothNessWeight * smoothnessDF
#     print('cc: %.5f smmothness: %.5f' % (crossCorr, smoothnessDF))
    #print('cc: %.5f smmothness: %.5f cycleLoss: %.5f' % (crossCorr, smoothnessDF, cycleLoss))
#     print('cc: %.5f smmothnessW: %.5f vecLengthW: %.5f cycleLossW: %.5f' % (self.userOpts.ccW, self.userOpts.smoothW, self.userOpts.vecLengthW, self.userOpts.cycleW))
#     print('loss: %.3f' % (loss))
      
    loss.backward()
    self.optimizer.step()
    return loss, defFields.detach()        