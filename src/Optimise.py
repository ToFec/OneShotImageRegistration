'''
Created on Nov 22, 2019

@author: fechter
'''
import numpy as np
import torch
import os
from Sampler import Sampler
import Utils

class Optimise(object):
    '''
    classdocs
    '''
    def __init__(self, userOpts):
      self.userOpts = userOpts
      self.net = None
      logfileName = self.userOpts.outputPath + os.path.sep + 'lossLog.csv'
      self.logFile = open(logfileName,'w')
      self.logFile.write('PatchIdx;Loss;CrossCorr;DSC;Smmoth;Cycle\n')
      self.logFile.flush()
      self.finalNumberIterations = [0,0]
      
    def __exit__(self, exc_type, exc_value, traceback):
      self.logFile.close()      

    def __enter__(self):
      return self

        
    def getDeformationField(self, imageData, samplingRate, patchSize, useMedianSampling, samplerShift):
      sampledValidationImgData, sampledValidationMaskData, sampledValidationLabelData, _ = Utils.sampleImgData(imageData, samplingRate)
      validationSampler = Sampler(sampledValidationMaskData, sampledValidationImgData, sampledValidationLabelData, patchSize) 
      idxs = validationSampler.getIndicesForOneShotSampling(samplerShift, useMedianSampling)
      currValidationField = torch.zeros((sampledValidationImgData.shape[0], sampledValidationImgData.shape[1] * 3, sampledValidationImgData.shape[2], sampledValidationImgData.shape[3], sampledValidationImgData.shape[4]), device=self.userOpts.device, requires_grad=False)
      for _ , idx in enumerate(idxs):
        validationImageSample = validationSampler.getSubSampleImg(idx, self.userOpts.normImgPatches)
        validationImageSample = validationImageSample.to(self.userOpts.device)
        defField = self.net(validationImageSample)
        currValidationField[:, :, idx[0]:idx[0]+validationImageSample.shape[2], idx[1]:idx[1]+validationImageSample.shape[3], idx[2]:idx[2]+validationImageSample.shape[4]] = defField
      return currValidationField
        
    def terminateLoopByLoss(self, loss, meanLoss, currIteration, itThreshold, iterIdx, tollerance):
      if (torch.abs(meanLoss - loss) < tollerance):
        self.finalLoss = loss
        self.finalNumberIterations[iterIdx] = currIteration
        return True
      else:
        return False
      
    def terminateLoopByLossAndItCount(self, loss, meanLoss, currIteration, itThreshold, iterIdx, tollerance):
      if (torch.abs(meanLoss - loss) < tollerance) or (currIteration >= itThreshold):
        self.finalLoss = loss
        self.finalNumberIterations[iterIdx] = currIteration
        return True
      else:
        return False
      
    def terminateLoopByItCount(self, loss, runningLoss, currIteration, itThreshold, iterIdx, tollerance):
      if (currIteration >= itThreshold):
        self.finalLoss = loss
        self.finalNumberIterations[iterIdx] = currIteration
        return True
      else:
        return False
      
    def getSubCurrDefFieldIdx(self, currDeffield, idx):
      newIdx = list(idx)
      offset = [0,0,0]
      radius = 1
      for i in range(radius,-1,-1):
        if idx[0] - i > 0:
          newIdx[0] = idx[0] - 1 - i
          newIdx[3] = idx[3] + 1 + i
          offset[0] = 1 + i
          break
      for i in range(radius,0,-1):
        if newIdx[0] < currDeffield.shape[2] - newIdx[3] - i:
          newIdx[3] = newIdx[3] + i + 1
          break      
      for i in range(radius,-1,-1):
        if idx[1] - i > 0:
          newIdx[1] = idx[1] - 1 - i
          newIdx[4] = idx[4] + 1 + i
          offset[1] = 1 + i
          break
      for i in range(radius,0,-1):
        if newIdx[1] < currDeffield.shape[3] - newIdx[4] - i:
          newIdx[4] = newIdx[4] + i + 1
          break          
      for i in range(radius,-1,-1):
        if idx[2] - i > i:
          newIdx[2] = idx[2] - 1 - i
          newIdx[5] = idx[5] + 1 + i
          offset[2] = 1 + i
          break
      for i in range(radius,0,-1):
        if newIdx[2] < currDeffield.shape[4] - newIdx[5] - i:
          newIdx[5] = newIdx[5] + i + 1
          break
      return newIdx, offset      
