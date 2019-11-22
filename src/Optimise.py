'''
Created on Nov 22, 2019

@author: fechter
'''
import numpy as np
import torch
import os

class Optimise(object):
    '''
    classdocs
    '''
    def __init__(self, userOpts):
      self.userOpts = userOpts
      logfileName = self.userOpts.outputPath + os.path.sep + 'lossLog.csv'
      self.logFile = open(logfileName,'w')
      self.logFile.write('PatchIdx;Loss;CrossCorr;DSC;Smmoth;Cycle\n')
      self.logFile.flush()
      self.finalNumberIterations = [0,0]
      self.resultModels = []
      
    def __exit__(self, exc_type, exc_value, traceback):
      self.logFile.close()      
        
    def getDownSampleRates(self):
      samplingRates = np.ones(self.userOpts.downSampleSteps + 1)     
      for samplingRateIdx in range(0,self.userOpts.downSampleSteps):
        samplingRates[samplingRateIdx] = 1.0 / (2**(self.userOpts.downSampleSteps-samplingRateIdx))
      return samplingRates[0:self.userOpts.stoptAtSampleStep]
    
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