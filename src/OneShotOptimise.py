'''
Created on Nov 22, 2019

@author: fechter
'''
import NetOptimizer
import torch.optim as optim
import numpy as np
import Utils
import torch
from Sampler import Sampler
import LossFunctions as lf
from Optimise import Optimise

class OneShotOptimise(Optimise):

    def __init__(self, data, net, userOpts):
      Optimise.__init__(self, userOpts)
      self.net = net
      optimizer = optim.Adam(self.net.parameters(),amsgrad=True)
      self.netOptim = NetOptimizer.NetOptimizer(self.net, data['image'].shape[1]*3, optimizer, self.userOpts)
      self.data = data
      modValue = 2**(self.userOpts.netDepth - 1)
      
      if self.userOpts.diffeomorphicRegistration:
        self.patchShift = int(self.userOpts.finalGaussKernelSize/2)
        padVal = (int(np.ceil(self.patchShift / float(modValue))) * modValue)
        self.padVals = (padVal, padVal, padVal, padVal, padVal, padVal)
        self.samplerShift = (self.patchShift*2,self.patchShift*2,self.patchShift*2)
      else:
        self.samplerShift = (0,0,0)
        
      self.samplingRates = self.getDownSampleRates()  
      if self.userOpts.trainTillConvergence:
        self.iterationValidation = self.terminateLoopByLossAndItCount
      else:
        self.iterationValidation = self.terminateLoopByItCount
        
    
    
    def runNoOverlap(self, spacing, samplingRates):
      currVectorField = None
      for samplingRateIdx, samplingRate in enumerate(samplingRates):
        print('sampleRate: ', samplingRate)
        sampledImgData, sampledMaskData, sampledLabelData, _ = Utils.sampleImgData(self.data, samplingRate)
        
        if currVectorField is None:
          currVectorField = torch.zeros((sampledImgData.shape[0], sampledImgData.shape[1] * 3, sampledImgData.shape[2], sampledImgData.shape[3], sampledImgData.shape[4]), device="cpu", requires_grad=False)
        
        sampler = Sampler(sampledMaskData, sampledImgData, sampledLabelData, self.userOpts.patchSize[samplingRateIdx]) 
        idxs = sampler.getIndicesForOneShotSampling(self.samplerShift, self.userOpts.useMedianForSampling[samplingRateIdx])
        print('idxs: ', idxs)
        
        lastVectorField = currVectorField.clone()
        
        if len(self.resultModels) > samplingRateIdx:
            self.net.load_state_dict(self.resultModels[samplingRateIdx]['model_state'])
        
        self.processPatchIdxNoOverlap(currVectorField, lastVectorField, idxs, samplingRateIdx, spacing, sampler)
  
        with torch.no_grad():
          if samplingRate < 1:
            if samplingRateIdx+1 == len(samplingRates):
              nextSamplingRate = 1.0
            else:
              nextSamplingRate = samplingRates[samplingRateIdx+1]
            upSampleRate = nextSamplingRate / samplingRate
            currVectorField = currVectorField * upSampleRate
            currVectorField = Utils.sampleImg(currVectorField, upSampleRate)
      return currVectorField 
    
    def runOverlap(self, spacing, samplingRates):
      currVectorField = None
      for samplingRateIdx, samplingRate in enumerate(samplingRates):
        print('sampleRate: ', samplingRate)
        sampledImgData, sampledMaskData, sampledLabelData, _ = Utils.sampleImgData(self.data, samplingRate)
        
        if currVectorField is None:
          currVectorField = torch.zeros((sampledImgData.shape[0], sampledImgData.shape[1] * 3, sampledImgData.shape[2], sampledImgData.shape[3], sampledImgData.shape[4]), device="cpu", requires_grad=False)
        
        sampledImgData, sampledMaskData, sampledLabelData = Utils.getPaddedData(sampledImgData, sampledMaskData, sampledLabelData, self.padVals)
        currVectorField, _, _ = Utils.getPaddedData(currVectorField, None, None, self.padVals)
        
        sampler = Sampler(sampledMaskData, sampledImgData, sampledLabelData, self.userOpts.patchSize[samplingRateIdx]) 
        idxs = sampler.getIndicesForOneShotSampling(self.samplerShift, self.userOpts.useMedianForSampling[samplingRateIdx])
        
        print('idxs: ', idxs)
        
        lastVectorField = currVectorField.clone()
        currVectorField.fill_(0)
        indexArray = torch.zeros((currVectorField.shape[2], currVectorField.shape[3], currVectorField.shape[4]), requires_grad=False, device="cpu")
  
        self.processPatchIdxOverlap(currVectorField, indexArray, lastVectorField, idxs, samplingRateIdx, spacing, sampler)
  
        with torch.no_grad():
          currVectorField[:,:,indexArray < 1] = lastVectorField[:,:,indexArray < 1]
          indexArray[indexArray < 1] = 1
          currVectorField = currVectorField / indexArray[None,None,...]
          
          indexArray = indexArray[self.padVals[0]:-self.padVals[0],self.padVals[0]:-self.padVals[0],self.padVals[0]:-self.padVals[0]]
          
  #           idxArray = sitk.GetImageFromArray(indexArray.cpu())
  #           dataloader.dataset.saveData(idxArray, self.userOpts.outputPath, 'idxArray.nrrd', 0, False)
          
          del indexArray
          currVectorField = currVectorField[:,:,self.padVals[0]:-self.padVals[0],self.padVals[0]:-self.padVals[0],self.padVals[0]:-self.padVals[0]]
          if samplingRate < 1:
            if samplingRateIdx+1 == len(samplingRates):
              nextSamplingRate = 1.0
            else:
              nextSamplingRate = samplingRates[samplingRateIdx+1]
            upSampleRate = nextSamplingRate / samplingRate
            currVectorField = currVectorField * upSampleRate
            currVectorField = Utils.sampleImg(currVectorField, upSampleRate)
      return currVectorField    
      
    def run(self, spacing):
      samplingRates = self.getDownSampleRates()
      if self.userOpts.diffeomorphicRegistration:
        return self.runOverlap(spacing, samplingRates)
      else:
        return self.runNoOverlap(spacing, samplingRates)

    def processPatchIdxNoOverlap(self, currVectorField, lastVectorField, idxs, samplingRateIdx, spacing, sampler):
      for patchIdx, idx in enumerate(idxs):
        print('register patch %i out of %i patches.' % (patchIdx, len(idxs)))
        optimizer = optim.Adam(self.net.parameters(),amsgrad=True)
        self.netOptim.setOptimizer(optimizer)
        
        imgDataToWork, labelDataToWork = sampler.getSubSample(idx, self.userOpts.normImgPatches)
        imgDataToWork = imgDataToWork.to(self.userOpts.device)
        if labelDataToWork is not None:
          labelDataToWork = labelDataToWork.to(self.userOpts.device)
        
        currDefFieldIdx, offset = self.getSubCurrDefFieldIdx(currVectorField, idx)
        currVectorFieldGPU = currVectorField[:, :, currDefFieldIdx[0]:currDefFieldIdx[0]+currDefFieldIdx[3], currDefFieldIdx[1]:currDefFieldIdx[1]+currDefFieldIdx[4], currDefFieldIdx[2]:currDefFieldIdx[2]+currDefFieldIdx[5]].to(device=self.userOpts.device)
        lastVectorFieldGPU = lastVectorField[:, :, currDefFieldIdx[0]:currDefFieldIdx[0]+currDefFieldIdx[3], currDefFieldIdx[1]:currDefFieldIdx[1]+currDefFieldIdx[4], currDefFieldIdx[2]:currDefFieldIdx[2]+currDefFieldIdx[5]].to(device=self.userOpts.device)
        
        patchIteration=0
        lossCounter = 0
        runningLoss = torch.ones(10, device=self.userOpts.device)
        runningCC = torch.ones(10, device=self.userOpts.device)
        runningDSC = torch.ones(10, device=self.userOpts.device)
        runningSmmoth = torch.ones(10, device=self.userOpts.device)
        runningCycle = torch.ones(10, device=self.userOpts.device)
        
        lossCalculator = lf.LossFunctions(imgDataToWork, spacing)
        
        while True:
          [loss, crossCorr, diceLoss, smoothnessDF, cycleLoss] = self.netOptim.optimizeNet(imgDataToWork, lossCalculator, labelDataToWork, lastVectorFieldGPU, currVectorFieldGPU, offset, samplingRateIdx, False)
          if False:
            self.printParameterInfo()
          detachLoss = loss.detach()                
          runningLoss[lossCounter] = detachLoss
          runningCC[lossCounter] = crossCorr.detach()
          runningDSC[lossCounter] = diceLoss.detach()
          runningSmmoth[lossCounter] = smoothnessDF.detach()
          runningCycle[lossCounter] = cycleLoss.detach()
          if lossCounter == 9:
            meanLoss = runningLoss.mean()
            self.logFile.write(str(float(meanLoss)) + ';' + str(patchIdx) + '\n')
            self.logFile.flush()
            lossCounter = 0
            if (self.iterationValidation(detachLoss, meanLoss, patchIteration, self.userOpts.numberOfEpochs[samplingRateIdx], 0, self.userOpts.lossTollerance)):
              break
          else:
            lossCounter+=1
          patchIteration+=1
        currVectorField[:, :, idx[0]:idx[0]+imgDataToWork.shape[2], idx[1]:idx[1]+imgDataToWork.shape[3], idx[2]:idx[2]+imgDataToWork.shape[4]] = currVectorFieldGPU[:,:,offset[0]:offset[0]+imgDataToWork.shape[2],offset[1]:offset[1]+imgDataToWork.shape[3],offset[2]:offset[2]+imgDataToWork.shape[4]].to("cpu")
            
    def processPatchIdxOverlap(self, currVectorField, indexArray, lastVectorField, idxs, samplingRateIdx, spacing, sampler):
      for patchIdx, idx in enumerate(idxs):
        print('register patch %i out of %i patches.' % (patchIdx, len(idxs)))
        optimizer = optim.Adam(self.net.parameters(),amsgrad=True)
        self.netOptim.setOptimizer(optimizer)
        
        imgDataToWork, labelDataToWork = sampler.getSubSample(idx, self.userOpts.normImgPatches)
        imgDataToWork = imgDataToWork.to(self.userOpts.device)
        if labelDataToWork is not None:
          labelDataToWork = labelDataToWork.to(self.userOpts.device)
        
        currDefFieldIdx, offset = self.getSubCurrDefFieldIdx(currVectorField, idx)
        currVectorFieldGPU = currVectorField[:, :, currDefFieldIdx[0]:currDefFieldIdx[0]+currDefFieldIdx[3], currDefFieldIdx[1]:currDefFieldIdx[1]+currDefFieldIdx[4], currDefFieldIdx[2]:currDefFieldIdx[2]+currDefFieldIdx[5]].to(device=self.userOpts.device)
        idxGPU = indexArray[currDefFieldIdx[0]:currDefFieldIdx[0]+currDefFieldIdx[3], currDefFieldIdx[1]:currDefFieldIdx[1]+currDefFieldIdx[4], currDefFieldIdx[2]:currDefFieldIdx[2]+currDefFieldIdx[5]].to(device=self.userOpts.device)
        idxGPU[idxGPU < 1] = 1
        currVectorFieldGPU = currVectorFieldGPU / idxGPU[None,None,...]
        lastVectorFieldGPU = lastVectorField[:, :, currDefFieldIdx[0]:currDefFieldIdx[0]+currDefFieldIdx[3], currDefFieldIdx[1]:currDefFieldIdx[1]+currDefFieldIdx[4], currDefFieldIdx[2]:currDefFieldIdx[2]+currDefFieldIdx[5]].to(device=self.userOpts.device)
        
        patchIteration=0
        lossCounter = 0
        runningLoss = torch.ones(10, device=self.userOpts.device)
        runningCC = torch.ones(10, device=self.userOpts.device)
        runningDSC = torch.ones(10, device=self.userOpts.device)
        runningSmmoth = torch.ones(10, device=self.userOpts.device)
        runningCycle = torch.ones(10, device=self.userOpts.device)
        
        lossCalculator = lf.LossFunctions(imgDataToWork, spacing)
        
        while True:
          [loss, crossCorr, diceLoss, smoothnessDF, cycleLoss] = self.netOptim.optimizeNet(imgDataToWork, lossCalculator, labelDataToWork, lastVectorFieldGPU, currVectorFieldGPU, offset, samplingRateIdx, False)
          if False:
            self.printParameterInfo()
          detachLoss = loss.detach()                
          runningLoss[lossCounter] = detachLoss
          runningCC[lossCounter] = crossCorr.detach()
          runningDSC[lossCounter] = diceLoss.detach()
          runningSmmoth[lossCounter] = smoothnessDF.detach()
          runningCycle[lossCounter] = cycleLoss.detach()
          if lossCounter == 9:
            meanLoss = runningLoss.mean()
            self.logFile.write(str(float(meanLoss)) + ';' + str(patchIdx) + '\n')
            self.logFile.flush()
            lossCounter = 0
            if (self.iterationValidation(detachLoss, meanLoss, patchIteration, self.userOpts.numberOfEpochs[samplingRateIdx], 0, self.userOpts.lossTollerance)):
              break
          else:
            lossCounter+=1
          patchIteration+=1
        currVectorField[:, :, idx[0]+self.patchShift:idx[0]+imgDataToWork.shape[2]-self.patchShift, 
              idx[1]+self.patchShift:idx[1]+imgDataToWork.shape[3]-self.patchShift, 
              idx[2]+self.patchShift:idx[2]+imgDataToWork.shape[4]-self.patchShift] += currVectorFieldGPU[:,:,offset[0]+self.patchShift:offset[0]+imgDataToWork.shape[2]-self.patchShift,offset[1]+self.patchShift:offset[1]+imgDataToWork.shape[3]-self.patchShift,offset[2]+self.patchShift:offset[2]+imgDataToWork.shape[4]-self.patchShift].to("cpu")
        indexArray[idx[0]+self.patchShift:idx[0]+imgDataToWork.shape[2]-self.patchShift, 
               idx[1]+self.patchShift:idx[1]+imgDataToWork.shape[3]-self.patchShift, 
               idx[2]+self.patchShift:idx[2]+imgDataToWork.shape[4]-self.patchShift] += 1           
        