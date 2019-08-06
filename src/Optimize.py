import torch
import torch.optim as optim

import numpy as np
import copy
from Utils import getDefField, smoothArray3D, getPatchSize, deformImage, saveImg, getReceptiveFieldOffset, sampleImgData, getPaddedData, sampleImg
import SimpleITK as sitk

import LossFunctions as lf
from torch.utils.data import dataloader
from eval.LandmarkHandler import PointProcessor, PointReader
import NetOptimizer
from Sampler import Sampler
import ScalingAndSquaring as sas

import time
import os


class Optimize():

  def __init__(self, net, userOpts):
    self.net = net
    self.userOpts = userOpts
    
    self.net.to(self.userOpts.device)
    self.finalNumberIterations = [0,0]
    
    logfileName = self.userOpts.outputPath + os.path.sep + 'lossLog.csv'
    self.logFile = open(logfileName,'w')
    self.logFile.write('PatchIdx;Loss;CrossCorr;DSC;Smmoth;Cycle\n')
    self.logFile.flush()
    
  def __enter__(self):
        return self
      
  def __exit__(self, exc_type, exc_value, traceback):
    self.logFile.close()
    
  
    
  def loadNet(self, filepath):
    self.net.load_state_dict(torch.load(filepath))
    
  def saveNet(self, filepath):
    torch.save(self.net.state_dict(), filepath)

        
  def saveResults(self, data, defFields, dataloader, datasetIdx):
    imgData = data['image']
    labelData = data['label']
    landmarkData = data['landmarks']
    pp = PointProcessor()
    pr = PointReader()
    for imgIdx in range(imgData.shape[0]):
      for chanIdx in range(-1, imgData.shape[1] - 1):
        imgToDef = imgData[None, None, imgIdx, chanIdx, ]
        imgToDef = imgToDef.to(self.userOpts.device)
        chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
        deformedTmp = deformImage(imgToDef, defFields[None, imgIdx, chanRange, ], self.userOpts.device)
        
        if (labelData is not None) and (labelData.dim() == imgData.dim()):
          labelToDef = labelData[None, None, imgIdx, chanIdx, ].float()
          labelToDef = labelToDef.to(self.userOpts.device)
          
          deformedLabelTmp = deformImage(labelToDef, defFields[None, imgIdx, chanRange, ], self.userOpts.device, NN=True)
          labelDataDef = sitk.GetImageFromArray(deformedLabelTmp[0, 0, ].cpu())
          labelDataOrig = sitk.GetImageFromArray(labelToDef[0, 0, ].cpu())
          dataloader.dataset.saveData(labelDataDef, self.userOpts.outputPath, 'deformedLabelDataset' + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
          dataloader.dataset.saveData(labelDataOrig, self.userOpts.outputPath, 'origLabelDataset' + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
        
        imgDataDef = sitk.GetImageFromArray(deformedTmp[0, 0, ].cpu())
        imgDataOrig = sitk.GetImageFromArray(imgToDef[0,0, ].cpu())
        
        dataloader.dataset.saveData(imgDataDef, self.userOpts.outputPath, 'deformedImgDataset' + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
        dataloader.dataset.saveData(imgDataOrig, self.userOpts.outputPath, 'origImgDataset' + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
        #deformation calculated for idx coordinates; transform to world coordinates
        dataSetSpacing = dataloader.dataset.getSpacing(datasetIdx)
        dataSetDirCosines = dataloader.dataset.getDirectionCosines(datasetIdx)
        defX = defFields[imgIdx, chanIdx * 3, ].detach() * dataSetSpacing[0] * dataSetDirCosines[0]
        defY = defFields[imgIdx, chanIdx * 3 + 1, ].detach() * dataSetSpacing[1] * dataSetDirCosines[4]
        defZ = defFields[imgIdx, chanIdx * 3 + 2, ].detach() * dataSetSpacing[2] * dataSetDirCosines[8]
        defField = getDefField(defX.cpu(), defY.cpu(), defZ.cpu())
        defDataToSave = sitk.GetImageFromArray(defField, isVector=True)
        dataloader.dataset.saveData(defDataToSave, self.userOpts.outputPath, 'deformationFieldDataset' + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
        
        
        if (len(landmarkData) > 0):
          defField = np.moveaxis(defField, 0, 2)
          defField = np.moveaxis(defField, 0, 1)
          defField = torch.from_numpy(defField)
          currLandmarks = landmarkData[chanIdx + 1] ##the def field points from output to input therefore we need no take the next landmarks to be able to deform them
          
          defFieldOrigin = dataloader.dataset.getOrigin(datasetIdx)
          
          deformedPoints = pp.deformPointsWithField(currLandmarks, defField, defFieldOrigin, dataSetSpacing, dataSetDirCosines)
          pr.saveDataTensor(self.userOpts.outputPath + os.path.sep + 'dataset' + str(datasetIdx) + 'channel' + str(chanIdx+1) + '0deformed.pts', deformedPoints)
            
  def printGPUMemoryAllocated(self):
    torch.cuda.synchronize()
    print(torch.cuda.memory_allocated())
  
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
  
  def updateStateDicts(self, netStateDicts, optimizerStateDicts, patchIdxsOld, patchIdxsNew):
    netStateDictsNew = []
    optimizerStateDictsNew = []
    for patchIdxNew in patchIdxsNew:
      lastMatchinIdx = 0
      for oldIdx, patchIdxOld in enumerate(patchIdxsOld):
        if patchIdxNew[0] <= patchIdxOld[0] and patchIdxNew[1] <= patchIdxOld[1] and patchIdxNew[2] <= patchIdxOld[2]:
          lastMatchinIdx = oldIdx
      netStateDictsNew.append( netStateDicts[lastMatchinIdx] )
      optimizerStateDictsNew.append( optimizerStateDicts[lastMatchinIdx] )
    return (netStateDictsNew, optimizerStateDictsNew) 
      
  
  def getDownSampleRates(self):
    samplingRates = np.ones(self.userOpts.downSampleSteps + 1)     
    for samplingRateIdx in range(0,self.userOpts.downSampleSteps):
      samplingRates[samplingRateIdx] = 1.0 / (2**(self.userOpts.downSampleSteps-samplingRateIdx))
    return samplingRates[0:self.userOpts.stoptAtSampleStep]
      
  def getSubCurrDefFieldIdx(self, currDeffield, idx):
    newIdx = list(idx)
    offset = [0,0,0]
    radius = 1
    for i in range(radius,-1,-1):
      if idx[0] > 0:
        newIdx[0] = idx[0] - 1 - i
        newIdx[3] = idx[3] + 1 + i
        offset[0] = 1 + i
        break
    for i in range(radius,0,-1):
      if newIdx[0] < currDeffield.shape[2] - newIdx[3] - i:
        newIdx[3] = newIdx[3] + i + 1
        break      
    for i in range(radius,-1,-1):
      if idx[1] > 0:
        newIdx[1] = idx[1] - 1 - i
        newIdx[4] = idx[4] + 1 + i
        offset[1] = 1 + i
        break
    for i in range(radius,0,-1):
      if newIdx[1] < currDeffield.shape[3] - newIdx[4] - i:
        newIdx[4] = newIdx[4] + i + 1
        break          
    for i in range(radius,-1,-1):
      if idx[2] > i:
        newIdx[2] = idx[2] - 1 - i
        newIdx[5] = idx[5] + 1 + i
        offset[2] = 1 + i
        break
    for i in range(radius,0,-1):
      if newIdx[2] < currDeffield.shape[4] - newIdx[5] - i:
        newIdx[5] = newIdx[5] + i + 1
        break
    return newIdx, offset
   
  def printParameterInfo(self):
      maxNorm = 0.0
      maxVal = 0.0
      total_norm = 0
      dataMean = []
      for p in self.net.parameters():
        dataMean.append(float(p.data.mean()))
        param_norm = p.grad.data.norm(2.0)
        param_val = p.grad.data.abs().max()
        if param_norm > maxNorm:
          maxNorm = param_norm
        if param_val > maxVal:
          maxVal = param_val
        total_norm += param_norm.item() ** 2.0
        total_norm = total_norm ** (1. / 2.0)
        
      print(total_norm, maxNorm, maxVal)
#       print(dataMean)
         
  def trainTestNetDownSamplePatch(self, dataloader):
      if self.userOpts.trainTillConvergence:
        iterationValidation = self.terminateLoopByLossAndItCount
      else:
        iterationValidation = self.terminateLoopByItCount
      
      numberOfiterations = self.userOpts.numberOfEpochs
      
      receptiveFieldOffset = getReceptiveFieldOffset(self.userOpts.netDepth)
      printLossAndCropGrads = False
      for i, data in enumerate(dataloader, 0):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        self.net.reset_params()
        optimizer = optim.Adam(self.net.parameters(),amsgrad=True)
        
        start = time.time()
        netOptim = NetOptimizer.NetOptimizer(self.net, dataloader.dataset.getSpacingXZFlip(i), data['image'].shape[1]*3, optimizer, self.userOpts)
        
        samplerShift = (0,0,0)
        if not self.userOpts.usePaddedNet:
          padVals = (receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset)
          data['image'], data['mask'], data['label'] = getPaddedData(data['image'], data['mask'], data['label'], padVals)
          samplerShift = (receptiveFieldOffset*2,receptiveFieldOffset*2,receptiveFieldOffset*2)
        
        samplingRates = self.getDownSampleRates()
        
        self.net.train()
        currVectorField = None
        for samplingRateIdx, samplingRate in enumerate(samplingRates):
          print('sampleRate: ', samplingRate)
         
          sampledImgData, sampledMaskData, sampledLabelData, _ = sampleImgData(data, samplingRate)
          sampler = Sampler(sampledMaskData, sampledImgData, sampledLabelData, self.userOpts.patchSize[samplingRateIdx]) 
          idxs = sampler.getIndicesForOneShotSampling(samplerShift, self.userOpts.useMedianForSampling[samplingRateIdx])
          
          print('idxs: ', idxs)
          
          if currVectorField is None:
            currVectorField = torch.zeros((sampledImgData.shape[0], sampledImgData.shape[1] * 3, sampledImgData.shape[2], sampledImgData.shape[3], sampledImgData.shape[4]), device="cpu", requires_grad=False)
          
          for ltIdx , lossTollerance in enumerate(self.userOpts.lossTollerances):
            print('lossTollerance: ', lossTollerance)
            
            lastVectorField = currVectorField.clone()
            for patchIdx, idx in enumerate(idxs):
              print('register patch %i out of %i patches.' % (patchIdx, len(idxs)))

              
              optimizer = optim.Adam(self.net.parameters(),amsgrad=True)
              netOptim.setOptimizer(optimizer)
              
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
              while True:
                [loss, crossCorr, diceLoss, smoothnessDF, cycleLoss] = netOptim.optimizeNet(imgDataToWork, labelDataToWork, lastVectorFieldGPU, currVectorFieldGPU, offset, samplingRateIdx+ltIdx, printLossAndCropGrads)
                if printLossAndCropGrads:
                  self.printParameterInfo()
                detachLoss = loss.detach()                
                runningLoss[lossCounter] = detachLoss
                runningCC[lossCounter] = crossCorr.detach()
                runningDSC[lossCounter] = diceLoss.detach()
                runningSmmoth[lossCounter] = smoothnessDF.detach()
                runningCycle[lossCounter] = cycleLoss.detach()
                if lossCounter == 9:
                  meanLoss = runningLoss.mean()
                  self.logFile.write(str(patchIdx) + ';' + str(float(meanLoss)) + ';' + str(float(runningCC.mean())) + ';' + str(float(runningDSC.mean())) + ';' + str(float(runningSmmoth.mean())) + ';' + str(float(runningCycle.mean())) + ';' + '\n')
                  self.logFile.flush()
                  lossCounter = 0
                  if (iterationValidation(detachLoss, meanLoss, patchIteration, numberOfiterations, 0, lossTollerance)):
                    break
                else:
                  lossCounter+=1
                patchIteration+=1
              currVectorField[:, :, idx[0]:idx[0]+imgDataToWork.shape[2], idx[1]:idx[1]+imgDataToWork.shape[3], idx[2]:idx[2]+imgDataToWork.shape[4]] = currVectorFieldGPU[:,:,offset[0]:offset[0]+imgDataToWork.shape[2],offset[1]:offset[1]+imgDataToWork.shape[3],offset[2]:offset[2]+imgDataToWork.shape[4]].to("cpu")
              
          with torch.no_grad():
            if samplingRate < 1:
              if samplingRateIdx+1 == len(samplingRates):
                nextSamplingRate = 1.0
              else:
                nextSamplingRate = samplingRates[samplingRateIdx+1]
              upSampleRate = nextSamplingRate / samplingRate
              currVectorField = currVectorField * upSampleRate
              currVectorField = sampleImg(currVectorField, upSampleRate)
              
        end = time.time()
        print('Registration of dataset %i took:' % (i), end - start, 'seconds')
        if not self.userOpts.usePaddedNet:
          data['image'] = data['image'][:,:,receptiveFieldOffset:-receptiveFieldOffset,receptiveFieldOffset:-receptiveFieldOffset,receptiveFieldOffset:-receptiveFieldOffset]
        
        currVectorField = currVectorField.to(self.userOpts.device)
        
        if self.userOpts.diffeomorphicRegistration:
          scalingSquaring = sas.ScalingAndSquaring()
          deformationField = scalingSquaring(currVectorField)
        else:
          deformationField = currVectorField
        
        self.saveResults(data, deformationField, dataloader, i)                  


