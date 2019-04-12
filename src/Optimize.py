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

import time
import os


class Optimize():

  def __init__(self, net, userOpts):
    self.net = net
    self.userOpts = userOpts
    
    self.net.to(self.userOpts.device)
    self.finalNumberIterations = [0,0]
    
    logfileName = self.userOpts.outputPath + os.path.sep + 'lossLog.csv'
    self.logFile = open(logfileName,'w', buffering=0)
    
  def __enter__(self):
        return self
      
  def __exit__(self, exc_type, exc_value, traceback):
    self.logFile.close()
    
  
    
  def loadNet(self, filepath):
    self.net.load_state_dict(torch.load(filepath))
    
  def saveNet(self, filepath):
    torch.save(self.net.state_dict(), filepath)

  def testNet(self, dataloader):
    self.net.eval()
    
    patchSize = self.userOpts.patchSize
    pp = PointProcessor()
    pr = PointReader()
    with torch.no_grad():
      for i, data in enumerate(dataloader, 0):
        imgData = data['image']
        labelData = data['label']
        maskData = data['mask']
        landmarkData = data['landmarks']
        
        imgShape = imgData.shape
        imgData = imgData.to(self.userOpts.device)
        idxs = self.getIndicesForUniformSampling(maskData, imgData, patchSize)
        numberofSamplesPerRun = 1
        patchSizes = getPatchSize(imgShape, patchSize)
        
        defFields = torch.zeros((imgShape[0], imgShape[1] * 3, imgShape[2], imgShape[3], imgShape[4]), device=self.userOpts.device, requires_grad=False)
        indexArray = torch.zeros((imgShape[2], imgShape[3], imgShape[4]), device=self.userOpts.device, requires_grad=False)
        for idx in idxs:
          subSamples = self.getUniformlyDistributedSubsamples(numberofSamplesPerRun, (idx,), 0)
          imgDataToWork = subSamples[0]
          indexArray[idx[0]:idx[0] + patchSizes[0], idx[1]:idx[1] + patchSizes[1], idx[2]:idx[2] + patchSizes[2]] += 1
          tmpField = self.net(imgDataToWork)
          defFields[:, :, idx[0]:idx[0] + patchSizes[0], idx[1]:idx[1] + patchSizes[1], idx[2]:idx[2] + patchSizes[2]] += tmpField

        indexArray[indexArray < 1] = 1
        
        for dim0 in range(0, defFields.shape[0]):
          for dim1 in range(0, defFields.shape[1]):
            defFieldsTmp = defFields[dim0, dim1, ] / indexArray
            defFields[dim0, dim1, ] = smoothArray3D(defFieldsTmp, self.userOpts.device)
  
        del indexArray
        
        self.saveResults(data, defFields, dataloader, i)
        
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
          
          deformedLabelTmp = deformImage(labelToDef, defFields[None, imgIdx, chanRange, ], self.userOpts.device)
          deformedLabelTmp = deformedLabelTmp.round().short()
          labelDataDef = sitk.GetImageFromArray(deformedLabelTmp[0, 0, ])
          labelDataOrig = sitk.GetImageFromArray(labelToDef[0, 0, ])
          dataloader.dataset.saveData(labelDataDef, self.userOpts.outputPath, 'deformedLabelDataset' + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
          dataloader.dataset.saveData(labelDataOrig, self.userOpts.outputPath, 'origLabelDataset' + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
        
        imgDataDef = sitk.GetImageFromArray(deformedTmp[0, 0, ])
        imgDataOrig = sitk.GetImageFromArray(imgToDef[0,0, ])
        
        dataloader.dataset.saveData(imgDataDef, self.userOpts.outputPath, 'deformedImgDataset' + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
        dataloader.dataset.saveData(imgDataOrig, self.userOpts.outputPath, 'origImgDataset' + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
        #deformation calculated for idx coordinates; transform to world coordinates
        dataSetSpacing = dataloader.dataset.getSpacing(datasetIdx)
        dataSetDirCosines = dataloader.dataset.getDirectionCosines(datasetIdx)
        defX = defFields[imgIdx, chanIdx * 3, ].detach() * dataSetSpacing[0] * dataSetDirCosines[0]
        defY = defFields[imgIdx, chanIdx * 3 + 1, ].detach() * dataSetSpacing[1] * dataSetDirCosines[4]
        defZ = defFields[imgIdx, chanIdx * 3 + 2, ].detach() * dataSetSpacing[2] * dataSetDirCosines[8]
        defField = getDefField(defX, defY, defZ)
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
      samplingRates[samplingRateIdx] = 1.0 / (2*(self.userOpts.downSampleSteps-samplingRateIdx))
    return samplingRates[0:self.userOpts.stoptAtSampleStep]
      
       
  def trainTestNet(self, dataloader):
    optimizer = optim.Adam(self.net.parameters())
    if self.userOpts.trainTillConvergence:
      iterationValidation = self.terminateLoopByLossAndItCount
    else:
      iterationValidation = self.terminateLoopByItCount
    
    numberOfiterations = self.userOpts.numberOfEpochs
    
    nuOfLayers = self.userOpts.netDepth
    receptiveFieldOffset = getReceptiveFieldOffset(nuOfLayers)
    
    for i, data in enumerate(dataloader, 0):
        
##SAMPLING CODE
        samplingRates = self.getDownSampleRates()    
        firstRun = True
        for samplingRateIdx, samplingRate in enumerate(samplingRates):
          imgData, maskData, labelData, landmarkData = sampleImgData(data, samplingRate)
          netOptim = NetOptimizer.NetOptimizer(self.net, dataloader.dataset.getSpacingXZFlip(i), optimizer, self.userOpts)
          if firstRun:
            defFields = torch.zeros((imgData.shape[0], imgData.shape[1] * 3, imgData.shape[2], imgData.shape[3], imgData.shape[4]), device=self.userOpts.device, requires_grad=False)
          indexArray = torch.zeros((imgData.shape[2], imgData.shape[3], imgData.shape[4]), device=self.userOpts.device, requires_grad=False)          
          
          
          samplerShift = (0,0,0)
          if not self.userOpts.usePaddedNet:
            padVals = (receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset)
            imgData, maskData, labelData = getPaddedData(imgData, maskData, labelData, padVals)
            samplerShift = (receptiveFieldOffset*2,receptiveFieldOffset*2,receptiveFieldOffset*2)
          
          imgData = imgData.to(self.userOpts.device)
          sampler = Sampler(maskData, imgData, labelData, self.userOpts.patchSize) 
          idxs = sampler.getIndicesForOneShotSampling(samplerShift)
#           idxs = sampler.getIndicesForUniformSamplingPathShift( (self.userOpts.receptiveField[0] + receptiveFieldOffset,self.userOpts.receptiveField[1] + receptiveFieldOffset,self.userOpts.receptiveField[2] + receptiveFieldOffset), self.userOpts.useMedianForSampling[samplingRateIdx] )
            

          if firstRun:  
            netStateDicts = [None for _ in idxs]
            optimizerStateDicts = [None for _ in idxs]
          else:
            netStateDicts, optimizerStateDicts = self.updateStateDicts(netStateDicts, optimizerStateDicts, oldIdxs, idxs)
          oldIdxs = idxs
          firstRun = False

          

          print('patches: ', idxs)
          oldLoss = 100.0
          while True:
            cumLoss = 0.0
#             if overlappingPatches:
#               defFields = torch.zeros((imgData.shape[0], imgData.shape[1] * 3, imgData.shape[2], imgData.shape[3], imgData.shape[4]), device=self.userOpts.device, requires_grad=False)
            indexArray = torch.zeros((imgData.shape[2], imgData.shape[3], imgData.shape[4]), device=self.userOpts.device, requires_grad=False)
            for patchIdx, idx in enumerate(idxs):
              print('register patch %i out of %i patches.' % (patchIdx, len(idxs)))
              if netStateDicts[patchIdx] is not None:
                stateDict = netStateDicts[patchIdx]
                optimizer.load_state_dict( optimizerStateDicts[patchIdx] )
                self.net.load_state_dict(stateDict)
                
              self.net.train()
              imgDataToWork = sampler.getSubSampleImg(idx, self.userOpts.normImgPatches)
              patchIteration=0
              lossCounter = 0
              runningLoss = torch.ones(10, device=self.userOpts.device)
              while True:
#                 loss = netOptim.optimizeNet(imgDataToWork, None)
                loss = netOptim.optimizeNet(imgDataToWork, None, defFields, idx)
                detachLoss = loss.detach()
     
                runningLoss[lossCounter] = detachLoss
                if lossCounter == 9:
                  meanLoss = runningLoss.mean()
                  self.logFile.write(str(float(meanLoss)) + ';' + str(patchIdx))
                  self.logFile.write('\n')
                  lossCounter = 0
                  if (iterationValidation(detachLoss, meanLoss, patchIteration, numberOfiterations, 0)):
                    netStateDicts[patchIdx] = copy.deepcopy(self.net.state_dict())
                    optimizerStateDicts[patchIdx] = copy.deepcopy(optimizer.state_dict())
                    cumLoss += meanLoss
                    break
                  
                else:
                  lossCounter+=1
                  
                patchIteration+=1
              
              with torch.no_grad():
                self.net.eval()
                tmpField = self.net(imgDataToWork)
                defFields[:, :, idx[0]:idx[0]+tmpField.shape[2], idx[1]:idx[1]+tmpField.shape[3], idx[2]:idx[2]+tmpField.shape[4]] = tmpField
#                 defFields[:, :, idx[0]:idx[0]+tmpField.shape[2], idx[1]:idx[1]+tmpField.shape[3], idx[2]:idx[2]+tmpField.shape[4]] += tmpField
                
                indexArray[idx[0]:idx[0]+tmpField.shape[2], idx[1]:idx[1]+tmpField.shape[3], idx[2]:idx[2]+tmpField.shape[4]] += 1#patchIdx
          
            indexArray[indexArray < 1] = 1
            for dim0 in range(0, defFields.shape[0]):
              for dim1 in range(0, defFields.shape[1]):
                defFieldsTmp = defFields[dim0, dim1, ] / indexArray
                defFields[dim0, dim1, ] = defFieldsTmp

            if torch.abs(oldLoss - cumLoss) < self.userOpts.cumulativeLossTollerance:
              break
            oldLoss = cumLoss          
              
          if samplingRate < 1:
            upSampleRate = samplingRates[samplingRateIdx+1] / samplingRate
            defFields = torch.nn.functional.interpolate(defFields,scale_factor=upSampleRate,mode='trilinear')
            
        if not self.userOpts.usePaddedNet:
          imgData = imgData[:,:,receptiveFieldOffset:-receptiveFieldOffset,receptiveFieldOffset:-receptiveFieldOffset,receptiveFieldOffset:-receptiveFieldOffset]
        
        saveImg(indexArray, self.userOpts.outputPath + os.path.sep + 'indexArray.nrrd')

  
        del indexArray
        
        self.saveResults({'image': imgData,'label': None,'mask': None,'landmarks': landmarkData}, defFields, dataloader, i)
   
  def trainTestNetDownSamplePatch(self, dataloader):
      if self.userOpts.trainTillConvergence:
        iterationValidation = self.terminateLoopByLossAndItCount
      else:
        iterationValidation = self.terminateLoopByItCount
      
      numberOfiterations = self.userOpts.numberOfEpochs
      
      receptiveFieldOffset = getReceptiveFieldOffset(self.userOpts.netDepth)
      
      for i, data in enumerate(dataloader, 0):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        self.net.reset_params()
        optimizer = optim.Adam(self.net.parameters())
        
        start = time.time()
        netOptim = NetOptimizer.NetOptimizer(self.net, dataloader.dataset.getSpacingXZFlip(i), optimizer, self.userOpts)
        
        samplerShift = (0,0,0)
        if not self.userOpts.usePaddedNet:
          padVals = (receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset)
          data['image'], data['mask'], data['label'] = getPaddedData(data['image'], data['mask'], data['label'], padVals)
          samplerShift = (receptiveFieldOffset*2,receptiveFieldOffset*2,receptiveFieldOffset*2)
        
        samplingRates = self.getDownSampleRates()
        
        self.net.train()
        currDefField = None
        for samplingRateIdx, samplingRate in enumerate(samplingRates):
          print('sampleRate: ', samplingRate)
        
          sampledImgData, sampledMaskData, sampledLabelData, _ = sampleImgData(data, samplingRate)
          sampledImgData = sampledImgData.to(self.userOpts.device)
          sampler = Sampler(sampledMaskData, sampledImgData, sampledLabelData, self.userOpts.patchSize) 
          idxs = sampler.getIndicesForOneShotSampling(samplerShift, self.userOpts.useMedianForSampling[samplingRateIdx])
          
          print('idxs: ', idxs)

          if currDefField is None:
            currDefField = torch.zeros((sampledImgData.shape[0], sampledImgData.shape[1] * 3, sampledImgData.shape[2], sampledImgData.shape[3], sampledImgData.shape[4]), device=self.userOpts.device, requires_grad=False)
          
          for ltIdx , lossTollerance in enumerate(self.userOpts.lossTollerances):
            print('lossTollerance: ', lossTollerance)
          
            lastDeffield = currDefField.clone()
            for patchIdx, idx in enumerate(idxs):
              print('register patch %i out of %i patches.' % (patchIdx, len(idxs)))
              
              imgDataToWork = sampler.getSubSampleImg(idx, self.userOpts.normImgPatches)
              imgDataToWork = imgDataToWork.to(self.userOpts.device)
              
              patchIteration=0
              lossCounter = 0
              runningLoss = torch.ones(10, device=self.userOpts.device)
              while True:
                loss = netOptim.optimizeNet(imgDataToWork, None, lastDeffield, currDefField, idx, samplingRateIdx+ltIdx)
                detachLoss = loss.detach()                
                runningLoss[lossCounter] = detachLoss
                if lossCounter == 9:
                  meanLoss = runningLoss.mean()
                  self.logFile.write(str(float(meanLoss)) + ';' + str(patchIdx) + '\n')
                  lossCounter = 0
                  if (iterationValidation(detachLoss, meanLoss, patchIteration, numberOfiterations, 0, lossTollerance)):
                    break
                else:
                  lossCounter+=1
                patchIteration+=1
            
          with torch.no_grad():
            if samplingRate < 1:
              if samplingRateIdx+1 == len(samplingRates):
                nextSamplingRate = 1.0
              else:
                nextSamplingRate = samplingRates[samplingRateIdx+1]
              upSampleRate = nextSamplingRate / samplingRate
              currDefField = currDefField * upSampleRate
              currDefField = sampleImg(currDefField, upSampleRate)
              
        end = time.time()
        print('Registration of dataset %i took:' % (i), end - start, 'seconds')
        if not self.userOpts.usePaddedNet:
          data['image'] = data['image'][:,:,receptiveFieldOffset:-receptiveFieldOffset,receptiveFieldOffset:-receptiveFieldOffset,receptiveFieldOffset:-receptiveFieldOffset]
        
        self.saveResults(data, currDefField, dataloader, i)
                  
  def trainNet(self, dataloader):
    self.net.train()
    optimizer = optim.Adam(self.net.parameters())
    imgPatchSize = self.userOpts.patchSize
    
    tmpEpochs = 1
    epochs = self.userOpts.numberOfEpochs
    if self.userOpts.oneShot:
      tmpEpochs = epochs
      epochs = 1
      epochValidation = self.terminateLoopByItCount
      if self.userOpts.trainTillConvergence:
        datasetIterationValidation = self.terminateLoopByLossAndItCount
      else:
        datasetIterationValidation = self.terminateLoopByItCount
    else:
      if self.userOpts.trainTillConvergence:
        epochValidation =  self.terminateLoopByLossAndItCount
      else:
        epochValidation =  self.terminateLoopByItCount
      datasetIterationValidation = self.terminateLoopByItCount
      
    lossCounter = 0
    runningLoss = torch.ones(10,device=self.userOpts.device)  
    print('epochs: ', epochs)
    epochCount = 0
    while True: ##epoch loop
      for i, data in enumerate(dataloader, 0):
          # get the inputs
          imgData = data['image']
          labelData = data['label']
          maskData = data['mask']
          netOptim = NetOptimizer.NetOptimizer(self.net, dataloader.dataset.getSpacingXZFlip(i), optimizer, self.userOpts)
          imgData = imgData.to(self.userOpts.device)
          sampler = Sampler(maskData, imgData, labelData, imgPatchSize)
          
          maxNumberOfPixs = imgPatchSize * imgPatchSize * imgPatchSize * imgData.shape[1] + 1
          
          numberofSamples = (torch.numel(imgData) / (maxNumberOfPixs/8)) + 1 #we divide by 8 as we need overlapping samples; e.g. when using uniform sampling we shift the sample window by half of the patch size
          numberOfiterations = (numberofSamples / self.userOpts.maxNumberOfSamples) + 1
          numberOfiterations *= tmpEpochs
          numberofSamplesPerRun = min(numberofSamples, self.userOpts.maxNumberOfSamples - 1)
          
          print('numberOfiterationsPerEpoch: ', numberOfiterations)
          print('numberofSamplesPerIteration: ', numberofSamplesPerRun)
          if torch.numel(imgData) >= maxNumberOfPixs:
            doSubSampling = True
            if self.userOpts.oneShot:
              idxs = sampler.getIndicesForUniformSampling()
              numberofSamplesPerRun = min(len(idxs), numberofSamplesPerRun)
              subSampleMethod = sampler.getUniformlyDistributedSubsamples
            else:
              idxs = sampler.getIndicesForRandomization()
              subSampleMethod = sampler.getRandomSubSamples
          else:
            doSubSampling = False
            imgDataToWork = imgData
            labelDataToWork = labelData
          
          imgIteration = 0
          while True:
            if (doSubSampling):
              subSamples = subSampleMethod(numberofSamplesPerRun, idxs, imgIteration)
              imgDataToWork = subSamples[0]
              labelDataToWork = subSamples[1]
            loss = netOptim.optimizeNet(imgDataToWork, labelDataToWork)
            
            detachLoss = loss.detach()
            runningLoss[lossCounter] = detachLoss
            if lossCounter == 9:
              meanLoss = runningLoss.mean()
              if (datasetIterationValidation(detachLoss, meanLoss, imgIteration, numberOfiterations, 0)):
                break
              self.logFile.write(str(float(meanLoss)) + ';')
              lossCounter = 0
            else:
              lossCounter+=1
            
            imgIteration+=1
            
      epochCount+=1
      if (epochValidation(numpyLoss, runningLoss, epochCount, epochs, 1)):
        break
      
      
    return loss    


