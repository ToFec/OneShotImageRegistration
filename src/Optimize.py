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
        
        self.saveResults(imgData, landmarkData, defFields, dataloader, i)
        
  def saveResults(self, imgData, landmarkData, defFields, dataloader, datasetIdx):
    pp = PointProcessor()
    pr = PointReader()
    for imgIdx in range(imgData.shape[0]):
      for chanIdx in range(-1, imgData.shape[1] - 1):
        imgToDef = imgData[None, None, imgIdx, chanIdx, ]
        imgToDef = imgToDef.to(self.userOpts.device)
        chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
        deformedTmp = deformImage(imgToDef, defFields[None, imgIdx, chanRange, ], self.userOpts.device)
        
        imgDataDef = sitk.GetImageFromArray(deformedTmp[0, 0, ])
        imgDataOrig = sitk.GetImageFromArray(imgToDef[0,0, ])
        
        dataloader.dataset.saveData(imgDataDef, self.userOpts.outputPath, 'deformedImgDataset' + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
        dataloader.dataset.saveData(imgDataOrig, self.userOpts.outputPath, 'origImgDataset' + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
        #deformation calculated for idx coordinates; transform to world coordinates
        defX = defFields[imgIdx, chanIdx * 3, ].detach() * dataloader.dataset.spacings[datasetIdx][0] * dataloader.dataset.directionCosines[datasetIdx][0]
        defY = defFields[imgIdx, chanIdx * 3 + 1, ].detach() * dataloader.dataset.spacings[datasetIdx][1] * dataloader.dataset.directionCosines[datasetIdx][4]
        defZ = defFields[imgIdx, chanIdx * 3 + 2, ].detach() * dataloader.dataset.spacings[datasetIdx][2] * dataloader.dataset.directionCosines[datasetIdx][8]
        defField = getDefField(defX, defY, defZ)
        defDataToSave = sitk.GetImageFromArray(defField, isVector=True)
        dataloader.dataset.saveData(defDataToSave, self.userOpts.outputPath, 'deformationFieldDataset' + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
        
        
        if (len(landmarkData) > 0):
          defField = np.moveaxis(defField, 0, 2)
          defField = np.moveaxis(defField, 0, 1)
          defField = torch.from_numpy(defField)
          currLandmarks = landmarkData[chanIdx + 1] ##the def field points from output to input therefore we need no take the next landmarks to be able to deform them
          deformedPoints = pp.deformPointsWithField(currLandmarks, defField, dataloader.dataset.origins[datasetIdx], dataloader.dataset.spacings[datasetIdx], dataloader.dataset.directionCosines[datasetIdx])
          pr.saveData(self.userOpts.outputPath + os.path.sep + str(chanIdx+1) + '0deformed.pts', deformedPoints)
            
  def save_grad(self, name):
  
      def hook(grad):
          print(name)
          print(torch.sum(grad))
  
      return hook
  
    
  def printGPUMemoryAllocated(self):
    torch.cuda.synchronize()
    print(torch.cuda.memory_allocated())
  
  def terminateLoopByLoss(self, loss, meanLoss, currIteration, itThreshold, iterIdx, tollerance):
    if (meanLoss - loss < tollerance):
      self.finalLoss = loss
      self.finalNumberIterations[iterIdx] = currIteration
      return True
    else:
      return False
    
  def terminateLoopByLossAndItCount(self, loss, meanLoss, currIteration, itThreshold, iterIdx, tollerance):
    if (meanLoss - loss < tollerance) or (currIteration >= itThreshold):
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
        samplingRates = self.userOpts.downSampleRates
        firstRun = True
        for samplingRateIdx, samplingRate in enumerate(samplingRates):
          imgData, maskData, labelData, landmarkData = sampleImgData(data, samplingRate)
          netOptim = NetOptimizer.NetOptimizer(self.net, dataloader.dataset.spacings[i], optimizer, self.userOpts)
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
        
        self.saveResults(imgData, landmarkData, defFields, dataloader, i)
   
  def trainTestNetDownSamplePatch(self, dataloader):
      optimizer = optim.Adam(self.net.parameters())
      if self.userOpts.trainTillConvergence:
        iterationValidation = self.terminateLoopByLossAndItCount
      else:
        iterationValidation = self.terminateLoopByItCount
      
      numberOfiterations = self.userOpts.numberOfEpochs
      
      receptiveFieldOffset = getReceptiveFieldOffset(self.userOpts.netDepth)
      
      for i, data in enumerate(dataloader, 0):
        imgData = data['image']
        labelData = data['label']
        maskData = data['mask']
        
        netOptim = NetOptimizer.NetOptimizer(self.net, dataloader.dataset.spacings[i], optimizer, self.userOpts)
        
        samplerShift = (0,0,0)
        if not self.userOpts.usePaddedNet:
          padVals = (receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset)
          imgData, maskData, labelData = getPaddedData(imgData, maskData, labelData, padVals)
          samplerShift = (receptiveFieldOffset*2,receptiveFieldOffset*2,receptiveFieldOffset*2)
        
#         imgData = imgData.to(self.userOpts.device)
        sampler = Sampler(maskData, imgData, labelData, self.userOpts.patchSize) 
        idxs = sampler.getIndicesForOneShotSampling(samplerShift)
        print('idxs: ', idxs)
        netStateDicts = [None for _ in idxs]
        optimizerStateDicts = [None for _ in idxs]
        currDefField = None
        samplingRates = self.userOpts.downSampleRates
        
        self.net.train()
        
        firstSamplingRate = samplingRates[0]
        currDefField = torch.zeros((imgData.shape[0], imgData.shape[1] * 3, int(imgData.shape[2]*firstSamplingRate), int(imgData.shape[3]*firstSamplingRate), int(imgData.shape[4]*firstSamplingRate)), device=self.userOpts.device, requires_grad=False)
        firstLT = self.userOpts.lossTollerances[0]
        firstImgData, firstmaskData, firstlabelData, _ = sampleImgData(data, firstSamplingRate)
        firstImgData = firstImgData.to(self.userOpts.device)
        firstsampler = Sampler(firstmaskData, firstImgData, firstlabelData, self.userOpts.patchSize) 
        firstIdxs = firstsampler.getIndicesForOneShotSampling(samplerShift, False)
        print('firstIdxs: ', firstIdxs)
        for patchIdx, idx in enumerate(firstIdxs):
          
          optimizer = optim.Adam(self.net.parameters())
          netOptim = NetOptimizer.NetOptimizer(self.net, dataloader.dataset.spacings[i], optimizer, self.userOpts)
          
          print('register patch %i out of %i patches.' % (patchIdx, len(firstIdxs)))
          imgDataToWork = firstsampler.getSubSampleImg(idx, self.userOpts.normImgPatches)
          imgDataToWork = sampleImg(imgDataToWork, 1.0)
          sampledIdx = idx
          imgDataToWork = imgDataToWork.to(self.userOpts.device)
          patchIteration=0
          lossCounter = 0
          runningLoss = torch.ones(10, device=self.userOpts.device)
          while True:
            loss, tmpField = netOptim.optimizeNet(imgDataToWork, None)
            currDefField[:, :, sampledIdx[0]:sampledIdx[0]+tmpField.shape[2], sampledIdx[1]:sampledIdx[1]+tmpField.shape[3], sampledIdx[2]:sampledIdx[2]+tmpField.shape[4]] = tmpField
            detachLoss = loss.detach()                
            runningLoss[lossCounter] = detachLoss
            if lossCounter == 9:
              meanLoss = runningLoss.mean()
              self.logFile.write(str(float(meanLoss)) + ';' + str(patchIdx))
              self.logFile.write('\n')
              lossCounter = 0
              if (iterationValidation(detachLoss, meanLoss, patchIteration, numberOfiterations, 0, firstLT)):
#                 netStateDicts[patchIdx] = copy.deepcopy(self.net.state_dict())
#                 optimizerStateDicts[patchIdx] = copy.deepcopy(optimizer.state_dict())
                break
            else:
              lossCounter+=1
            patchIteration+=1
            
        with torch.no_grad():
          if firstSamplingRate < 1:
            upSampleRate = samplingRates[1] / firstSamplingRate
            currDefField = currDefField * upSampleRate
            currDefField = sampleImg(currDefField, upSampleRate)
            
            
        del firstImgData, firstmaskData, firstlabelData, firstsampler, firstIdxs
        
        upSampleRate = samplingRates[2] / samplingRates[1]
        tmpField = tmpField * upSampleRate
        tmpField = sampleImg(currDefField, upSampleRate)
        
        
        defX = tmpField[0, 0 * 3, ].detach() * dataloader.dataset.spacings[0][0] * dataloader.dataset.directionCosines[0][0]
        defY = tmpField[0, 0 * 3 + 1, ].detach() * dataloader.dataset.spacings[0][1] * dataloader.dataset.directionCosines[0][4]
        defZ = tmpField[0, 0 * 3 + 2, ].detach() * dataloader.dataset.spacings[0][2] * dataloader.dataset.directionCosines[0][8]
        defField = getDefField(defX, defY, defZ)
        defDataToSave = sitk.GetImageFromArray(defField, isVector=True)
        dataloader.dataset.saveData(defDataToSave, self.userOpts.outputPath, 'defFieldScaling' + str(00) + 'image' + str(00) + 'channel' + str(00) + '.nrrd', 00, False)
        
        for samplingRateIdx, samplingRate in enumerate(samplingRates[1:],1):
          print('sampleRate: ', samplingRate)
          for ltIdx, lossTollerance in enumerate(self.userOpts.lossTollerances):
            print('lossTollerance: ', lossTollerance)
            
            lastDeffield = currDefField.clone()
            for patchIdx, idx in enumerate(idxs):
              
              optimizer = optim.Adam(self.net.parameters())
              netOptim = NetOptimizer.NetOptimizer(self.net, dataloader.dataset.spacings[i], optimizer, self.userOpts)              
              
              print('register patch %i out of %i patches.' % (patchIdx, len(idxs)))
#               if netStateDicts[patchIdx] is not None:
#                 stateDict = netStateDicts[patchIdx]
#                 optimizer.load_state_dict( optimizerStateDicts[patchIdx] )
#                 self.net.load_state_dict(stateDict)
                
              
              imgDataToWork = sampler.getSubSampleImg(idx, self.userOpts.normImgPatches)
              imgDataToWork = sampleImg(imgDataToWork, samplingRate)
              sampledIdx = [int(tmp*samplingRate) for tmp in idx]
              imgDataToWork = imgDataToWork.to(self.userOpts.device)
              
              patchIteration=0
              lossCounter = 0
              runningLoss = torch.ones(10, device=self.userOpts.device)
              while True:
                loss, tmpField = netOptim.optimizeNet(imgDataToWork, None, lastDeffield, sampledIdx, samplingRateIdx)
                currDefField[:, :, sampledIdx[0]:sampledIdx[0]+tmpField.shape[2], sampledIdx[1]:sampledIdx[1]+tmpField.shape[3], sampledIdx[2]:sampledIdx[2]+tmpField.shape[4]] = lastDeffield[:, :, sampledIdx[0]:sampledIdx[0]+tmpField.shape[2], sampledIdx[1]:sampledIdx[1]+tmpField.shape[3], sampledIdx[2]:sampledIdx[2]+tmpField.shape[4]]+ tmpField
                detachLoss = loss.detach()                
                runningLoss[lossCounter] = detachLoss
                if lossCounter == 9:
                  meanLoss = runningLoss.mean()
                  self.logFile.write(str(float(meanLoss)) + ';' + str(patchIdx))
                  self.logFile.write('\n')
                  lossCounter = 0
                  if (iterationValidation(detachLoss, meanLoss, patchIteration, numberOfiterations, 0, lossTollerance)):
#                     netStateDicts[patchIdx] = copy.deepcopy(self.net.state_dict())
#                     optimizerStateDicts[patchIdx] = copy.deepcopy(optimizer.state_dict())
                    break
                else:
                  lossCounter+=1
                patchIteration+=1
            
          with torch.no_grad():
            tmpField = currDefField - lastDeffield
            if samplingRate < 1:
              upSampleRate = samplingRates[samplingRateIdx+1] / samplingRate
              currDefField = currDefField * upSampleRate
              currDefField = sampleImg(currDefField, upSampleRate)
              
              
              tmpField = tmpField * upSampleRate
              tmpField = sampleImg(tmpField, upSampleRate)
              
          
          defX = currDefField[0, 0 * 3, ].detach() * dataloader.dataset.spacings[0][0] * dataloader.dataset.directionCosines[0][0]
          defY = currDefField[0, 0 * 3 + 1, ].detach() * dataloader.dataset.spacings[0][1] * dataloader.dataset.directionCosines[0][4]
          defZ = currDefField[0, 0 * 3 + 2, ].detach() * dataloader.dataset.spacings[0][2] * dataloader.dataset.directionCosines[0][8]
          defField = getDefField(defX, defY, defZ)
          defDataToSave = sitk.GetImageFromArray(defField, isVector=True)
          dataloader.dataset.saveData(defDataToSave, self.userOpts.outputPath, 'defFieldScaling' + str(samplingRateIdx) + 'image' + str(samplingRateIdx) + 'channel' + str(samplingRateIdx) + '.nrrd', 00, False)

          defX = tmpField[0, 0 * 3, ].detach() * dataloader.dataset.spacings[0][0] * dataloader.dataset.directionCosines[0][0]
          defY = tmpField[0, 0 * 3 + 1, ].detach() * dataloader.dataset.spacings[0][1] * dataloader.dataset.directionCosines[0][4]
          defZ = tmpField[0, 0 * 3 + 2, ].detach() * dataloader.dataset.spacings[0][2] * dataloader.dataset.directionCosines[0][8]
          defField = getDefField(defX, defY, defZ)
          defDataToSave = sitk.GetImageFromArray(defField, isVector=True)
          dataloader.dataset.saveData(defDataToSave, self.userOpts.outputPath, 'singleDefField' + str(samplingRateIdx) + 'image' + str(samplingRateIdx) + 'channel' + str(samplingRateIdx) + '.nrrd', 00, False)
                
        if not self.userOpts.usePaddedNet:
          imgData = imgData[:,:,receptiveFieldOffset:-receptiveFieldOffset,receptiveFieldOffset:-receptiveFieldOffset,receptiveFieldOffset:-receptiveFieldOffset]
        
        
        self.saveResults(imgData, data['landmarks'], currDefField, dataloader, i)
                  
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
          netOptim = NetOptimizer.NetOptimizer(self.net, dataloader.dataset.spacings[i], optimizer, self.userOpts)
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


