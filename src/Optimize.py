import torch
import torch.optim as optim

import numpy as np
import copy
from Utils import getDefField, getZeroDefField, smoothArray3D, getPatchSize, deformImage, saveImg, getReceptiveFieldOffset, compareDicts
import SimpleITK as sitk

import LossFunctions as lf
from torch.utils.data import dataloader
from eval.LandmarkHandler import PointProcessor, PointReader

from Sampler import Sampler

import os
import Visualize


class Optimize():

  def __init__(self, net, userOpts):
    self.net = net
    self.userOpts = userOpts
    self.normalizeWeights()
    
    self.net.to(self.userOpts.device)
    self.finalNumberIterations = [0,0]
    
    logfileName = self.userOpts.outputPath + os.path.sep + 'lossLog.csv'
    self.logFile = open(logfileName,'w', buffering=0)
    
  def __enter__(self):
        return self
      
  def __exit__(self, exc_type, exc_value, traceback):
    self.logFile.close()
    
  
  def normalizeWeights(self):
    weightSum = self.userOpts.ccW + self.userOpts.smoothW + self.userOpts.vecLengthW + self.userOpts.cycleW 
    self.userOpts.ccW = self.userOpts.ccW  / weightSum
    self.userOpts.smoothW = self.userOpts.smoothW  / weightSum
    self.userOpts.vecLengthW = self.userOpts.vecLengthW  / weightSum
    self.userOpts.cycleW = self.userOpts.cycleW  / weightSum
    
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
        defX = defFields[imgIdx, chanIdx * 3, ].detach()
        defY = defFields[imgIdx, chanIdx * 3 + 1, ].detach()
        defZ = defFields[imgIdx, chanIdx * 3 + 2, ].detach()
        defField = getDefField(defX, defY, defZ)
        defDataToSave = sitk.GetImageFromArray(defField, isVector=True)
        dataloader.dataset.saveData(defDataToSave, self.userOpts.outputPath, 'deformationFieldDataset' + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
        
        
        if (len(landmarkData) > 0):
          defField = np.moveaxis(defField, 0, 2)
          defField = np.moveaxis(defField, 0, 1)
          defField = torch.from_numpy(defField)
          currLandmarks = landmarkData[chanIdx + 1] ##the def field points from output to input therefore we need no take the next landmarks to be able to deform them
          deformedPoints = pp.deformPointsWithField(currLandmarks, defField, dataloader.dataset.origins[datasetIdx], dataloader.dataset.spacings[datasetIdx])
          pr.saveData(self.userOpts.outputPath + os.path.sep + str(chanIdx+1) + '0deformed.pts', deformedPoints)
            
  def save_grad(self, name):
  
      def hook(grad):
          print(name)
          print(torch.sum(grad))
  
      return hook
  
    
  def printGPUMemoryAllocated(self):
    torch.cuda.synchronize()
    print(torch.cuda.memory_allocated())
  
   
  def optimizeNet(self, imgDataToWork, labelToWork, optimizer, currDefFields = None, idx=None):
    
    # zero the parameter gradients
    optimizer.zero_grad()
        
    defFields = self.net(imgDataToWork)
    
    if (currDefFields is not None) and (idx is not None):
      detachedFiels = defFields.detach()
      padVals = torch.Size([1,1,1,1,1,1])
      detachedFiels = torch.nn.functional.pad(detachedFiels, padVals, "replicate")
      currDefFields = torch.nn.functional.pad(currDefFields, padVals, "constant", 0)
      currDefFields = currDefFields[:, :, idx[0]:idx[0]+defFields.shape[2]+2, idx[1]:idx[1]+defFields.shape[3]+2, idx[2]:idx[2]+defFields.shape[4]+2]
      currDefFields[:, :, idx[0]+1:idx[0]+defFields.shape[2], idx[1]+1:idx[1]+defFields.shape[3], idx[2]+1:idx[2]+defFields.shape[4]] = detachedFiels[:, :, idx[0]+1:idx[0]+defFields.shape[2], idx[1]+1:idx[1]+defFields.shape[3], idx[2]+1:idx[2]+defFields.shape[4]]
      currDefFields[currDefFields==0] = detachedFiels[currDefFields==0]
      if imgDataToWork.shape[1] > 3:
        smoothnessDF = lf.smoothnessVecFieldPatchNeighborsT(defFields, currDefFields, self.userOpts.device)
      else:
        smoothnessDF = lf.smoothnessVecFieldPatchNeighbors(defFields, currDefFields, self.userOpts.device)
    else:
      if imgDataToWork.shape[1] > 3:
        smoothnessDF = lf.smoothnessVecFieldT(defFields, self.userOpts.device)
      else:
        smoothnessDF = lf.smoothnessVecField(defFields, self.userOpts.device)
    
    cropStart0 = (imgDataToWork.shape[2]-defFields.shape[2])/2
    cropStart1 = (imgDataToWork.shape[3]-defFields.shape[3])/2
    cropStart2 = (imgDataToWork.shape[4]-defFields.shape[4])/2
    imgDataToWork = imgDataToWork[:,:,cropStart0:cropStart0+defFields.shape[2], cropStart1:cropStart1+defFields.shape[3], cropStart2:cropStart2+defFields.shape[4]]
    
#     zeroDefField = getZeroDefField(imgDataToWork.shape, self.userOpts.device)
    
    
    imgDataDef = torch.empty(imgDataToWork.shape, device=self.userOpts.device, requires_grad=False)
    
#     cycleImgData = torch.empty(defFields.shape, device=self.userOpts.device)
    
    #         cycleIdxData = torch.empty((imgData.shape[0:2]) + zeroDefField.shape[1:], device=device)
   # cycleIdxData = zeroDefField.clone()
    
    for chanIdx in range(-1, imgDataToWork.shape[1] - 1):
      imgToDef = imgDataToWork[:, None, chanIdx, ]
      chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
      deformedTmp = deformImage(imgToDef, defFields[: , chanRange, ], self.userOpts.device, False)
      imgDataDef[:, chanIdx + 1, ] = deformedTmp[:, 0, ]
      
#       cycleImgData[:, chanRange, ] = torch.nn.functional.grid_sample(defFields[:, chanRange, ], cycleIdxData.clone(), mode='bilinear', padding_mode='border')
#                   
#       cycleIdxData[..., 0] = cycleIdxData[..., 0] + defFields[:, chanIdx * 3, ].detach() / (imgToDef.shape[2] / 2)
#       cycleIdxData[..., 1] = cycleIdxData[..., 1] + defFields[:, chanIdx * 3 + 1, ].detach() / (imgToDef.shape[3] / 2)
#       cycleIdxData[..., 2] = cycleIdxData[..., 2] + defFields[:, chanIdx * 3 + 2, ].detach() / (imgToDef.shape[4] / 2)
#     
#     del zeroDefField, cycleIdxData
          
    crossCorr = lf.normCrossCorr(imgDataToWork, imgDataDef)

#     
#     cycleLoss = lf.cycleLoss(cycleImgData, self.userOpts.device)
#     loss = self.userOpts.ccW * crossCorr + self.userOpts.smoothW * smoothnessDF + self.userOpts.cycleW * cycleLoss
    loss = self.userOpts.ccW * crossCorr + self.userOpts.smoothW * smoothnessDF
#     print('cc: %.5f smmothness: %.5f' % (crossCorr, smoothnessDF))
    #print('cc: %.5f smmothness: %.5f cycleLoss: %.5f' % (crossCorr, smoothnessDF, cycleLoss))
#     print('cc: %.5f smmothnessW: %.5f vecLengthW: %.5f cycleLossW: %.5f' % (self.userOpts.ccW, self.userOpts.smoothW, self.userOpts.vecLengthW, self.userOpts.cycleW))
#     print('loss: %.3f' % (loss))
      
    loss.backward()
    optimizer.step()
    return loss
  
  def terminateLoopByLoss(self, loss, meanLoss, currIteration, itThreshold, iterIdx):
    if (meanLoss - loss < self.userOpts.lossTollerance):
      self.finalLoss = loss
      self.finalNumberIterations[iterIdx] = currIteration
      return True
    else:
      return False
    
  def terminateLoopByLossAndItCount(self, loss, meanLoss, currIteration, itThreshold, iterIdx):
    if (meanLoss - loss < self.userOpts.lossTollerance) or (currIteration >= itThreshold):
      self.finalLoss = loss
      self.finalNumberIterations[iterIdx] = currIteration
      return True
    else:
      return False
    
  def terminateLoopByItCount(self, loss, runningLoss, currIteration, itThreshold, iterIdx):
    if (currIteration >= itThreshold):
      self.finalLoss = loss
      self.finalNumberIterations[iterIdx] = currIteration
      return True
    else:
      return False
  
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
        # get the inputs
        imgData = data['image']
        labelData = data['label']
        maskData = data['mask']
        landmarkData = data['landmarks']
        
        defFields = torch.zeros((imgData.shape[0], imgData.shape[1] * 3, imgData.shape[2], imgData.shape[3], imgData.shape[4]), device=self.userOpts.device, requires_grad=False)
        indexArray = torch.zeros((imgData.shape[2], imgData.shape[3], imgData.shape[4]), device=self.userOpts.device, requires_grad=False)
        
        samplerShift = (0,0,0)
        if not self.userOpts.usePaddedNet:
          padVals = (receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset)
          imgData = torch.nn.functional.pad(imgData, padVals, "constant", 0)
          if (maskData.dim() == imgData.dim()):
            maskData = maskData.float()
            maskData = torch.nn.functional.pad(maskData, padVals, "constant", 0)
            maskData = maskData.byte()
          if (labelData.dim() == imgData.dim()):
            labelData = torch.nn.functional.pad(labelData, padVals, "constant", 0)
          
          samplerShift = (receptiveFieldOffset*2,receptiveFieldOffset*2,receptiveFieldOffset*2)
        
        imgData = imgData.to(self.userOpts.device)
        sampler = Sampler(maskData, imgData, labelData, self.userOpts.patchSize) 
        idxs = sampler.getIndicesForOneShotSampling(samplerShift)
#         idxs = sampler.getIndicesForUniformSampling()
        
        netStateDicts = [None for tmp in idxs]
        optimizerStateDicts = [None for tmp in idxs]
        print('patches: ', idxs)
        imgSetIeration = 0
        oldLoss = 100.0
        while True:
          patchIdx=0
          cumLoss = 0.0
          for idx in idxs:
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
#               loss = self.optimizeNet(imgDataToWork, None, optimizer)
              loss = self.optimizeNet(imgDataToWork, None, optimizer, defFields, idx)
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
              startImgIdx0 = idx[0]
              startImgIdx1 = idx[1]
              startImgIdx2 = idx[2]
              defFields[:, :, startImgIdx0:startImgIdx0+tmpField.shape[2], startImgIdx1:startImgIdx1+tmpField.shape[3], startImgIdx2:startImgIdx2+tmpField.shape[4]] = tmpField
#               defFields[:, :, startImgIdx0:startImgIdx0+tmpField.shape[2], startImgIdx1:startImgIdx1+tmpField.shape[3], startImgIdx2:startImgIdx2+tmpField.shape[4]] += tmpField
              
              indexArray[startImgIdx0:startImgIdx0+tmpField.shape[2], startImgIdx1:startImgIdx1+tmpField.shape[3], startImgIdx2:startImgIdx2+tmpField.shape[4]] = patchIdx
            
            patchIdx+=1
            
          defX = defFields[0, 0, ]
          defY = defFields[0, 1, ]
          defZ = defFields[0, 2, ]
          defField = getDefField(defX, defY, defZ)
          defDataToSave = sitk.GetImageFromArray(defField, isVector=True)
          dataloader.dataset.saveData(defDataToSave, self.userOpts.outputPath, 'tmpCheckDefField' + str(imgSetIeration) + '.nrrd', i, False)
          
          imgSetIeration = imgSetIeration+1 
          print(oldLoss - cumLoss)
          if torch.abs(oldLoss - cumLoss) < self.userOpts.lossTollerance:
            break
          oldLoss = cumLoss
          
        if not self.userOpts.usePaddedNet:
          imgData = imgData[:,:,receptiveFieldOffset:-receptiveFieldOffset,receptiveFieldOffset:-receptiveFieldOffset,receptiveFieldOffset:-receptiveFieldOffset]
        
        saveImg(indexArray, self.userOpts.outputPath + os.path.sep + 'indexArray.nrrd')
        indexArray[indexArray < 1] = 1
        
#         for dim0 in range(0, defFields.shape[0]):
#           for dim1 in range(0, defFields.shape[1]):
#             defFieldsTmp = defFields[dim0, dim1, ] / indexArray
#             defFields[dim0, dim1, ] = defFieldsTmp
  
        del indexArray
        
        self.saveResults(imgData, landmarkData, defFields, dataloader, i)
            
    
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
          
          imgData = imgData.to(self.userOpts.device)
          sampler = Sampler(maskData, imgData, labelData, imgPatchSize)
          
          icc = lf.normCrossCorr(imgData, imgData[:,range(-1,imgData.shape[1]-1),])
          print('inital cross corr: ', icc)
          
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
            loss = self.optimizeNet(imgDataToWork, labelDataToWork, optimizer)
            
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


