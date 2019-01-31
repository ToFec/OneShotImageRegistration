import torch
import torch.optim as optim

import numpy as np
from Utils import getDefField, getZeroDefField, smoothArray3D, getMaxIdxs, getPatchSize, deformImage
import SimpleITK as sitk

import LossFunctions as lf
from torch.utils.data import dataloader
from eval.LandmarkHandler import PointProcessor, PointReader

import os
import Visualize


class Optimize():

  def __init__(self, net, userOpts):
    self.net = net
    self.userOpts = userOpts
    self.normalizeWeights()
    
    self.net.to(self.userOpts.device)
    self.finalNumberIterations = [0,0]

  
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
        
        idxs = self.getIndicesForUniformSampling(maskData, imgData, patchSize)
        numberofSamplesPerRun = 1
        patchSizes = getPatchSize(imgShape, patchSize)
        
        defFields = torch.zeros((imgShape[0], imgShape[1] * 3, imgShape[2], imgShape[3], imgShape[4]), device=self.userOpts.device, requires_grad=False)
        indexArray = torch.zeros((imgShape[2], imgShape[3], imgShape[4]), device=self.userOpts.device, requires_grad=False)
        for idx in idxs:
          subSamples = self.getUniformlyDistributedSubsamples(numberofSamplesPerRun, (idx,), patchSizes, imgData, labelData, 0)
          imgDataToWork = subSamples[0]
          imgDataToWork = imgDataToWork.to(self.userOpts.device)
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
            
  def getIndicesForRandomization(self, maskData, imgData, imgPatchSize):
    maxIdxs = getMaxIdxs(imgData.shape, imgPatchSize)
    if (maskData.dim() == imgData.dim()):
      maskDataCrop = maskData[:, :, 0:maxIdxs[0], 0:maxIdxs[1], 0:maxIdxs[2]]
    else: 
      maskDataCrop = torch.ones((imgData.shape[0], imgData.shape[1], maxIdxs[0], maxIdxs[1], maxIdxs[2]), dtype=torch.int8)
    
    maskChanSum = torch.sum(maskDataCrop, 1)
    idxs = np.where(maskChanSum > 0)
  
    return idxs
  
  def getIndicesForUniformSampling(self, maskData, imgData, imgPatchSize):
    imgShape = imgData.shape
    if (maskData.dim() != imgData.dim()):
      maskData = torch.ones(imgShape, dtype=torch.int8)

    maxIdxs = getMaxIdxs(imgShape, imgPatchSize)
    patchSizes = getPatchSize(imgData.shape, imgPatchSize)
    maskChanSum = torch.sum(maskData, 1)
    idxs = []
    for patchIdx0 in range(0, maxIdxs[0], patchSizes[0]/2):
      for patchIdx1 in range(0, maxIdxs[1], patchSizes[1]/2):
        for patchIdx2 in range(0, maxIdxs[2], patchSizes[2]/2):
          if (maskChanSum[:,patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[1], patchIdx2:patchIdx2 + patchSizes[2]].median() > 0):
            idxs.append( (patchIdx0, patchIdx1, patchIdx2) )
       
    leftover0 = imgShape[2] % patchSizes[0]
    startidx0 = imgShape[2] - patchSizes[0] if (leftover0 > 0) & (maxIdxs[0] > patchSizes[0])  else 0
    leftover1 = imgShape[3] % patchSizes[1]
    startidx1 = imgShape[3] - patchSizes[1] if (leftover1 > 0) & (maxIdxs[1] > patchSizes[1])  else 0
    leftover2 = imgShape[4] % patchSizes[2]
    startidx2 = imgShape[4] - patchSizes[2] if (leftover2 > 0) & (maxIdxs[2] > patchSizes[2])  else 0
    
    if (startidx2 + startidx1 + startidx0 > 0) :               
      for patchIdx0 in range(startidx0, maxIdxs[0], patchSizes[0]/2):
        for patchIdx1 in range(startidx1, maxIdxs[1], patchSizes[1]/2):
          for patchIdx2 in range(startidx2, maxIdxs[2], patchSizes[2]/2):
            if (maskChanSum[:,patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[1], patchIdx2:patchIdx2 + patchSizes[2]].median() > 0):
              idxs.append( (patchIdx0, patchIdx1, patchIdx2) )
              
    return idxs
      
  def save_grad(self, name):
  
      def hook(grad):
          print(name)
          print(torch.sum(grad))
  
      return hook
  
    
  def printGPUMemoryAllocated(self):
    torch.cuda.synchronize()
    print(torch.cuda.memory_allocated())
  
  def getUniformlyDistributedSubsamples(self,numberofSamplesPerRun, idxs, patchSizes, imgData, labelData, currIteration):
    
    startIdx = currIteration % numberofSamplesPerRun
    
    
    imgDataNew = torch.empty((numberofSamplesPerRun, imgData.shape[1], patchSizes[0], patchSizes[1], patchSizes[2]), requires_grad=False)
    if (labelData.dim() == imgData.dim()):
      labelDataNew = torch.empty((numberofSamplesPerRun, imgData.shape[1], patchSizes[0], patchSizes[1], patchSizes[2]), requires_grad=False)
    
    iterationRange = np.arange(startIdx,startIdx+numberofSamplesPerRun)
    iterationRange[iterationRange >= len(idxs)] = iterationRange[iterationRange >= len(idxs)] - len(idxs)
    j = 0
    for i in iterationRange:
      idx = idxs[i]
      imgDataNew[j, ] = imgData[:, :, idx[0]:idx[0] + patchSizes[0], idx[1]:idx[1] + patchSizes[1], idx[2]:idx[2] + patchSizes[2]]
      if (labelData.dim() == imgData.dim()):
        labelDataNew[j, ] = labelData[:, :, idx[0]:idx[0] + patchSizes[0], idx[1]:idx[1] + patchSizes[1], idx[2]:idx[2] + patchSizes[2]]
      j=j+1
      
    imgDataToWork = imgDataNew
    if (labelData.dim() == imgData.dim()):
      labelDataToWork = labelDataNew
    else:
      labelDataToWork = torch.Tensor();
      
    return (imgDataToWork, labelDataToWork)
    
    
  def getRandomSubSamples(self, numberofSamplesPerRun, idxs, patchSizes, imgData, labelData, currIteration=0):
   
    imgDataNew = torch.empty((numberofSamplesPerRun, imgData.shape[1], patchSizes[0], patchSizes[1], patchSizes[2]), requires_grad=False)
    if (labelData.dim() == imgData.dim()):
      labelDataNew = torch.empty((numberofSamplesPerRun, imgData.shape[1], patchSizes[0], patchSizes[1], patchSizes[2]), requires_grad=False)
    
    randSampleIdxs = np.random.randint(0, len(idxs[0]), (numberofSamplesPerRun,))
    for j in range(0, numberofSamplesPerRun):
      idx0 = idxs[0][randSampleIdxs[j]]
      idx2 = idxs[1][randSampleIdxs[j]]
      idx3 = idxs[2][randSampleIdxs[j]]
      idx4 = idxs[3][randSampleIdxs[j]]
      imgDataNew[j, ] = imgData[idx0, : , idx2:idx2 + patchSizes[0], idx3:idx3 + patchSizes[1], idx4:idx4 + patchSizes[2]]
  #     indexArrayTest[idx2:idx2 + patchSizes[0], idx3:idx3 + patchSizes[1], idx4:idx4 + patchSizes[2]] += 1
      if (labelData.dim() == imgData.dim()):
        labelDataNew[j, ] = labelData[idx0, : , idx2:idx2 + patchSizes[0], idx3:idx3 + patchSizes[1], idx4:idx4 + patchSizes[2]]
    
    imgDataToWork = imgDataNew
    if (labelData.dim() == imgData.dim()):
      labelDataToWork = labelDataNew
    else:
      labelDataToWork = torch.Tensor();
      
    return (imgDataToWork, labelDataToWork)
  
  def optimizeNet(self, imgDataToWork, labelToWork, optimizer):
    zeroDefField = getZeroDefField(imgDataToWork.shape)
    imgDataToWork = imgDataToWork.to(self.userOpts.device)
    zeroDefField = zeroDefField.to(self.userOpts.device)
    
    # zero the parameter gradients
    optimizer.zero_grad()
    
    # forward + backward + optimize
    defFields = self.net(imgDataToWork)
    
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
#     if imgDataToWork.shape[1] > 3:
#       smoothnessDF = lf.smoothnessVecFieldT(defFields, self.userOpts.device)
#     else:
#       smoothnessDF = lf.smoothnessVecField(defFields, self.userOpts.device)
#     
#     cycleLoss = lf.cycleLoss(cycleImgData, self.userOpts.device)
#     loss = self.userOpts.ccW * crossCorr + self.userOpts.smoothW * smoothnessDF + self.userOpts.cycleW * cycleLoss
    loss = crossCorr
    print('cc: %.5f ' % (crossCorr))
    #print('cc: %.5f smmothness: %.5f cycleLoss: %.5f' % (crossCorr, smoothnessDF, cycleLoss))
#     print('cc: %.5f smmothnessW: %.5f vecLengthW: %.5f cycleLossW: %.5f' % (self.userOpts.ccW, self.userOpts.smoothW, self.userOpts.vecLengthW, self.userOpts.cycleW))
#     print('loss: %.3f' % (loss))
      
    loss.backward()
    optimizer.step()
    return loss
  
  def terminateLoopByLoss(self, loss, runningLoss, currIteration, itThreshold, iterIdx):
    if (loss == runningLoss.mean()):
      self.finalLoss = loss
      self.finalNumberIterations[iterIdx] = currIteration
      return True
    else:
      return False
    
  def terminateLoopByLossAndItCount(self, loss, runningLoss, currIteration, itThreshold, iterIdx):
    if (np.isclose(loss, runningLoss.mean(),atol=self.userOpts.lossTollerance) or currIteration == itThreshold):
      self.finalLoss = loss
      self.finalNumberIterations[iterIdx] = currIteration
      return True
    else:
      return False
    
  def terminateLoopByItCount(self, loss, runningLoss, currIteration, itThreshold, iterIdx):
    if (currIteration == itThreshold):
      self.finalLoss = loss
      self.finalNumberIterations[iterIdx] = currIteration
      return True
    else:
      return False
  
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
      
    logfile = self.userOpts.outputPath + os.path.sep + 'lossLog.csv'    
    logFile = open(logfile,'w', buffering=0)  
    lossCounter = 0
    runningLoss = np.ones(10)  
    print('epochs: ', epochs)
    epochCount = 0
    while True: ##epoch loop
      for i, data in enumerate(dataloader, 0):
          # get the inputs
          imgData = data['image']
          labelData = data['label']
          maskData = data['mask']
          
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
            patchSizes = getPatchSize(imgData.shape, imgPatchSize)
            doSubSampling = True
            if self.userOpts.oneShot:
              idxs = self.getIndicesForUniformSampling(maskData, imgData, imgPatchSize)
              numberofSamplesPerRun = min(len(idxs), numberofSamplesPerRun)
              subSampleMethod = self.getUniformlyDistributedSubsamples
            else:
              idxs = self.getIndicesForRandomization(maskData, imgData, imgPatchSize)
              subSampleMethod = self.getRandomSubSamples
          else:
            doSubSampling = False
            imgDataToWork = imgData
            labelDataToWork = labelData
          
          imgIteration = 0
          while True:
            if (doSubSampling):
              subSamples = subSampleMethod(numberofSamplesPerRun, idxs, patchSizes, imgData, labelData, imgIteration)
              imgDataToWork = subSamples[0]
              labelDataToWork = subSamples[1]
            loss = self.optimizeNet(imgDataToWork, labelDataToWork, optimizer)
            
            numpyLoss = loss.detach().cpu().numpy()
            runningLoss[lossCounter] = numpyLoss
            if lossCounter == 9:
              meanLoss = runningLoss.mean()
              logFile.write(str(meanLoss) + ';')
              lossCounter = 0
            else:
              lossCounter+=1
            
            
            imgIteration+=1
            if (datasetIterationValidation(numpyLoss, runningLoss, imgIteration, numberOfiterations, 0)):
              break
            
      epochCount+=1
      if (epochValidation(numpyLoss, runningLoss, epochCount, epochs, 1)):
        break
      
    logFile.close()  
    return loss    


