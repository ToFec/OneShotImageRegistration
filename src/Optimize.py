import torch
import torch.optim as optim

import numpy as np
from Utils import getDefField, getZeroDefField, smoothArray3D, getMaxIdxs, getPatchSize, deformImage
import SimpleITK as sitk

import LossFunctions as lf
from torch.utils.data import dataloader
from eval.LandmarkHandler import PointProcessor, PointReader

import os


class Optimize():

  def __init__(self, net, userOpts):
    self.net = net
    self.userOpts = userOpts
    weightSum = self.userOpts.ccW + self.userOpts.smoothW + self.userOpts.vecLengthW + self.userOpts.cycleW 
    self.userOpts.ccW = self.userOpts.ccW  / weightSum
    self.userOpts.smoothW = self.userOpts.smoothW  / weightSum
    self.userOpts.vecLengthW = self.userOpts.vecLengthW  / weightSum
    self.userOpts.cycleW = self.userOpts.cycleW  / weightSum
    
    self.net.to(self.userOpts.device)

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
        
        if (maskData.dim() != imgData.dim()):
          maskData = torch.ones(imgShape, dtype=torch.int8)
  
        maxIdxs = getMaxIdxs(imgShape, patchSize)
        patchSizes = getPatchSize(imgShape, patchSize)
        
        defFields = torch.zeros((imgShape[0], imgShape[1] * 3, imgShape[2], imgShape[3], imgShape[4]), device=self.userOpts.device, requires_grad=False)
        indexArray = torch.zeros((imgShape[2], imgShape[3], imgShape[4]), device=self.userOpts.device, requires_grad=False)
        for patchIdx0 in range(0, maxIdxs[0], patchSizes[0]):
          for patchIdx1 in range(0, maxIdxs[1], patchSizes[1]):
            for patchIdx2 in range(0, maxIdxs[2], patchSizes[2]):
              if (maskData[:, :, patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[1], patchIdx2:patchIdx2 + patchSizes[2]].sum() > 0):
                subImgData = imgData[:, :, patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[1], patchIdx2:patchIdx2 + patchSizes[2]]
                indexArray[patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[1], patchIdx2:patchIdx2 + patchSizes[2]] += 1
                defFields[:, :, patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[2], patchIdx2:patchIdx2 + patchSizes[2]] += self.net(subImgData)
           
        leftover0 = imgShape[2] % patchSizes[0]
        startidx0 = patchSizes[0] / 2 if (leftover0 > 0) & (maxIdxs[0] > patchSizes[0])  else leftover0
        leftover1 = imgShape[3] % patchSizes[1]
        startidx1 = patchSizes[1] / 2 if (leftover1 > 0) & (maxIdxs[1] > patchSizes[1])  else leftover1
        leftover2 = imgShape[4] % patchSizes[2]
        startidx2 = patchSizes[2] / 2 if (leftover2 > 0) & (maxIdxs[2] > patchSizes[2])  else leftover2
        
        if (startidx2 + startidx1 + startidx0 > 0) :               
          for patchIdx0 in range(startidx0, maxIdxs[0], patchSizes[0]):
            for patchIdx1 in range(startidx1, maxIdxs[1], patchSizes[1]):
              for patchIdx2 in range(startidx2, maxIdxs[2], patchSizes[2]):
                if (maskData[:, :, patchIdx0, patchIdx1, patchIdx2].sum() > 0):
                  subImgData = imgData[:, :, patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[1], patchIdx2:patchIdx2 + patchSizes[2]]
                  indexArray[patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[1], patchIdx2:patchIdx2 + patchSizes[2]] += 1
                  defFields[:, :, patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[1], patchIdx2:patchIdx2 + patchSizes[2]] += self.net(subImgData)    
        
        del maskData, subImgData
         
        indexArray[indexArray < 1] = 1
        
        for dim0 in range(0, defFields.shape[0]):
          for dim1 in range(0, defFields.shape[1]):
            defFieldsTmp = defFields[dim0, dim1, ] / indexArray
            defFields[dim0, dim1, ] = smoothArray3D(defFieldsTmp, self.userOpts.device)
  
        del indexArray
        
        zeroDefField = getZeroDefField(imgShape)
        zeroDefField = zeroDefField.to(self.userOpts.device)
        for imgIdx in range(imgShape[0]):
          for chanIdx in range(-1, imgShape[1] - 1):
            imgToDef = imgData[None, None, imgIdx, chanIdx, ]
            chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
            deformedTmp = deformImage(imgToDef, defFields[None, imgIdx, chanRange, ], self.userOpts.device)
            
            imgDataDef = sitk.GetImageFromArray(deformedTmp[0, 0, ])
            imgDataOrig = sitk.GetImageFromArray(imgToDef[0,0, ])
            
            dataloader.dataset.saveData(imgDataDef, self.userOpts.outputPath, 'deformedImgDataset' + str(i) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', i, False)
            dataloader.dataset.saveData(imgDataOrig, self.userOpts.outputPath, 'origImgDataset' + str(i) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', i, False)
            defX = defFields[imgIdx, chanIdx * 3, ].detach() * (imgToDef.shape[2] / 2)
            defY = defFields[imgIdx, chanIdx * 3 + 1, ].detach() * (imgToDef.shape[3] / 2)
            defZ = defFields[imgIdx, chanIdx * 3 + 2, ].detach() * (imgToDef.shape[4] / 2)
            defField = getDefField(defX, defY, defZ)
            defDataToSave = sitk.GetImageFromArray(defField, isVector=True)
            dataloader.dataset.saveData(defDataToSave, self.userOpts.outputPath, 'deformationFieldDataset' + str(i) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', i, False)
            
            
            if (len(landmarkData) > 0):
              defField = np.moveaxis(defField, 0, 2)
              defField = np.moveaxis(defField, 0, 1)
              defField = torch.from_numpy(defField)
              currLandmarks = landmarkData[chanIdx + 1] ##the def field points from output to input therefore we need no take the next landmarks to be able to deform them
              deformedPoints = pp.deformPointsWithField(currLandmarks, defField, dataloader.dataset.origins[i], dataloader.dataset.spacings[i])
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
  

  def save_grad(self, name):
  
      def hook(grad):
          print(name)
          print(torch.sum(grad))
  
      return hook
  
    
  def printGPUMemoryAllocated(self):
    torch.cuda.synchronize()
    print(torch.cuda.memory_allocated())
  
    
  def getRandomSubSamples(self, numberofSamplesPerRun, idxs, patchSizes, imgData, labelData):
   
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
    
    cycleImgData = torch.empty(defFields.shape, device=self.userOpts.device)
    
    #         cycleIdxData = torch.empty((imgData.shape[0:2]) + zeroDefField.shape[1:], device=device)
    cycleIdxData = zeroDefField.clone()
    
    for chanIdx in range(-1, imgDataToWork.shape[1] - 1):
      imgToDef = imgDataToWork[:, None, chanIdx, ]
      chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
      deformedTmp = deformImage(imgToDef, defFields[: , chanRange, ], self.userOpts.device, False)
      imgDataDef[:, chanIdx + 1, ] = deformedTmp[:, 0, ]
      
      cycleImgData[:, chanRange, ] = torch.nn.functional.grid_sample(defFields[:, chanRange, ], cycleIdxData.clone(), mode='bilinear', padding_mode='border')
                  
      cycleIdxData[..., 0] = cycleIdxData[..., 0] + defFields[:, chanIdx * 3, ].detach()
      cycleIdxData[..., 1] = cycleIdxData[..., 1] + defFields[:, chanIdx * 3 + 1, ].detach()
      cycleIdxData[..., 2] = cycleIdxData[..., 2] + defFields[:, chanIdx * 3 + 2, ].detach()
    
    del zeroDefField, cycleIdxData
          
    crossCorr = lf.normCrossCorr(imgDataToWork, imgDataDef)
    if imgDataToWork.shape[1] > 3:
      smoothnessDF = lf.smoothnessVecFieldT(defFields, self.userOpts.device)
    else:
      smoothnessDF = lf.smoothnessVecField(defFields, self.userOpts.device)
    
  #   vecLengthLoss = lf.vecLength(defFields)
    vecLengthLoss = torch.abs(defFields).mean()
    cycleLoss = lf.cycleLoss(cycleImgData, self.userOpts.device)
    loss = self.userOpts.ccW * crossCorr + self.userOpts.smoothW * smoothnessDF + self.userOpts.vecLengthW * vecLengthLoss + self.userOpts.cycleW * cycleLoss
#     print('cc: %.5f smmothness: %.5f vecLength: %.5f cycleLoss: %.5f' % (crossCorr, smoothnessDF, vecLengthLoss, cycleLoss))
#     print('loss: %.3f' % (loss))
      
    loss.backward()
    optimizer.step()
    return loss
  
  def terminateLoopByLoss(self, loss, runningLoss, currIteration, itThreshold):
    if (loss == runningLoss.mean()):
      self.finalLoss = loss
      self.finalNumberIterations = currIteration
      return True
    else:
      return False
    
  def terminateLoopByLossAndItCount(self, loss, runningLoss, currIteration, itThreshold):
    if (np.isclose(loss, runningLoss.mean(),atol=self.userOpts.lossTollerance)[0] or currIteration == itThreshold):
      self.finalLoss = loss
      self.finalNumberIterations = currIteration
      return True
    else:
      return False
    
  def terminateLoopByItCount(self, loss, runningLoss, currIteration, itThreshold):
    if (currIteration == itThreshold):
      self.finalLoss = loss
      self.finalNumberIterations = currIteration
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
          
          numberofSamples = (torch.numel(imgData) / maxNumberOfPixs) + 1
          numberOfiterations = (numberofSamples / self.userOpts.maxNumberOfSamples) + 1
          numberOfiterations *= tmpEpochs
          numberofSamplesPerRun = min(numberofSamples, self.userOpts.maxNumberOfSamples - 1)
          
          print('numberOfiterationsPerEpoch: ', numberOfiterations)
          print('numberofSamplesPerIteration: ', numberofSamplesPerRun)
          if torch.numel(imgData) >= maxNumberOfPixs:
            doRandomSubSampling = True
            patchSizes = getPatchSize(imgData.shape, imgPatchSize)
            idxs = self.getIndicesForRandomization(maskData, imgData, imgPatchSize)
          else:
            doRandomSubSampling = False
            imgDataToWork = imgData
            labelDataToWork = labelData
          
          imgIteration = 0
          while True:
            if (doRandomSubSampling):
              randomSubSamples = self.getRandomSubSamples(numberofSamplesPerRun, idxs, patchSizes, imgData, labelData)
              imgDataToWork = randomSubSamples[0]
              labelDataToWork = randomSubSamples[1]
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
            if (datasetIterationValidation(numpyLoss, runningLoss, imgIteration, numberOfiterations)):
              break
            
      epochCount+=1
      if (epochValidation(numpyLoss, runningLoss, epochCount, epochs)):
        break
      
    logFile.close()  
    return loss    


