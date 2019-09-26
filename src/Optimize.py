import torch as torch
import torch.optim as optim

import numpy as np
import copy
from Utils import getDefField, deformImage, deformWholeImage, getReceptiveFieldOffset, sampleImgData, getPaddedData, sampleImg,\
  combineDeformationFields
import SimpleITK as sitk

from eval.LandmarkHandler import PointProcessor, PointReader
import NetOptimizer
from Sampler import Sampler

import time
import os
from __builtin__ import False


class Optimize():

  def __init__(self, net, userOpts):
    self.net = net
    self.userOpts = userOpts
    
    self.net.to(self.userOpts.device)
    self.finalNumberIterations = [0,0]
    
    logfileName = self.userOpts.outputPath + os.path.sep + 'lossLog.csv'
    self.logFile = open(logfileName,'w')
    
    if hasattr(userOpts, 'validationFileNameCSV'):
      validationLogfileName = self.userOpts.outputPath + os.path.sep + 'lossLogValidation.csv'
      self.validationLogFile = open(validationLogfileName,'w')
    self.resultModels=[]
    
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
  
  def saveDefField(self, defFields, dataloader, datasetIdx, outputName='deformationFieldDataset'):
    dataSetSpacing = dataloader.dataset.getSpacing(datasetIdx)
    dataSetDirCosines = dataloader.dataset.getDirectionCosines(datasetIdx)
    for imgIdx in range(defFields.shape[0]):
      for chanIdx in range(-1, (defFields.shape[1]/3) - 1):
        defX = defFields[imgIdx, chanIdx * 3, ].detach() * dataSetSpacing[0] * dataSetDirCosines[0]
        defY = defFields[imgIdx, chanIdx * 3 + 1, ].detach() * dataSetSpacing[1] * dataSetDirCosines[4]
        defZ = defFields[imgIdx, chanIdx * 3 + 2, ].detach() * dataSetSpacing[2] * dataSetDirCosines[8]
        defField = getDefField(defX, defY, defZ)
        defDataToSave = sitk.GetImageFromArray(defField, isVector=True)
        dataloader.dataset.saveData(defDataToSave, self.userOpts.outputPath, outputName + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
          
  def saveDeformedData(self, data, dataloader, datasetIdx, imgName='imgDataset'):
    imgData = data['image']
    labelData = data['label']
    landmarkData = data['landmarks']
    pr = PointReader()
    for imgIdx in range(imgData.shape[0]):
      for chanIdx in range(-1, imgData.shape[1] - 1):
        imgToDef = imgData[None, None, imgIdx, chanIdx, ]
        
        if (labelData is not None) and (labelData.dim() == imgData.dim()):
          labelToDef = labelData[None, None, imgIdx, chanIdx, ].float()
          labelToDef = labelToDef.to(self.userOpts.device)
          labelDataOrig = sitk.GetImageFromArray(labelToDef[0, 0, ])
          dataloader.dataset.saveData(labelDataOrig, self.userOpts.outputPath, 'labelDataset' + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
        
        imgDataOrig = sitk.GetImageFromArray(imgToDef[0,0, ])
        dataloader.dataset.saveData(imgDataOrig, self.userOpts.outputPath, imgName + str(datasetIdx) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', datasetIdx, False)
        
        if (len(landmarkData) > 0):
          currLandmarks = landmarkData[chanIdx + 1] ##the def field points from output to input therefore we need no take the next landmarks to be able to deform them
          pr.saveDataTensor(self.userOpts.outputPath + os.path.sep + 'dataset' + str(datasetIdx) + 'channel' + str(chanIdx+1) + '0deformed.pts', currLandmarks)
            
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
  
  def validateModel(self, validationDataLoader, netOptim, samplingRate, samplingRateIdx, padVals, samplerShift):
    with torch.no_grad():
      self.net.eval()
      useContext = self.userOpts.useContext
      self.userOpts.useContext = False
      validationLosses = []
      landmarkDistances = []
      pp = PointProcessor()
      for validationDataIdx , validationData in enumerate(validationDataLoader, 0):
        if not self.userOpts.usePaddedNet:
          validationData['image'], validationData['mask'], validationData['label'] = getPaddedData(validationData['image'], validationData['mask'], validationData['label'], padVals)
        dataBeforeDeformation = validationData
        if len(self.resultModels) > 0:
          
          currentState = copy.deepcopy(self.net.state_dict())
          lastField = None
          for modelIdx, previousModels in enumerate(self.resultModels):
            self.net.load_state_dict(previousModels['model_state'])
            defField = self.getDeformationField(validationData, previousModels['samplingRate'], self.userOpts.patchSize[modelIdx], self.userOpts.useMedianForSampling[modelIdx], samplerShift)
            if lastField is None:
              lastField = defField
            else:
              lastField = combineDeformationFields(defField, lastField)            
          self.net.load_state_dict(currentState)
        
        currValidationField = self.getDeformationField(validationData, samplingRate, self.userOpts.patchSize[samplingRateIdx], self.userOpts.useMedianForSampling[samplingRateIdx], samplerShift)
        currValidationField = combineDeformationFields(currValidationField, lastField)
        
        validationLoss = netOptim.calculateLoss(validationData['image'].to(self.userOpts.device), currValidationField, samplingRateIdx, (0, 0, 0, validationData['image'].shape[2],validationData['image'].shape[3], validationData['image'].shape[4]))
        validationLosses.append(float(validationLoss.detach()))
        
        if len(dataBeforeDeformation['landmarks']) > 0:
          validationData['landmarks'] = self.deformLandmarks(validationData['landmarks'], validationData['image'], currValidationField, validationDataLoader.dataset.getSpacing(validationDataIdx),
                                validationDataLoader.dataset.getOrigin(validationDataIdx), 
                                validationDataLoader.dataset.getDirectionCosines(validationDataIdx))
          totalMeanPointDist = 0.0
          for landmarkChannel in range(-1, len(dataBeforeDeformation['landmarks']) - 1):
            meanPointDistance, _ = pp.calculatePointSetDistance(dataBeforeDeformation['landmarks'][landmarkChannel+1], validationData['landmarks'][landmarkChannel], False)
            totalMeanPointDist += meanPointDistance
          landmarkDistances.append(totalMeanPointDist / float(len(dataBeforeDeformation['landmarks'])))
                
      del validationData, currValidationField
      
      self.userOpts.useContext = useContext 
      return validationLosses, landmarkDistances
    
  def applyModel(self, modelToApply, data, previousSampleIdxs, samplerShift, datasetIdx, dataset):
    self.net.load_state_dict(modelToApply['model_state'])
    defField = self.getDeformationField(data, modelToApply['samplingRate'], self.userOpts.patchSize[previousSampleIdxs], self.userOpts.useMedianForSampling[previousSampleIdxs], samplerShift)
    deformedData = {'image': deformWholeImage(data['image'].to(self.userOpts.device), defField, channelOffset = 0)}
    if (data['mask'].dim() == data['image'].dim()):
      deformedData['mask'] = deformWholeImage(data['mask'].to(self.userOpts.device), defField,True, channelOffset = 0)
    else:
      deformedData['mask'] = torch.tensor([1])
    if (data['label'].dim() == data['image'].dim()):
      deformedData['label'] = deformWholeImage(data['label'].to(self.userOpts.device), defField,True, channelOffset = 0)
    else:
      deformedData['label'] = torch.tensor([1])      
      
    deformedData['landmarks'] = self.deformLandmarks(data['landmarks'], data['image'], defField, dataset.getSpacing(datasetIdx), dataset.getOrigin(datasetIdx), dataset.getDirectionCosines(datasetIdx))
    
    return deformedData, defField
  
  def applyDefField(self, defField, data, datasetIdx, dataset):
    deformedData = {'image': deformWholeImage(data['image'].to(self.userOpts.device), defField, channelOffset = 0)}
    if (data['mask'].dim() == data['image'].dim()):
      deformedData['mask'] = deformWholeImage(data['mask'].to(self.userOpts.device), defField,True, channelOffset = 0)
    else:
      deformedData['mask'] = torch.tensor([1])
    if (data['label'].dim() == data['image'].dim()):
      deformedData['label'] = deformWholeImage(data['label'].to(self.userOpts.device), defField,True, channelOffset = 0)
    else:
      deformedData['label'] = torch.tensor([1])      
      
    deformedData['landmarks'] = self.deformLandmarks(data['landmarks'], data['image'], defField, dataset.getSpacing(datasetIdx), dataset.getOrigin(datasetIdx), dataset.getDirectionCosines(datasetIdx))
    
    return deformedData  
  
  def deformLandmarks(self, landmarkData, image, defField, spacing, origin, cosines):
    if (len(landmarkData) > 0):
      pp = PointProcessor()
      deformedlandmarkData = list(landmarkData)
      for imgIdx in range(image.shape[0]):
        for chanIdx in range(-1, image.shape[1] - 1):
          dataSetSpacing = spacing
          dataSetDirCosines = cosines
          defX = defField[imgIdx, chanIdx * 3, ].detach() * dataSetSpacing[0] * dataSetDirCosines[0]
          defY = defField[imgIdx, chanIdx * 3 + 1, ].detach() * dataSetSpacing[1] * dataSetDirCosines[4]
          defZ = defField[imgIdx, chanIdx * 3 + 2, ].detach() * dataSetSpacing[2] * dataSetDirCosines[8]
          defFieldPerturbated = getDefField(defX, defY, defZ)
          defFieldPerturbated = np.moveaxis(defFieldPerturbated, 0, 2)
          defFieldPerturbated = np.moveaxis(defFieldPerturbated, 0, 1)
          defFieldPerturbated = torch.from_numpy(defFieldPerturbated)
          currLandmarks = landmarkData[chanIdx + 1] ##the def field points from output to input therefore we need no take the next landmarks to be able to deform them
          defFieldOrigin = origin
          deformedPoints = pp.deformPointsWithField(currLandmarks, defFieldPerturbated, defFieldOrigin, dataSetSpacing, dataSetDirCosines)
          deformedlandmarkData[chanIdx + 1]=deformedPoints
      deformedData=deformedlandmarkData
    else:
      deformedData=[]   
    return deformedData 
  
  def trainModel(self, samplingRate, samplingRateIdx, dataloader, validationDataLoader):
    self.net.train()
    optimizer = optim.Adam(self.net.parameters(),amsgrad=True)
    netOptim = NetOptimizer.NetOptimizer(self.net, None, optimizer, self.userOpts)
    
    receptiveFieldOffset = getReceptiveFieldOffset(self.userOpts.netDepth)
    padVals = (receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset)
    samplerShift = (0,0,0)
    if not self.userOpts.usePaddedNet:
      samplerShift = (receptiveFieldOffset*2,receptiveFieldOffset*2,receptiveFieldOffset*2)
    
    for epoch in range(self.userOpts.numberOfEpochs[samplingRateIdx]):
      for i, data in enumerate(dataloader, 0):
        netOptim.setSpacing(dataloader.dataset.getSpacingXZFlip(i))
        if not self.userOpts.usePaddedNet:
          data['image'], data['mask'], data['label'] = getPaddedData(data['image'], data['mask'], data['label'], padVals)
         
        lastField = None 
        if len(self.resultModels) > 0:
          currentState = copy.deepcopy(self.net.state_dict())
          with torch.no_grad():
            self.net.eval()
            useContext = self.userOpts.useContext
            self.userOpts.useContext = False
            for previousSampleIdxs in range(samplingRateIdx):
              modelToApply = self.resultModels[previousSampleIdxs]
              self.net.load_state_dict(modelToApply['model_state'])
              defField = self.getDeformationField(data, modelToApply['samplingRate'], self.userOpts.patchSize[previousSampleIdxs], self.userOpts.useMedianForSampling[previousSampleIdxs], samplerShift)
              if lastField is None:
                lastField = defField
              else:
                lastField = combineDeformationFields(defField, lastField)
            self.userOpts.useContext = useContext
            self.net.load_state_dict(currentState)
            self.net.train()
        
        
        sampledImgData, sampledMaskData, sampledLabelData, _ = sampleImgData(data, samplingRate)
        if lastField is None:
          currDefField = torch.zeros((data['image'].shape[0], data['image'].shape[1] * 3, data['image'].shape[2], data['image'].shape[3], data['image'].shape[4]), device="cpu", requires_grad=False)
        else:
          currDefField = lastField
              
        
        currDefField = currDefField * samplingRate
        currDefField = sampleImg(currDefField, samplingRate)
        
        sampler = Sampler(sampledMaskData, sampledImgData, sampledLabelData, self.userOpts.patchSize[samplingRateIdx]) 
        numberofSamplesPerRun = int(sampledImgData.numel() / (self.userOpts.patchSize[0] * self.userOpts.patchSize[1] * self.userOpts.patchSize[2]))
        if numberofSamplesPerRun < 1:
          numberofSamplesPerRun = 1
        idxs = sampler.getIndicesForRandomization()
        (imgDataToWork, _, usedIdx) = sampler.getRandomSubSamples(numberofSamplesPerRun, idxs)
        imgDataToWork = imgDataToWork.to(self.userOpts.device)
        lastDeffield = currDefField.clone()
        for sample in range(numberofSamplesPerRun):
          
          currDefFieldIdx, offset = self.getSubCurrDefFieldIdx(currDefField, usedIdx[sample])
          currDefFieldGPU = currDefField[:, :, currDefFieldIdx[0]:currDefFieldIdx[0]+currDefFieldIdx[3], currDefFieldIdx[1]:currDefFieldIdx[1]+currDefFieldIdx[4], currDefFieldIdx[2]:currDefFieldIdx[2]+currDefFieldIdx[5]].to(device=self.userOpts.device)
          lastDeffieldGPU = lastDeffield[:, :, currDefFieldIdx[0]:currDefFieldIdx[0]+currDefFieldIdx[3], currDefFieldIdx[1]:currDefFieldIdx[1]+currDefFieldIdx[4], currDefFieldIdx[2]:currDefFieldIdx[2]+currDefFieldIdx[5]].to(device=self.userOpts.device)
          
#           loss = netOptim.optimizeNetTrain(imgDataToWork[sample,None,...], samplingRateIdx)
          loss = netOptim.optimizeNetOneShot(imgDataToWork[sample,None,...], None, lastDeffieldGPU, currDefFieldGPU, offset, samplingRateIdx, False)
          
          detachLoss = loss.detach()                
          self.logFile.write(str(epoch) + ';' + str(float(detachLoss)) + '\n')
          self.logFile.flush()
          
          currDefField[:, :, usedIdx[sample][0]:usedIdx[sample][0]+imgDataToWork.shape[2], usedIdx[sample][1]:usedIdx[sample][1]+imgDataToWork.shape[3], usedIdx[sample][2]:usedIdx[sample][2]+imgDataToWork.shape[4]] = currDefFieldGPU[:,:,offset[0]:offset[0]+imgDataToWork.shape[2],offset[1]:offset[1]+imgDataToWork.shape[3],offset[2]:offset[2]+imgDataToWork.shape[4]].to("cpu")
        del imgDataToWork, sampledImgData, sampledMaskData, sampledLabelData 
      
      
      ##
      ## Validation
      ##
      if epoch % self.userOpts.validationIntervall == 0:
        torch.save({
              'epoch': epoch,
              'model_state_dict': self.net.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': loss,
              'samplingRate': samplingRate
              }, self.userOpts.outputPath + os.path.sep + 'registrationModel'+str(samplingRateIdx)+'.pt')
        validationLosses, landmarkDistances = self.validateModel(validationDataLoader, netOptim, samplingRate, samplingRateIdx, padVals, samplerShift)
        if len(landmarkDistances) > 0:
          self.validationLogFile.write(str(epoch) + ';' + str(np.mean(validationLosses[0])) + ';' + str(np.std(validationLosses)) + ';' + str(np.mean(landmarkDistances)) + '\n')
        else:
          self.validationLogFile.write(str(epoch) + ';' + str(np.mean(validationLosses[0])) + ';' + str(np.std(validationLosses)) + ';' + '0.0' + '\n')
        self.validationLogFile.flush()
        del validationLosses
        self.net.train() 
  
  def setOldModels(self, oldModelList):
    for modelPath in oldModelList:
      modelDict = torch.load(modelPath)
      self.resultModels.append({'samplingRate': modelDict['samplingRate'], 'model_state': modelDict['model_state_dict']})
  
  def resetNet(self):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    self.net.reset_params()
  
  def testNetDownSamplePatch(self, dataloader):
    samplerShift = (0,0,0)
    receptiveFieldOffset = getReceptiveFieldOffset(self.userOpts.netDepth)
    if not self.userOpts.usePaddedNet:
      samplerShift = (receptiveFieldOffset*2,receptiveFieldOffset*2,receptiveFieldOffset*2)
    with torch.no_grad():
      self.net.eval()
      useContext = self.userOpts.useContext
      self.userOpts.useContext = False
      for datasetIdx , data in enumerate(dataloader, 0):
        if not self.userOpts.usePaddedNet:
          data['image'], data['mask'], data['label'] = getPaddedData(data['image'], data['mask'], data['label'], data)
        if len(self.resultModels) > 0:
          pp = PointProcessor()
          dataBeforeDeformation = data
          
          lastField = None
          for modelIdx, previousModels in enumerate(self.resultModels):
            self.net.load_state_dict(previousModels['model_state'])
            defField = self.getDeformationField(data, previousModels['samplingRate'], self.userOpts.patchSize[modelIdx], self.userOpts.useMedianForSampling[modelIdx], samplerShift)
            if lastField is None:
              lastField = defField
            else:
              lastField = combineDeformationFields(defField, lastField)            
            
            self.saveDefField(defField, dataloader, datasetIdx, 'tmpField'+str(modelIdx))
          
          self.saveDefField(lastField, dataloader, datasetIdx)
          data = self.applyDefField(lastField, data, datasetIdx, dataloader.dataset)
          self.saveDeformedData(data, dataloader, datasetIdx)
          for landmarkChannel in range(-1, len(dataBeforeDeformation['landmarks']) - 1):
            meanPointDistance, _ = pp.calculatePointSetDistance(dataBeforeDeformation['landmarks'][landmarkChannel+1], data['landmarks'][landmarkChannel], False)
            print(meanPointDistance)
         
      self.userOpts.useContext = useContext
            
    
  def trainNetDownSamplePatch(self, dataloader, validationDataLoader):
    samplingRates = self.getDownSampleRates()
    
    for samplingRateIdx, samplingRate in enumerate(samplingRates):
      if len(self.resultModels) > samplingRateIdx:
        if self.userOpts.fineTuneOldModel[samplingRateIdx]:
          resultModelsBUP = self.resultModels
          self.resultModels = self.resultModels[0:samplingRateIdx]
          self.net.load_state_dict(resultModelsBUP[samplingRateIdx]['model_state'])
          self.trainModel(samplingRate, samplingRateIdx, dataloader, validationDataLoader)
          self.resultModels = resultModelsBUP
          self.resultModels[samplingRateIdx] = {'samplingRate': samplingRate, 'model_state': copy.deepcopy(self.net.state_dict())}
          torch.save({
          'model_state_dict': self.net.state_dict(),
          'samplingRate': samplingRate
          }, self.userOpts.outputPath + os.path.sep + 'finalModel'+str(samplingRateIdx)+'.pt')
        else:
          continue
      else:
        self.resetNet()
        self.trainModel(samplingRate, samplingRateIdx, dataloader, validationDataLoader)
        self.resultModels.append({'samplingRate': samplingRate, 'model_state': copy.deepcopy(self.net.state_dict())})
        torch.save({
          'model_state_dict': self.net.state_dict(),
          'samplingRate': samplingRate
          }, self.userOpts.outputPath + os.path.sep + 'finalModel'+str(samplingRateIdx)+'.pt')
     
  def getDeformationField(self, imageData, samplingRate, patchSize, useMedianSampling, samplerShift):
    sampledValidationImgData, sampledValidationMaskData, sampledValidationLabelData, _ = sampleImgData(imageData, samplingRate)
    validationSampler = Sampler(sampledValidationMaskData, sampledValidationImgData, sampledValidationLabelData, patchSize) 
    idxs = validationSampler.getIndicesForOneShotSampling(samplerShift, useMedianSampling)
    currValidationField = torch.zeros((sampledValidationImgData.shape[0], sampledValidationImgData.shape[1] * 3, sampledValidationImgData.shape[2], sampledValidationImgData.shape[3], sampledValidationImgData.shape[4]), device=self.userOpts.device, requires_grad=False)
    for _ , idx in enumerate(idxs):
      validationImageSample = validationSampler.getSubSampleImg(idx, self.userOpts.normImgPatches)
      validationImageSample = validationImageSample.to(self.userOpts.device)
      defField = self.net(validationImageSample)
      currValidationField[:, :, idx[0]:idx[0]+validationImageSample.shape[2], idx[1]:idx[1]+validationImageSample.shape[3], idx[2]:idx[2]+validationImageSample.shape[4]] = defField
      
    upSampleRate = 1.0 / samplingRate
    currValidationField = currValidationField * upSampleRate
    currValidationField = sampleImg(currValidationField, upSampleRate)
    return currValidationField
        
  def trainTestNetDownSamplePatch(self, dataloader):
      if self.userOpts.trainTillConvergence:
        iterationValidation = self.terminateLoopByLossAndItCount
      else:
        iterationValidation = self.terminateLoopByItCount
      
      numberOfiterations = self.userOpts.numberOfEpochs[0]
      
      receptiveFieldOffset = getReceptiveFieldOffset(self.userOpts.netDepth)
      printLossAndCropGrads = False
      for i, data in enumerate(dataloader, 0):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        self.net.reset_params()
        optimizer = optim.Adam(self.net.parameters(),amsgrad=True)
        
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
          sampler = Sampler(sampledMaskData, sampledImgData, sampledLabelData, self.userOpts.patchSize[samplingRateIdx]) 
          idxs = sampler.getIndicesForOneShotSampling(samplerShift, self.userOpts.useMedianForSampling[samplingRateIdx])
          
          print('idxs: ', idxs)
          
          if currDefField is None:
            currDefField = torch.zeros((sampledImgData.shape[0], sampledImgData.shape[1] * 3, sampledImgData.shape[2], sampledImgData.shape[3], sampledImgData.shape[4]), device="cpu", requires_grad=False)
          
          for ltIdx , lossTollerance in enumerate(self.userOpts.lossTollerances):
            print('lossTollerance: ', lossTollerance)
            
            lastDeffield = currDefField.clone()
            for patchIdx, idx in enumerate(idxs):
              print('register patch %i out of %i patches.' % (patchIdx, len(idxs)))

              optimizer = optim.Adam(self.net.parameters(),amsgrad=True)
              netOptim.setOptimizer(optimizer)
              
              imgDataToWork = sampler.getSubSampleImg(idx, self.userOpts.normImgPatches)
              imgDataToWork = imgDataToWork.to(self.userOpts.device)
              
              currDefFieldIdx, offset = self.getSubCurrDefFieldIdx(currDefField, idx)
              currDefFieldGPU = currDefField[:, :, currDefFieldIdx[0]:currDefFieldIdx[0]+currDefFieldIdx[3], currDefFieldIdx[1]:currDefFieldIdx[1]+currDefFieldIdx[4], currDefFieldIdx[2]:currDefFieldIdx[2]+currDefFieldIdx[5]].to(device=self.userOpts.device)
              lastDeffieldGPU = lastDeffield[:, :, currDefFieldIdx[0]:currDefFieldIdx[0]+currDefFieldIdx[3], currDefFieldIdx[1]:currDefFieldIdx[1]+currDefFieldIdx[4], currDefFieldIdx[2]:currDefFieldIdx[2]+currDefFieldIdx[5]].to(device=self.userOpts.device)
              
              patchIteration=0
              lossCounter = 0
              runningLoss = torch.ones(10, device=self.userOpts.device)
              while True:
                loss = netOptim.optimizeNetOneShot(imgDataToWork, None, lastDeffieldGPU, currDefFieldGPU, offset, samplingRateIdx+ltIdx, printLossAndCropGrads)
                if printLossAndCropGrads:
                  self.printParameterInfo()
                detachLoss = loss.detach()                
                runningLoss[lossCounter] = detachLoss
                if lossCounter == 9:
                  meanLoss = runningLoss.mean()
                  self.logFile.write(str(float(meanLoss)) + ';' + str(patchIdx) + '\n')
                  self.logFile.flush()
                  lossCounter = 0
                  if (iterationValidation(detachLoss, meanLoss, patchIteration, numberOfiterations, 0, lossTollerance)):
                    break
                else:
                  lossCounter+=1
                patchIteration+=1
              currDefField[:, :, idx[0]:idx[0]+imgDataToWork.shape[2], idx[1]:idx[1]+imgDataToWork.shape[3], idx[2]:idx[2]+imgDataToWork.shape[4]] = currDefFieldGPU[:,:,offset[0]:offset[0]+imgDataToWork.shape[2],offset[1]:offset[1]+imgDataToWork.shape[3],offset[2]:offset[2]+imgDataToWork.shape[4]].to("cpu")
              
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
        
        currDefField = currDefField.to(self.userOpts.device)
        self.saveResults(data, currDefField, dataloader, i)                  


