import torch

import numpy as np
import copy
from Utils import getDefField, combineDeformationFields, deformWholeImage, deformImage, deformLandmarks, deformWithNearestNeighborInterpolation
import SimpleITK as sitk

from eval.LandmarkHandler import PointProcessor, PointReader
import ScalingAndSquaring as sas

import time
import os
from OneShotOptimise import OneShotOptimise
from TrainOptimise import TrainOptimise
from Sampler import Sampler
from Optimise import Optimise


class OptimizeCaller():

  def __init__(self, net, userOpts):
    self.net = net
    self.userOpts = userOpts
    
    self.net.to(self.userOpts.device)
    self.finalNumberIterations = [0,0]

    self.resultModels=[]
    
  def __enter__(self):
        return self
      
  def __exit__(self, exc_type, exc_value, traceback):
    pass
  
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
          
          if int(torch.__version__[0]) < 1:
            deformedLabelTmp = deformWithNearestNeighborInterpolation(labelToDef, defFields[None, imgIdx, chanRange, ], self.userOpts.device)
          else:
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
      
    deformedData['landmarks'] = deformLandmarks(data['landmarks'], data['image'], defField, dataset.getSpacing(datasetIdx), dataset.getOrigin(datasetIdx), dataset.getDirectionCosines(datasetIdx))
    
    return deformedData  
  
  
 
  def setOldModels(self, oldModelList):
    for modelPath in oldModelList:
      modelDict = torch.load(modelPath)
      if modelDict.has_key('optimizer_state_dict'):
        self.resultModels.append({'samplingRate': modelDict['samplingRate'], 'model_state': modelDict['model_state_dict'], 'optimizer_state': modelDict['optimizer_state_dict']})
      else:
        self.resultModels.append({'samplingRate': modelDict['samplingRate'], 'model_state': modelDict['model_state_dict']})


  def resetRandomSeeds(self):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
  
  def resetNet(self):
    self.resetRandomSeeds()
    self.net.reset_params()
  
    
  def testNetDownSamplePatch(self, dataloader):
    samplerShift = (0,0,0)
    logfileName = self.userOpts.outputPath + os.path.sep + 'testLog.csv'
    optimise = Optimise(self.userOpts)
    scalingSquaring = sas.ScalingAndSquaring(self.userOpts.sasSteps)
    with open(logfileName,'w') as testlogFile:
      with torch.no_grad():
#         self.net.eval()
        useContext = self.userOpts.useContext
        self.userOpts.useContext = False
        for datasetIdx , data in enumerate(dataloader, 0):
          if len(self.resultModels) > 0:
            pp = PointProcessor()
            dataBeforeDeformation = data
            
            lastField = None
            for modelIdx, previousModels in enumerate(self.resultModels):
              self.net.load_state_dict(previousModels['model_state'])
              defField = optimise.getDeformationField(data, previousModels['samplingRate'], self.userOpts.patchSize[modelIdx], self.userOpts.useMedianForSampling[modelIdx], samplerShift)
              if lastField is None:
                lastField = defField
              else:
                lastField = combineDeformationFields(defField, lastField)            
              
              self.saveDefField( scalingSquaring(defField) , dataloader, datasetIdx, 'tmpField'+str(modelIdx))
            
            deformationField = scalingSquaring(lastField)
            self.saveDefField(deformationField, dataloader, datasetIdx)
            data = self.applyDefField(deformationField, data, datasetIdx, dataloader.dataset)
            self.saveDeformedData(data, dataloader, datasetIdx)
            pointDistance = ''
            for landmarkChannel in range(-1, len(dataBeforeDeformation['landmarks']) - 1):
              meanPointDistance, _ = pp.calculatePointSetDistance(dataBeforeDeformation['landmarks'][landmarkChannel+1], data['landmarks'][landmarkChannel], False)
              pointDistance += str(meanPointDistance) + ';'
            testlogFile.write(str(datasetIdx) + ';' + pointDistance + '\n')
        self.userOpts.useContext = useContext
    
##
## TODO: find a better way to deal with the resulsModels list; not very elegant to pass them to the trainOptimiser.run method
##
  def trainNetDownSamplePatch(self, dataloader, validationDataLoader):
    with TrainOptimise(self.userOpts) as trainOptimiser:
        samplingRates = self.getDownSampleRates()
        for samplingRateIdx, samplingRate in enumerate(samplingRates):
          self.resetNet()
          if len(self.resultModels) > samplingRateIdx:
            print("found model for saplingrate: " + str(samplingRate))
            if self.userOpts.fineTuneOldModel[samplingRateIdx]:
              print("finetuning model for saplingrate: " + str(samplingRate))
              resultModelsBUP = self.resultModels
              self.resultModels = self.resultModels[0:samplingRateIdx]
              self.net.load_state_dict(resultModelsBUP[samplingRateIdx]['model_state'])
              trainOptimiser.run(self.net, samplingRate, samplingRateIdx, dataloader, validationDataLoader, self.resultModels)
              self.resultModels = resultModelsBUP
              self.resultModels[samplingRateIdx] = {'samplingRate': samplingRate, 'model_state': copy.deepcopy(self.net.state_dict())}
              torch.save({
              'model_state_dict': self.net.state_dict(),
              'samplingRate': samplingRate
              }, self.userOpts.outputPath + os.path.sep + 'finalModel'+str(samplingRateIdx)+'.pt')
            else:
              continue
          else:
            print("train model for saplingrate: " + str(samplingRate))
            trainOptimiser.run(self.net, samplingRate, samplingRateIdx, dataloader, validationDataLoader, self.resultModels)
            self.resultModels.append({'samplingRate': samplingRate, 'model_state': copy.deepcopy(self.net.state_dict())})
            torch.save({
              'model_state_dict': self.net.state_dict(),
              'samplingRate': samplingRate
              }, self.userOpts.outputPath + os.path.sep + 'finalModel'+str(samplingRateIdx)+'.pt')
         
  def trainTestNetDownSamplePatch(self, dataloader):

      with OneShotOptimise(self.userOpts) as oneShotOptimiser:
          samplingRates = self.getDownSampleRates()     
          for i, data in enumerate(dataloader, 0):
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            np.random.seed(0)
            self.net.reset_params()
            self.net.train()
            start = time.time()
            
            currVectorField = oneShotOptimiser.run(self.net, data, dataloader.dataset.getSpacingXZFlip(i), samplingRates)
            
                  
            end = time.time()
            print('Registration of dataset %i took:' % (i), end - start, 'seconds')
            with torch.no_grad():
              
              if self.userOpts.diffeomorphicRegistration:
                currVectorField = currVectorField.to('cpu') # for big datasets we run out of memory when scaling and squaring is run on gpu
                scalingSquaring = sas.ScalingAndSquaring(self.userOpts.sasSteps)
                deformationField = scalingSquaring(currVectorField)
                deformationField = deformationField.to(self.userOpts.device)
              else:
                currVectorField = currVectorField.to(self.userOpts.device)
                deformationField = currVectorField
              
              self.saveResults(data, deformationField, dataloader, i)                  

