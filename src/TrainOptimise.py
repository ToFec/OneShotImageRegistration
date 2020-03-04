'''
Created on Nov 22, 2019

@author: fechter
'''
from Optimise import Optimise
from torch import optim
import NetOptimizer
import Utils
import copy
import torch
import numpy as np
import os
from eval.LandmarkHandler import PointProcessor
from Sampler import Sampler
import LossFunctions as lf
from ScalingAndSquaring import ScalingAndSquaring
from Utils import deformLandmarks, sampleImg

class TrainOptimise(Optimise):


    def __init__(self, userOpts):
      Optimise.__init__(self, userOpts)
      if hasattr(userOpts, 'validationFileNameCSV'):
        validationLogfileName = self.userOpts.outputPath + os.path.sep + 'lossLogValidation.csv'
        self.validationLogFile = open(validationLogfileName,'w')
        
    def __exit__(self, exc_type, exc_value, traceback):
      self.validationLogFile.close()         
      
    def run(self, net, samplingRate, samplingRateIdx, dataloader, validationDataLoader, resultModels=[]):
      self.net = net
      self.net.train()
      optimizer = optim.Adam(self.net.parameters(),amsgrad=True)
      netOptim = NetOptimizer.NetOptimizer(self.net, None, optimizer, self.userOpts)
      
      receptiveFieldOffset = Utils.getReceptiveFieldOffset(self.userOpts.netDepth)
      padVals = (receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset, receptiveFieldOffset)
      samplerShift = (0,0,0)
      
      for epoch in range(self.userOpts.numberOfEpochs[samplingRateIdx]):
        for i, data in enumerate(dataloader, 0):
          netOptim.initSmoother(data['image'].shape[1]*3)
           
          lastField = None 
          if len(resultModels) > 0:
            currentState = copy.deepcopy(self.net.state_dict())
            with torch.no_grad():
  #             self.net.eval()
              useContext = self.userOpts.useContext
              self.userOpts.useContext = False
              for previousSampleIdxs in range(samplingRateIdx):
                modelToApply = resultModels[previousSampleIdxs]
                self.net.load_state_dict(modelToApply['model_state'])
                defField = self.getDeformationField(data, modelToApply['samplingRate'], self.userOpts.patchSize[previousSampleIdxs], self.userOpts.useMedianForSampling[previousSampleIdxs], samplerShift)
                if lastField is None:
                  lastField = defField
                else:
                  lastField = Utils.combineDeformationFields(defField, lastField)
              self.userOpts.useContext = useContext
            self.net.load_state_dict(currentState)
            self.net.train()
          
          
          sampledImgData, sampledMaskData, sampledLabelData, _ = Utils.sampleImgData(data, samplingRate)
          if lastField is None:
            currDefField = torch.zeros((data['image'].shape[0], data['image'].shape[1] * 3, data['image'].shape[2], data['image'].shape[3], data['image'].shape[4]), device="cpu", requires_grad=False)
          else:
            currDefField = lastField
                
          
          currDefField = currDefField * samplingRate
          currDefField = Utils.sampleImg(currDefField, samplingRate)
          
          sampler = Sampler(sampledMaskData, sampledImgData, sampledLabelData, self.userOpts.patchSize[samplingRateIdx])
          
          if self.userOpts.randomSampling[samplingRateIdx]:
            numberofSamplesPerRun = int(sampledImgData.numel() / (self.userOpts.patchSize[samplingRateIdx] * self.userOpts.patchSize[samplingRateIdx] * self.userOpts.patchSize[samplingRateIdx]))
            if numberofSamplesPerRun < 1:
              numberofSamplesPerRun = 1
            idxs = sampler.getIndicesForRandomization()
            idxs = sampler.getRandomSubSamplesIdxs(numberofSamplesPerRun, idxs)
          else:
            idxs = sampler.getIndicesForOneShotSampling(samplerShift, self.userOpts.useMedianForSampling[samplingRateIdx])
          
          lastDeffield = currDefField.clone()
          for _ , idx in enumerate(idxs):
            imgDataToWork = sampler.getSubSampleImg(idx, self.userOpts.normImgPatches)
            imgDataToWork = imgDataToWork.to(self.userOpts.device)
            lossCalculator = lf.LossFunctions(imgDataToWork, dataloader.dataset.getSpacingXZFlip(i))
            currDefFieldIdx, offset = self.getSubCurrDefFieldIdx(currDefField, idx)
            currDefFieldGPU = currDefField[:, :, currDefFieldIdx[0]:currDefFieldIdx[0]+currDefFieldIdx[3], currDefFieldIdx[1]:currDefFieldIdx[1]+currDefFieldIdx[4], currDefFieldIdx[2]:currDefFieldIdx[2]+currDefFieldIdx[5]].to(device=self.userOpts.device)
            lastDeffieldGPU = lastDeffield[:, :, currDefFieldIdx[0]:currDefFieldIdx[0]+currDefFieldIdx[3], currDefFieldIdx[1]:currDefFieldIdx[1]+currDefFieldIdx[4], currDefFieldIdx[2]:currDefFieldIdx[2]+currDefFieldIdx[5]].to(device=self.userOpts.device)
            
            [loss, crossCorr, diceLoss, smoothnessDF, cycleLoss] = netOptim.optimizeNet(imgDataToWork, lossCalculator, None, lastDeffieldGPU, currDefFieldGPU, offset, samplingRateIdx, False)
            #TODO: log also other loss values
            detachLoss = loss.detach()                
            self.logFile.write(str(epoch) + ';' + str(float(detachLoss)) + '\n')
            self.logFile.flush()
            
            currDefField[:, :, idx[0]:idx[0]+imgDataToWork.shape[2], idx[1]:idx[1]+imgDataToWork.shape[3], idx[2]:idx[2]+imgDataToWork.shape[4]] = currDefFieldGPU[:,:,offset[0]:offset[0]+imgDataToWork.shape[2],offset[1]:offset[1]+imgDataToWork.shape[3],offset[2]:offset[2]+imgDataToWork.shape[4]].to("cpu")
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
                }, self.userOpts.outputPath + os.path.sep + 'registrationModel'+str(samplingRateIdx)+str(epoch)+'.pt')
          validationLosses, landmarkDistances = self.validateModel(validationDataLoader, netOptim, samplingRate, samplingRateIdx, padVals, samplerShift, resultModels)
          if len(landmarkDistances) > 0:
            self.validationLogFile.write(str(epoch) + ';' + str(np.mean(validationLosses)) + ';' + str(np.std(validationLosses)) + ';' + str(np.mean(landmarkDistances)) + ';' + '\n')
          else:
            self.validationLogFile.write(str(epoch) + ';' + str(np.mean(validationLosses)) + ';' + str(np.std(validationLosses)) + ';' + '0.0' + '\n')
          self.validationLogFile.flush()
          del validationLosses
          self.net.train()
      return self.net 
          
    def validateModel(self, validationDataLoader, netOptim, samplingRate, samplingRateIdx, padVals, samplerShift, resultModels = []):
      with torch.no_grad():
  #       self.net.eval()
        useContext = self.userOpts.useContext
        self.userOpts.useContext = False
        validationLosses = []
        landmarkDistances = []
        pp = PointProcessor()
        for validationDataIdx , validationData in enumerate(validationDataLoader, 0):
          landmarksBeforeDeformation = validationData['landmarks']
          
          lastField = None
          if len(resultModels) > 0:
            currentState = copy.deepcopy(self.net.state_dict())
            for modelIdx, previousModels in enumerate(resultModels):
              self.net.load_state_dict(previousModels['model_state'])
              defField = self.getDeformationField(validationData, previousModels['samplingRate'], self.userOpts.patchSize[modelIdx], self.userOpts.useMedianForSampling[modelIdx], samplerShift)
              if lastField is None:
                lastField = defField
              else:
                lastField = Utils.combineDeformationFields(defField, lastField)            
            self.net.load_state_dict(currentState)
          
          #############
          #############
          #############
  #         sampler = Sampler( validationData['mask'], validationData['image'], validationData['label'], self.userOpts.patchSize[samplingRateIdx])
  #         idxs = sampler.getIndicesForOneShotSampling(samplerShift, self.userOpts.useMedianForSampling[samplingRateIdx])        
  #         if lastField is None:
  #           currValidationField = torch.zeros((validationData['image'].shape[0], validationData['image'].shape[1] * 3, validationData['image'].shape[2], validationData['image'].shape[3], validationData['image'].shape[4]), device="cpu", requires_grad=False)
  #         else:
  #           currValidationField = lastField.clone()
  #           
  #         currValidationField = currValidationField * samplingRate
  #         currValidationField = sampleImg(currValidationField, samplingRate)
  #         
  #         for _ , idx in enumerate(idxs):
  #           imgDataToWork = sampler.getSubSampleImg(idx, self.userOpts.normImgPatches)
  #           imgDataToWork = imgDataToWork.to(self.userOpts.device)
  #           currDefFieldIdx, offset = self.getSubCurrDefFieldIdx(currValidationField, idx)
  #           currDefFieldGPU = currValidationField[:, :, currDefFieldIdx[0]:currDefFieldIdx[0]+currDefFieldIdx[3], currDefFieldIdx[1]:currDefFieldIdx[1]+currDefFieldIdx[4], currDefFieldIdx[2]:currDefFieldIdx[2]+currDefFieldIdx[5]].to(device=self.userOpts.device)
  #           lastDeffieldGPU = lastField[:, :, currDefFieldIdx[0]:currDefFieldIdx[0]+currDefFieldIdx[3], currDefFieldIdx[1]:currDefFieldIdx[1]+currDefFieldIdx[4], currDefFieldIdx[2]:currDefFieldIdx[2]+currDefFieldIdx[5]].to(device=self.userOpts.device)
  #            
  #           loss = netOptim.optimizeNetOneShot(imgDataToWork, None, lastDeffieldGPU, currDefFieldGPU, offset, samplingRateIdx, False, optimizeTmp=False)
  #            
  #           detachLoss = loss.detach()      
  #           validationLosses.append(float(detachLoss))          
  #  
  #           currValidationField[:, :, idx[0]:idx[0]+imgDataToWork.shape[2], idx[1]:idx[1]+imgDataToWork.shape[3], idx[2]:idx[2]+imgDataToWork.shape[4]] = currDefFieldGPU[:,:,offset[0]:offset[0]+imgDataToWork.shape[2],offset[1]:offset[1]+imgDataToWork.shape[3],offset[2]:offset[2]+imgDataToWork.shape[4]].to("cpu")        
           
          #############
          #############
          #############
          
          currValidationField = self.getDeformationField(validationData, samplingRate, self.userOpts.patchSize[samplingRateIdx], self.userOpts.useMedianForSampling[samplingRateIdx], samplerShift)
          if lastField is not None:
            currValidationField = Utils.combineDeformationFields(currValidationField, lastField)
          #validationLoss = netOptim.calculateLoss(validationData['image'].to(self.userOpts.device), currValidationField, samplingRateIdx, (0, 0, 0, validationData['image'].shape[2],validationData['image'].shape[3], validationData['image'].shape[4]))
          validationLosses.append(0.0)#float(validationLoss.detach()))
          
          if len(landmarksBeforeDeformation) > 0:
            if self.userOpts.diffeomorphicRegistration:
              sas = ScalingAndSquaring(self.userOpts.sasSteps)
              deformationField = sas(currValidationField)
            else:
              deformationField = currValidationField
            upSampleRate = 1.0 / samplingRate
            deformationField = deformationField * upSampleRate
            deformationField = sampleImg(deformationField, upSampleRate)
            validationData['landmarks'] = deformLandmarks(validationData['landmarks'], validationData['image'], deformationField, validationDataLoader.dataset.getSpacing(validationDataIdx),
                                  validationDataLoader.dataset.getOrigin(validationDataIdx), 
                                  validationDataLoader.dataset.getDirectionCosines(validationDataIdx))
            totalMeanPointDist = 0.0
            for landmarkChannel in range(-1, len(landmarksBeforeDeformation) - 1):
              meanPointDistance, _ = pp.calculatePointSetDistance(landmarksBeforeDeformation[landmarkChannel+1], validationData['landmarks'][landmarkChannel], False)
              totalMeanPointDist += meanPointDistance
            landmarkDistances.append(totalMeanPointDist / float(len(landmarksBeforeDeformation)))
                  
        del validationData, currValidationField
        
        self.userOpts.useContext = useContext 
        return validationLosses, landmarkDistances          
        