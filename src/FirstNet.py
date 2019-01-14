import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import os
import sys, getopt
import numpy as np
from Utils import deform, getDefField, getZeroDefField
import SimpleITK as sitk

import time

from HeadAndNeckDataset import HeadAndNeckDataset, ToTensor
from GaussSmoothing import GaussianSmoothing
from Net import UNet
import LossFunctions as lf
from Visualize import plotImageData

def smoothArray3D(inputArray, device):
    smoothing = GaussianSmoothing(1, 5, 2, 3, device)
    input = nn.functional.pad(inputArray, (2,2,2,2,2,2))
    input = input[None, None, ]
    input = smoothing(input)
    input = nn.functional.pad(input, (2,2,2,2,2,2))
    return smoothing(input)[0,0]

def testNet(net, device, dataloader, outputPath):
  net.to(device)
  net.eval()
  
  patchSize = 64
  
  with torch.no_grad():
    for i, data in enumerate(dataloader, 0):
      imgData = data['image']
      labelData = data['label']
      maskData = data['mask']
      
      imgShape = imgData.shape
      imgData = imgData.to(device)
      
      if (maskData.dim() != imgData.dim()):
        maskData = torch.ones(imgShape, dtype=torch.int8)

      maxIdxs = getMaxIdxs(imgShape, patchSize)
      patchSizes = getPatchSize(imgShape, patchSize)
      
      defFields = torch.zeros((imgShape[0], imgShape[1] * 3, imgShape[2], imgShape[3], imgShape[4]), device=device, requires_grad=False)
      indexArray = torch.zeros((imgShape[2], imgShape[3], imgShape[4]), device=device, requires_grad=False)
      for patchIdx0 in range(0, maxIdxs[0], patchSizes[0]):
        for patchIdx1 in range(0, maxIdxs[1], patchSizes[1]):
          for patchIdx2 in range(0, maxIdxs[2], patchSizes[2]):
            if (maskData[:, :, patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[1], patchIdx2:patchIdx2 + patchSizes[2]].sum() > 0):
              subImgData = imgData[:, :, patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[1], patchIdx2:patchIdx2 + patchSizes[2]]
              indexArray[patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[1], patchIdx2:patchIdx2 + patchSizes[2]] += 1
              defFields[:, :, patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[2], patchIdx2:patchIdx2 + patchSizes[2]] += net(subImgData)
         
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
                defFields[:, :, patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[1], patchIdx2:patchIdx2 + patchSizes[2]] += net(subImgData)    
      
      del maskData, subImgData
       
      indexArray[indexArray < 1] = 1
      indexArray = smoothArray3D(indexArray, device)
      
      for dim0 in range(0, defFields.shape[0]):
        for dim1 in range(0, defFields.shape[1]):
          defFields[dim0, dim1, ] /= indexArray

      itkIndexArray = sitk.GetImageFromArray(indexArray)
      dataloader.dataset.saveData(itkIndexArray, outputPath, 'indexArray' + str(i) + '.nrrd', i, False)

      del indexArray
      
      zeroDefField = getZeroDefField(imgShape)
      zeroDefField = zeroDefField.to(device)
      for imgIdx in range(imgShape[0]):
        for chanIdx in range(-1, imgShape[1] - 1):
          imgToDef = imgData[None, None, imgIdx, chanIdx, ]
          chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
          deformedTmp = deformImage(imgToDef, defFields[None, imgIdx, chanRange, ], device)
          
          imgDataDef = sitk.GetImageFromArray(deformedTmp[0, 0, ])
          
          dataloader.dataset.saveData(imgDataDef, outputPath, 'deformedImgDataset' + str(i) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', i)
          defX = defFields[imgIdx, chanIdx * 3, ].detach() * (imgToDef.shape[2] / 2)
          defY = defFields[imgIdx, chanIdx * 3 + 1, ].detach() * (imgToDef.shape[3] / 2)
          defZ = defFields[imgIdx, chanIdx * 3 + 2, ].detach() * (imgToDef.shape[4] / 2)
          defDataToSave = sitk.GetImageFromArray(getDefField(defX, defY, defZ), isVector=True)
          dataloader.dataset.saveData(defDataToSave, outputPath, 'deformationFieldDataset' + str(i) + 'image' + str(imgIdx) + 'channel' + str(chanIdx) + '.nrrd', i)


def deformImage(imgToDef, defFields, device):
  zeroDefField = getZeroDefField(imgToDef.shape)
  zeroDefField = zeroDefField.to(device)  
  currDefField = torch.empty(zeroDefField.shape, device=device, requires_grad=False)
  currDefField[..., 0] = zeroDefField[..., 0] + defFields[:, 0, ].detach()
  currDefField[..., 1] = zeroDefField[..., 1] + defFields[:, 1, ].detach()
  currDefField[..., 2] = zeroDefField[..., 2] + defFields[:, 2, ].detach()
  deformedTmp = torch.nn.functional.grid_sample(imgToDef, currDefField, mode='bilinear', padding_mode='border')
  return deformedTmp
  

def getMaxIdxs(imgShape, imgPatchSize):
  if imgShape[2] - imgPatchSize > 0:
    maxidx0 = imgShape[2] - imgPatchSize
  else:
    maxidx0 = 1
  
  if imgShape[3] - imgPatchSize > 0:
    maxidx1 = imgShape[3] - imgPatchSize
  else:
    maxidx1 =  1
  
  if imgShape[4] - imgPatchSize > 0:
    maxidx2 = imgShape[4] - imgPatchSize
  else:
    maxidx2 = 1
  return (maxidx0, maxidx1, maxidx2)

def getPatchSize(imgShape, imgPatchSize):
  if imgShape[2] - imgPatchSize > 0:
    patchSize0 = imgPatchSize
  else:
    patchSize0 = imgShape[2]
  
  if imgShape[3] - imgPatchSize > 0:
    patchSize1 = imgPatchSize
  else:
    patchSize1 = imgShape[3]
  
  if imgShape[4] - imgPatchSize > 0:
    patchSize2 = imgPatchSize
  else:
    patchSize2 = imgShape[4]  
    
  return (patchSize0, patchSize1, patchSize2)


indexArrayTest=[]

def getIndicesForRandomization(maskData, imgData, imgPatchSize):
  maxIdxs = getMaxIdxs(imgData.shape, imgPatchSize)
  print(maxIdxs)
  if (maskData.dim() == imgData.dim()):
    maskDataCrop = maskData[:, :, 0:maxIdxs[0], 0:maxIdxs[1], 0:maxIdxs[2]]
  else: 
    maskDataCrop = torch.ones((imgData.shape[0], imgData.shape[1], maxIdxs[0], maxIdxs[1], maxIdxs[2]), dtype=torch.int8)
  
  maskChanSum = torch.sum(maskDataCrop, 1)
  idxs = np.where(maskChanSum > 0)
  print(min(idxs[0]))
  print(min(idxs[1]))
  print(min(idxs[2]))
  print(min(idxs[3]))
  return idxs


def save_grad(name):

    def hook(grad):
        print(name)
        print(torch.sum(grad))

    return hook

  
def printGPUMemoryAllocated():
  torch.cuda.synchronize()
  print(torch.cuda.memory_allocated())

  
def getRandomSubSamples(numberofSamplesPerRun, idxs, patchSizes, imgData, labelData):
 
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
    
    global indexArrayTest
    indexArrayTest[idx2:idx2 + patchSizes[0], idx3:idx3 + patchSizes[1], idx4:idx4 + patchSizes[2]] += 1
    
    if (labelData.dim() == imgData.dim()):
      labelDataNew[j, ] = labelData[idx0, : , idx2:idx2 + patchSizes[0], idx3:idx3 + patchSizes[1], idx4:idx4 + patchSizes[2]]
  
  imgDataToWork = imgDataNew
  if (labelData.dim() == imgData.dim()):
    labelDataToWork = labelDataNew
  else:
    labelDataToWork = torch.Tensor();
    
  return (imgDataToWork, labelDataToWork)
  
def optimizeNet(net, imgDataToWork, labelToWork, device, optimizer, lossWeights):
  zeroDefField = getZeroDefField(imgDataToWork.shape)
  imgDataToWork = imgDataToWork.to(device)
  zeroDefField = zeroDefField.to(device)
  
  # zero the parameter gradients
  optimizer.zero_grad()
  
  # forward + backward + optimize
  defFields = net(imgDataToWork)
  
  imgDataDef = torch.empty(imgDataToWork.shape, device=device, requires_grad=False)
  
  cycleImgData = torch.empty(defFields.shape, device=device)
  
  #         cycleIdxData = torch.empty((imgData.shape[0:2]) + zeroDefField.shape[1:], device=device)
  cycleIdxData = zeroDefField.clone()
  
  for chanIdx in range(-1, imgDataToWork.shape[1] - 1):
    imgToDef = imgDataToWork[:, None, chanIdx, ]
    chanRange = range(chanIdx * 3, chanIdx * 3 + 3)
    deformedTmp = deformImage(imgToDef, defFields[: , chanRange, ], device)
    imgDataDef[:, chanIdx + 1, ] = deformedTmp[:, 0, ]
    
    cycleImgData[:, chanRange, ] = torch.nn.functional.grid_sample(defFields[:, chanRange, ], cycleIdxData.clone(), mode='bilinear', padding_mode='border')
                
    cycleIdxData[..., 0] = cycleIdxData[..., 0] + defFields[:, chanIdx * 3, ].detach()
    cycleIdxData[..., 1] = cycleIdxData[..., 1] + defFields[:, chanIdx * 3 + 1, ].detach()
    cycleIdxData[..., 2] = cycleIdxData[..., 2] + defFields[:, chanIdx * 3 + 2, ].detach()
  
  del zeroDefField, cycleIdxData
        
  crossCorr = lf.normCrossCorr(imgDataToWork, imgDataDef)
  if imgDataToWork.shape[1] > 3:
    smoothnessDF = lf.smoothnessVecFieldT(defFields, device)
  else:
    smoothnessDF = lf.smoothnessVecField(defFields, device)
  
  vecLengthLoss = torch.abs(defFields).mean()
  cycleLoss = lf.cycleLoss(cycleImgData, device)
  loss = lossWeights['ccW'] * crossCorr + lossWeights['smoothW'] * smoothnessDF + lossWeights['vecLengthW'] * vecLengthLoss + lossWeights['cycleW'] * cycleLoss
  print('cc: %.3f smmothness: %.3f vecLength: %.3f cycleLoss: %.3f' % (crossCorr, smoothnessDF, vecLengthLoss, cycleLoss))
  print('loss: %.3f' % (loss))
    
  loss.backward()
  optimizer.step()    
  
def trainNet(net, device, dataloader, epochs):
  net.to(device)
#   optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
  optimizer = optim.Adam(net.parameters())
  lossWeights = {'ccW' : 0.79, 'smoothW' : 0.1, 'vecLengthW' : 0.01, 'cycleW' : 0.1}
  
  imgPatchSize = 64
  maxNumberOfPixs = imgPatchSize * imgPatchSize * imgPatchSize * 10 + 1
  maxNumberOfSamples = 6  # samples for one batch must be < maxNumberOfSamples
  
  for epoch in range(epochs):  # loop over the dataset multiple times
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        imgData = data['image']
        labelData = data['label']
        maskData = data['mask']
        
        global indexArrayTest
        indexArrayTest = torch.zeros((imgData.shape[2], imgData.shape[3], imgData.shape[4]), device=device, requires_grad=False)
        
        numberofSamples = (torch.numel(imgData) / maxNumberOfPixs) + 1
        numberOfiterations = (numberofSamples / maxNumberOfSamples) + 1
        numberofSamplesPerRun = min(numberofSamples, maxNumberOfSamples - 1)
        
        
        if torch.numel(imgData) >= maxNumberOfPixs:
          doRandomSubSampling = True
          patchSizes = getPatchSize(imgData.shape, imgPatchSize)
          idxs = getIndicesForRandomization(maskData, imgData, imgPatchSize)
        else:
          doRandomSubSampling = False
          imgDataToWork = imgData
          labelDataToWork = labelData
        
        for imgIteration in range(0, numberOfiterations):
          if (doRandomSubSampling):
            randomSubSamples = getRandomSubSamples(numberofSamplesPerRun, idxs, patchSizes, imgData, labelData)
            imgDataToWork = randomSubSamples[0]
            labelDataToWork = randomSubSamples[1]
          
          optimizeNet(net, imgDataToWork, labelDataToWork, device, optimizer, lossWeights)
  
  itkIndexArray = sitk.GetImageFromArray(indexArrayTest)
  dataloader.dataset.saveData(itkIndexArray, 'IdxTmp', 'indexArray' + str(i) + '.nrrd', i, False)          
  print('Finished Training') 


def main(argv):
  
  torch.backends.cudnn.enabled = False
  CUDA_LAUNCH_BLOCKING = 1
  callString = 'FirstNet.py --trainingFiles=files.csv --valdiationFiles=files.csv --device=device --numberOfEpochs=500 --outputPath=PATH --testMode --trainMode'
  
  try:
    opts, args = getopt.getopt(argv, '', ['trainingFiles=', 'valdiationFiles=', 'device=', 'numberOfEpochs=', 'outputPath=', 'testMode', 'trainMode'])
  except getopt.GetoptError:
    print(callString)
    return
    
  if not (len(opts)):
    print(callString)
    return

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  numberOfEpochs = 500
  testMode = False
  trainMode = False
  
  outputPath = 'RegResults'  
  for opt, arg in opts:
    if opt == '--trainingFiles':
      trainingFileNamesCSV = arg
    elif opt == '--valdiationFiles':
      validationFileNamesCSV = arg
    elif opt == '--device':
      device = arg      
    elif opt == '--outputPath':
      outputPath = arg      
    elif opt == '--numberOfEpochs':
      numberOfEpochs = int(arg)
    elif opt == '--testMode':
      testMode = True
    elif opt == '--trainMode':
      trainMode = True          
      
  torch.manual_seed(0)
  np.random.seed(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  headAndNeckTrainSet = HeadAndNeckDataset(trainingFileNamesCSV, ToTensor())
  
  dataloader = DataLoader(headAndNeckTrainSet, batch_size=1,
                        shuffle=False, num_workers=0)
  
  net = UNet(headAndNeckTrainSet.getChannels(), True, False, 3)
  
  if not os.path.isdir(outputPath):
    os.makedirs(outputPath)
  modelFileName = outputPath + os.path.sep + 'UNetfinalParams.pt'
  if trainMode:
    start = time.time()
    if False:  # device == "cpu":
      net.share_memory()
      processes = []
      num_processes = 2
      for rank in range(num_processes):
        p = mp.Process(target=trainNet, args=(net, device, dataloader, numberOfEpochs))
        p.start()
        processes.append(p)
      for p in processes:
        p.join()
          
    else:
      trainNet(net, device, dataloader, numberOfEpochs)
    end = time.time()
    print(end - start)
    torch.save(net.state_dict(), modelFileName)
    
  if testMode:
    net.load_state_dict(torch.load(modelFileName))
    testNet(net, device, dataloader, outputPath)

if __name__ == "__main__":
  main(sys.argv[1:])  
