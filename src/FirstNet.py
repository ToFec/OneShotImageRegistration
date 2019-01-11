import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import sys, getopt
import numpy as np
from Utils import deform, getDefField, getZeroDefField
import SimpleITK as sitk

import time

from HeadAndNeckDataset import HeadAndNeckDataset, ToTensor
from Net import UNet
import LossFunctions as lf
from Visualize import plotImageData

def testNet(net, device, dataloader, outputPath):
  net.to(device)
  net.eval()
  
  patchSize = 64
  
  with torch.no_grad():
    for i, data in enumerate(dataloader, 0):
      imgData = data['image']
      labelData = data['label']
      maskData = data['mask']
      imgData = imgData.to(device)
      
      if (maskData.dim() != imgData.dim()):
        maskData = torch.ones(imgData.shape,dtype=torch.int8)

      maxidx0 = imgData.shape[2]-patchSize if imgData.shape[2]-patchSize > 0 else imgData.shape[2]
      maxidx1 = imgData.shape[3]-patchSize if imgData.shape[3]-patchSize > 0 else imgData.shape[3]
      maxidx2 = imgData.shape[4]-patchSize if imgData.shape[4]-patchSize > 0 else imgData.shape[4]
      patchSize0 = maxidx0 if maxidx0 < patchSize else patchSize
      patchSize1 = maxidx1 if maxidx1 < patchSize else patchSize
      patchSize2 = maxidx2 if maxidx2 < patchSize else patchSize
      
      defFields = torch.zeros((imgData.shape[0], imgData.shape[1] * 3, imgData.shape[2], imgData.shape[3], imgData.shape[4]),device=device, requires_grad=False)
      indexArray = torch.zeros((imgData.shape[2], imgData.shape[3], imgData.shape[4]),device=device, requires_grad=False)
      for patchIdx0 in range(0,maxidx0,patchSize0):
        for patchIdx1 in range(0,maxidx1,patchSize1):
          for patchIdx2 in range(0,maxidx2,patchSize2):
            if (maskData[:,:,patchIdx0, patchIdx1, patchIdx2].sum() > 0):
              subImgData = imgData[:,:,patchIdx0:patchIdx0+patchSize0, patchIdx1:patchIdx1+patchSize1, patchIdx2:patchIdx2+patchSize2]
              indexArray[patchIdx0:patchIdx0+patchSize0, patchIdx1:patchIdx1+patchSize1, patchIdx2:patchIdx2+patchSize2] += 1
              defFields[:,:,patchIdx0:patchIdx0+patchSize0, patchIdx1:patchIdx1+patchSize1, patchIdx2:patchIdx2+patchSize2] += net(subImgData)
         
      leftover0 = imgData.shape[2] % patchSize0
      startidx0 = patchSize0/2 if (leftover0 > 0) & (maxidx0 > patchSize0)  else leftover0
      leftover1 = imgData.shape[3] % patchSize1
      startidx1 = patchSize1/2 if (leftover1 > 0) & (maxidx1 > patchSize1)  else leftover1
      leftover2 = imgData.shape[4] % patchSize2
      startidx2 = patchSize2/2 if (leftover2 > 0) & (maxidx2 > patchSize2)  else leftover2
            
      if (startidx2+startidx1+startidx0 > 0) :               
        for patchIdx0 in range(startidx0,maxidx0,patchSize0):
          for patchIdx1 in range(startidx1,maxidx1,patchSize1):
            for patchIdx2 in range(startidx2,maxidx2,patchSize2):
              if (maskData[:,:,patchIdx0, patchIdx1, patchIdx2].sum() > 0):
                subImgData = imgData[:,:,patchIdx0:patchIdx0+patchSize0, patchIdx1:patchIdx1+patchSize1, patchIdx2:patchIdx2+patchSize2]
                indexArray[patchIdx0:patchIdx0+patchSize0, patchIdx1:patchIdx1+patchSize1, patchIdx2:patchIdx2+patchSize2] += 1
                defFields[:,:,patchIdx0:patchIdx0+patchSize0, patchIdx1:patchIdx1+patchSize1, patchIdx2:patchIdx2+patchSize2] += net(subImgData)    
        
      indicesToDivide = indexArray > 0
      defFields[:,:,indicesToDivide] /= indexArray[indicesToDivide]
      
      
      zeroDefField = getZeroDefField(imgData)
      zeroDefField = zeroDefField.to(device)
      for imgIdx in range(imgData.shape[0]):
        for chanIdx in range(-1,imgData.shape[1]-1):
          imgToDef = imgData[None, None, imgIdx, chanIdx,]
          currDefField = torch.empty(zeroDefField.shape, device=device, requires_grad=False)
          currDefField[imgIdx,...,0] = zeroDefField[imgIdx,...,0] +  defFields[imgIdx, chanIdx * 3,].detach()
          currDefField[imgIdx,...,1] = zeroDefField[imgIdx,...,1] +  defFields[imgIdx, chanIdx * 3 + 1,].detach()
          currDefField[imgIdx,...,2] = zeroDefField[imgIdx,...,2] +  defFields[imgIdx, chanIdx * 3 + 2,].detach()
          deformedTmp = torch.nn.functional.grid_sample(imgToDef, currDefField, mode='bilinear', padding_mode='border')

          imgDataDef =  sitk.GetImageFromArray(deformedTmp[0,0,])
          
          dataloader.dataset.saveData(imgDataDef, outputPath, 'deformedImgDataset' + str(i) + 'image' + str(imgIdx)+ 'channel' + str(chanIdx) + '.nrrd', i)
          defX = defFields[imgIdx, chanIdx * 3,].detach() * (imgToDef.shape[2]/2)
          defY = defFields[imgIdx, chanIdx * 3 + 1,].detach() * (imgToDef.shape[3]/2)
          defZ = defFields[imgIdx, chanIdx * 3 + 2,].detach() * (imgToDef.shape[4]/2)
          defDataToSave = sitk.GetImageFromArray(getDefField(defX, defY, defZ),isVector=True)
          dataloader.dataset.saveData(defDataToSave, outputPath, 'deformationFieldDataset' + str(i) + 'image' + str(imgIdx)+ 'channel' + str(chanIdx) + '.nrrd')

def save_grad(name):
    def hook(grad):
        print(name)
        print(torch.sum(grad))
    return hook
  
def printGPUMemoryAllocated():
  torch.cuda.synchronize()
  print(torch.cuda.memory_allocated())
  
def trainNet(net, device, dataloader, epochs):
  net.to(device)
#   optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
  optimizer = optim.Adam(net.parameters())
  ccW = 0.79
  smoothW = 0.1
  vecLengthW = 0.01
  cycleW = 0.1
  imgPatchSize = 64
  maxNumberOfPixs = imgPatchSize*imgPatchSize*imgPatchSize*10 + 1
  maxNumberOfSamples = 3 #samples for one batch must be < maxNumberOfSamples
  
  for epoch in range(epochs):  # loop over the dataset multiple times
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        imgData = data['image']
        labelData = data['label']
        maskData = data['mask']
        
        #in case of large images, split them
        numberofSamples = (torch.numel(imgData) / maxNumberOfPixs) + 1
        numberOfiterations = (numberofSamples / maxNumberOfSamples) + 1
        numberofSamplesPerRun = min(numberofSamples, maxNumberOfSamples - 1)
        for imgIteration in range(0,numberOfiterations):
          if (torch.numel(imgData) >= maxNumberOfPixs):
            maxidx0 = imgData.shape[2]-imgPatchSize if imgData.shape[2]-imgPatchSize > 0 else imgData.shape[2]
            maxidx1 = imgData.shape[3]-imgPatchSize if imgData.shape[3]-imgPatchSize > 0 else imgData.shape[3]
            maxidx2 = imgData.shape[4]-imgPatchSize if imgData.shape[4]-imgPatchSize > 0 else imgData.shape[4]
            if (maskData.dim() == imgData.dim()):
              maskDataCrop = maskData[:,:,0:maxidx0,0:maxidx1,0:maxidx2]
            else: 
              maskDataCrop = torch.ones((imgData.shape[0], imgData.shape[1], maxidx0, maxidx1, maxidx2),dtype=torch.int8)
            maskChanSum = torch.sum(maskDataCrop,1)
            idxs = np.where(maskChanSum > 0)
            
            randSampleIdxs = np.random.randint(0,len(idxs[0]),(numberofSamplesPerRun,))
            imgDataNew = torch.empty((numberofSamplesPerRun,imgData.shape[1],imgPatchSize,imgPatchSize,imgPatchSize),requires_grad=False)
            if (labelData.dim() == imgData.dim()):
              labelDataNew = torch.empty((numberofSamplesPerRun,imgData.shape[1],imgPatchSize,imgPatchSize,imgPatchSize),requires_grad=False)
            for j in range(0,numberofSamplesPerRun):
              idx0 = idxs[0][randSampleIdxs[j]]
              idx2 = idxs[1][randSampleIdxs[j]]
              idx3 = idxs[2][randSampleIdxs[j]]
              idx4 = idxs[3][randSampleIdxs[j]]
              imgDataNew[j,] = imgData[idx0, : , idx2:idx2+imgPatchSize, idx3:idx3+imgPatchSize,idx4:idx4+imgPatchSize]
              if (labelData.dim() == imgData.dim()):
                labelDataNew[j,] = labelData[idx0, : , idx2:idx2+imgPatchSize, idx3:idx3+imgPatchSize,idx4:idx4+imgPatchSize]
            
            imgDataToWork = imgDataNew
            if (labelData.dim() == imgData.dim()):
              labelDataToWork = labelDataNew
          else:
            imgDataToWork = imgData
            if (labelData.dim() == imgData.dim()):
              labelDataToWork = labelData
            
          zeroDefField = getZeroDefField(imgDataToWork)
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
          
          for chanIdx in range(-1,imgDataToWork.shape[1]-1):
            imgToDef = imgDataToWork[:, None, chanIdx,]
            currDefField = torch.empty(zeroDefField.shape, device=device, requires_grad=False)
            currDefField[...,0] = zeroDefField[...,0] +  defFields[:, chanIdx * 3,]
            currDefField[...,1] = zeroDefField[...,1] +  defFields[:, chanIdx * 3 + 1,]
            currDefField[...,2] = zeroDefField[...,2] +  defFields[:, chanIdx * 3 + 2,]
            
            deformedTmp = torch.nn.functional.grid_sample(imgToDef, currDefField, mode='bilinear', padding_mode='border')
            imgDataDef[:, chanIdx+1,] = deformedTmp[:,0,]
            
            chanRange = range(chanIdx * 3,chanIdx * 3 +3)
            
            cycleImgData[:, chanRange,] = torch.nn.functional.grid_sample(defFields[:, chanRange,], cycleIdxData.clone(), mode='bilinear', padding_mode='border')
                        
            cycleIdxData[...,0] = cycleIdxData[...,0] + defFields[:, chanIdx * 3,].detach()
            cycleIdxData[...,1] = cycleIdxData[...,1] + defFields[:, chanIdx * 3 + 1,].detach()
            cycleIdxData[...,2] = cycleIdxData[...,2] + defFields[:, chanIdx * 3 + 2,].detach()
                
                
          crossCorr = lf.normCrossCorr(imgDataToWork, imgDataDef)
          if imgDataToWork.shape[1] > 3:
            smoothnessDF = lf.smoothnessVecFieldT(defFields, device)
          else:
            smoothnessDF = lf.smoothnessVecField(defFields, device)
          
          vecLengthLoss = torch.abs(defFields).mean()
          cycleLoss = lf.cycleLoss(cycleImgData, device)
          loss = ccW * crossCorr + smoothW * smoothnessDF + vecLengthW * vecLengthLoss + cycleW * cycleLoss
          print('cc: %.3f smmothness: %.3f vecLength: %.3f cycleLoss: %.3f' % (crossCorr, smoothnessDF, vecLengthLoss, cycleLoss))
          print('loss: %.3f' % (loss))
            
          loss.backward()
          optimizer.step()

  print('Finished Training') 

def main(argv):
  
  torch.backends.cudnn.enabled = False
  CUDA_LAUNCH_BLOCKING=1
  
  try:
    opts, args = getopt.getopt(argv,'',['trainingFiles=', 'valdiationFiles=', 'device=', 'numberOfEpochs=', 'outputPath='])
  except getopt.GetoptError:
    print('FirstNet.py --trainingFiles=files.csv --valdiationFiles=files.csv --device=device --numberOfEpochs=500 --outputPath=PATH')
    return
    
  if not (len(opts)):
    print('FirstNet.py --trainingFiles=files.csv --valdiationFiles=files.csv --device=device --numberOfEpochs=500 --outputPath=PATH')
    return

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  numberOfEpochs = 500
  
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
      
  torch.manual_seed(0)
  np.random.seed(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  

  print(device)
  
  
  headAndNeckTrainSet = HeadAndNeckDataset(trainingFileNamesCSV,ToTensor())
  
  dataloader = DataLoader(headAndNeckTrainSet, batch_size=1,
                        shuffle=False, num_workers=0)
  
  
  
  
  net = UNet(headAndNeckTrainSet.getChannels(), True, False, 3)
  print(net)

  start = time.time()
  if False: #device == "cpu":
    net.share_memory()
    processes = []
    num_processes=2
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
  testNet(net, device, dataloader,outputPath)

if __name__ == "__main__":
  main(sys.argv[1:])  
