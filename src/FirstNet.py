import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import sys, getopt
import matplotlib.pyplot as plt
import numpy as np
from Utils import deform, getDefField, getZeroDefField
import SimpleITK as sitk

import time

from HeadAndNeckDataset import HeadAndNeckDataset, ToTensor, saveData
from Net import UNet
import LossFunctions as lf

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
def testNet(net, device, dataloader):
  net.to(device)
  net.eval()
  with torch.no_grad():
    for i, data in enumerate(dataloader, 0):
      imgData = data['image']
      labelData = data['label']
      maskData = data['mask']
      imgData = imgData.to(device)
      zeroDefField = getZeroDefField(imgData)
      zeroDefField = zeroDefField.to(device)
      
      defFields = net(imgData)
      for imgIdx in range(imgData.shape[0]):
        for chanIdx in range(-1,imgData.shape[1]-1):
          imgToDef = imgData[None, None, imgIdx, chanIdx,]
          currDefField = torch.empty(zeroDefField.shape, device=device, requires_grad=False)
          currDefField[imgIdx,...,0] = zeroDefField[imgIdx,...,0] +  defFields[imgIdx, chanIdx * 3,].detach()
          currDefField[imgIdx,...,1] = zeroDefField[imgIdx,...,1] +  defFields[imgIdx, chanIdx * 3 + 1,].detach()
          currDefField[imgIdx,...,2] = zeroDefField[imgIdx,...,2] +  defFields[imgIdx, chanIdx * 3 + 2,].detach()
          deformedTmp = torch.nn.functional.grid_sample(imgToDef, currDefField, mode='bilinear', padding_mode='border')

          imgDataDef =  sitk.GetImageFromArray(deformedTmp[0,0,])
          
          saveData(imgDataDef, 'deformedImgDataset' + str(i) + 'image' + str(imgIdx)+ 'channel' + str(chanIdx) + '.nrrd')
          defX = defFields[imgIdx, chanIdx * 3,].detach() * (imgToDef.shape[2]/2)
          defY = defFields[imgIdx, chanIdx * 3 + 1,].detach() * (imgToDef.shape[3]/2)
          defZ = defFields[imgIdx, chanIdx * 3 + 2,].detach() * (imgToDef.shape[4]/2)
          defDataToSave = sitk.GetImageFromArray(getDefField(defX, defY, defZ),isVector=True)
          saveData(defDataToSave, 'deformationFieldDataset' + str(i) + 'image' + str(imgIdx)+ 'channel' + str(chanIdx) + '.nrrd')

def save_grad(name):
    def hook(grad):
        print(name)
        print(torch.sum(grad))
    return hook
  
def trainNet(net, device, dataloader, epochs):
  net.to(device)
#   optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
  optimizer = optim.Adam(net.parameters())
  lambda0 = 1
  lambda1 = 0
  lambda2 = 0
  lambda3 = 1
  
  for epoch in range(epochs):  # loop over the dataset multiple times
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        imgData = data['image']
        zeroDefField = getZeroDefField(imgData)
        labelData = data['label']
        maskData = data['mask']
        imgData = imgData.to(device)
        zeroDefField = zeroDefField.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        defFields = net(imgData)
        imgDataDef = torch.empty(imgData.shape, device=device, requires_grad=False)
        cycleIdx = torch.empty(defFields.shape, device=device)
        
        oldIdx = zeroDefField.clone()
        
        for imgIdx in range(imgData.shape[0]):
          for chanIdx in range(-1,imgData.shape[1]-1):
            imgToDef = imgData[None, None, imgIdx, chanIdx,]
            currDefField = torch.empty(zeroDefField.shape, device=device, requires_grad=False)
            currDefField[imgIdx,...,0] = zeroDefField[imgIdx,...,0] +  defFields[imgIdx, chanIdx * 3,]
            currDefField[imgIdx,...,1] = zeroDefField[imgIdx,...,1] +  defFields[imgIdx, chanIdx * 3 + 1,]
            currDefField[imgIdx,...,2] = zeroDefField[imgIdx,...,2] +  defFields[imgIdx, chanIdx * 3 + 2,]
            deformedTmp = torch.nn.functional.grid_sample(imgToDef, currDefField, mode='bilinear', padding_mode='border')
            imgDataDef[imgIdx, chanIdx+1,] = deformedTmp[0,0,]
            
            tmp0 = torch.nn.functional.grid_sample(defFields[None, None, imgIdx, chanIdx * 3,], oldIdx[None, imgIdx,], mode='bilinear', padding_mode='border')
            cycleIdx[imgIdx, chanIdx * 3,] = tmp0[0,0,]
            tmp1 = torch.nn.functional.grid_sample(defFields[None, None, imgIdx, chanIdx * 3 + 1,], oldIdx[None, imgIdx,], mode='bilinear', padding_mode='border')
            cycleIdx[imgIdx, chanIdx * 3 + 1,] = tmp1[0,0,]
            tmp2 = torch.nn.functional.grid_sample(defFields[None, None, imgIdx, chanIdx * 3 + 2,], oldIdx[None, imgIdx,], mode='bilinear', padding_mode='border')
            cycleIdx[imgIdx, chanIdx * 3 + 2,] = tmp2[0,0,]
            
            oldIdx[imgIdx,...,0] = oldIdx[imgIdx,...,0] + defFields[imgIdx, chanIdx * 3,].detach()
            oldIdx[imgIdx,...,1] = oldIdx[imgIdx,...,1] + defFields[imgIdx, chanIdx * 3 + 1,].detach()
            oldIdx[imgIdx,...,2] = oldIdx[imgIdx,...,2] + defFields[imgIdx, chanIdx * 3 + 2,].detach()
              
              
        crossCorr = lf.normCrossCorr(imgData, imgDataDef)
        if imgData.shape[1] > 3:
          smoothnessDF = lf.smoothnessVecFieldT(defFields, device)
        else:
          smoothnessDF = lf.smoothnessVecField(defFields, device)
        
        vecLengthLoss = torch.abs(defFields).mean()
        cycleLoss = lf.cycleLoss(cycleIdx, device)
        loss = lambda0 * crossCorr + lambda1 * smoothnessDF + lambda2 * vecLengthLoss + lambda3 * cycleLoss
        print('cc: %.3f smmothness: %.3f vecLength: %.3f cycleLoss: %.3f' % (crossCorr, smoothnessDF, vecLengthLoss, cycleLoss))
        print('loss: %.3f' % (loss))
          
        loss.backward()
        optimizer.step()

  print('Finished Training') 

def plotDataset(dataset):
  nuOfimg=min(4,len(dataset))
  nuOfImgPerAxes=max(1,round(nuOfimg/2,0))
  for i in range(0, nuOfimg):
      plt.subplot(nuOfImgPerAxes,nuOfImgPerAxes,i+1)
      sample = dataset[i]
      if (sample['image'].dim() == 4):
        slice = int(sample['image'].shape[3] / 2)
        plt.imshow(sample['image'][0,:,:,slice],cmap='gray')
        if (sample['label'].dim() > 1):
          plt.imshow(sample['label'][0,:,:,slice],cmap='jet', alpha=0.5)
      elif (sample['image'].dim() == 3):
        slice = int(sample['image'].shape[2] / 2)
        plt.imshow(sample['image'][:,:,slice],cmap='gray')
        if (sample['label'].dim() > 1):
          plt.imshow(sample['label'][:,:,slice],cmap='jet', alpha=0.5)
  plt.show()

def main(argv):
  try:
    opts, args = getopt.getopt(argv,'',['trainingFiles=', 'valdiationFiles=', 'device=', 'numberOfEpochs='])
  except getopt.GetoptError:
    print('FirstNet.py --trainingFiles=files.csv --valdiationFiles=files.csv --device=device --numberOfEpochs=500')
    return
    
  if not (len(opts)):
    print('FirstNet.py --trainingFiles=files.csv --valdiationFiles=files.csv --device=device --numberOfEpochs=500')
    return

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  numberOfEpochs = 500
    
  for opt, arg in opts:
    if opt == '--trainingFiles':
      trainingFileNamesCSV = arg
    elif opt == '--valdiationFiles':
      validationFileNamesCSV = arg
    elif opt == '--device':
      device = arg      
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
  
  
  net = UNet(2, True, True, 3)
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
  testNet(net, device, dataloader)

if __name__ == "__main__":
  main(sys.argv[1:])  
