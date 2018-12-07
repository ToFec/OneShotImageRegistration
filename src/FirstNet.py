import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys, getopt
import matplotlib.pyplot as plt
import numpy as np
from Utils import deform, getDefField

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
      
      defFields = net(imgData)
      for imgIdx in range(imgData.shape[0]):
        for chanIdx in range(-1,imgData.shape[1]-1):
          imgToDef = imgData[imgIdx, chanIdx,]
          defX = defFields[imgIdx, chanIdx * 3,].detach()
          defY = defFields[imgIdx, chanIdx * 3 + 1,].detach()
          defZ = defFields[imgIdx, chanIdx * 3 + 2,].detach()
          imgDataDef = deform(imgToDef, defX, defY, defZ)
          saveData(imgDataDef, 'deformedImgDataset' + str(i) + 'image' + str(imgIdx)+ 'channel' + str(chanIdx) + '.nii.gz')
          saveData(getDefField(defX, defY, defZ), 'deformdationFieldDataset' + str(i) + 'image' + str(imgIdx)+ 'channel' + str(chanIdx) + '.nii.gz')
  
def trainNet(net, device, dataloader):
  net.to(device)
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  lambda0 = 1
  lambda1 = 1
  for epoch in range(20):  # loop over the dataset multiple times
    start = time.time()
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        imgData = data['image']
        labelData = data['label']
        maskData = data['mask']
        imgData = imgData.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        defFields = net(imgData)
        imgDataDef = torch.zeros(imgData.shape, requires_grad=False)
        for imgIdx in range(imgData.shape[0]):
          for chanIdx in range(-1,imgData.shape[1]-1):
            imgToDef = imgData[imgIdx, chanIdx,]
            defX = defFields[imgIdx, chanIdx * 3,].detach()
            defY = defFields[imgIdx, chanIdx * 3 + 1,].detach()
            defZ = defFields[imgIdx, chanIdx * 3 + 2,].detach()
            imgDataDef[imgIdx, chanIdx+1,] = torch.from_numpy(deform(imgToDef, defX, defY, defZ))
        imgDataDef.requires_grad = True
        crossCorr = lf.normCrossCorr(imgData, imgDataDef)
        smoothnessDF = lf.smoothnessVecField(defFields)
        loss = lambda0 * crossCorr + lambda1 * smoothnessDF
        print('cc: %.3f smmothness: %.3f' % (crossCorr, smoothnessDF))
        print('loss: %.3f' % (loss))
        loss.backward()
        optimizer.step()

    end = time.time()
    print(end - start)
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
    opts, args = getopt.getopt(argv,'',['trainingFiles=', 'valdiationFiles='])
  except getopt.GetoptError:
    print('FirstNet.py --trainingFiles=files.csv --valdiationFiles=files.csv')
    return
    
  if not (len(opts)):
    print('FirstNet.py --trainingFiles=files.csv --valdiationFiles=files.csv')
    return
    
  for opt, arg in opts:
    if opt == '--trainingFiles':
      trainingFileNamesCSV = arg
    elif opt == '--valdiationFiles':
      validationFileNamesCSV = arg
      
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  device = "cpu" ## for testing 
  print(device)
  
  headAndNeckTrainSet = HeadAndNeckDataset(trainingFileNamesCSV,ToTensor())
#   plotDataset(headAndNeckTrainSet)
  
  dataloader = DataLoader(headAndNeckTrainSet, batch_size=1,
                        shuffle=False, num_workers=0)
  
#   for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].shape)

  net = UNet(2, True, True, 2)
  print(net)
  trainNet(net, device, dataloader)
  testNet(net, device, dataloader)

if __name__ == "__main__":
  main(sys.argv[1:])  
