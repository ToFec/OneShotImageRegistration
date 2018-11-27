import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import sys, getopt
import unicodecsv as csv
import matplotlib.pyplot as plt
import numpy as np

import nibabel as nib
import HeadAndNeckDataset
import DummyNet

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    

  
def trainNet(net, device, dataloader):
  net.to(device)
  net.zero_grad()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  
  criterion = nn.CrossEntropyLoss()
  
 
  for epoch in range(2):  # loop over the dataset multiple times
  
      running_loss = 0.0
      for i, data in enumerate(dataloader, 0):
          # get the inputs
          inputs, labels = data
          inputs, labels = inputs.to(device), labels.to(device)
          
          # zero the parameter gradients
          optimizer.zero_grad()
  
          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
  
          # print statistics
          running_loss += loss.item()
          if i % 2000 == 1999:    # print every 2000 mini-batches
              print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
              running_loss = 0.0
  
  print('Finished Training') 

def plotDataset(dataset):
  nuOfimg=min(4,len(dataset))
  nuOfImgPerAxes=max(1,round(nuOfimg/2,0))
  for i in range(0, nuOfimg):
      plt.subplot(nuOfImgPerAxes,nuOfImgPerAxes,i+1)
      sample = dataset[i]
      plt.imshow(sample['image'][90,:,:],cmap='gray')
      plt.imshow(sample['label'][90,:,:],cmap='jet', alpha=0.5)
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
  print(device)
  
  headAndNeckTrainSet = HeadAndNeckDataset.HeadAndNeckDataset(trainingFileNamesCSV,HeadAndNeckDataset.ToTensor())
  
  dataloader = DataLoader(headAndNeckTrainSet, batch_size=1,
                        shuffle=False, num_workers=1)
  

  net = DummyNet.DummyNet()
  trainNet(net, device, dataloader)

if __name__ == "__main__":
  main(sys.argv[1:])  
