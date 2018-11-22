import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import sys, getopt
import unicodecsv as csv
import matplotlib.pyplot as plt
import numpy as np

import nibabel as nib
import HeadAndNeckDataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    

  
def trainNet(net):
  net.zero_grad()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  
  criterion = nn.CrossEntropyLoss()
  
  transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)
  
  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                           shuffle=False, num_workers=2)
  
  classes = ('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  
  for epoch in range(2):  # loop over the dataset multiple times
  
      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
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


def main(argv):
  try:
    opts, args = getopt.getopt(argv,'',['trainingFiles=', 'valdiationFiles='])
  except getopt.GetoptError:
    print 'FirstNet.py --trainingFiles=files.csv --valdiationFiles=files.csv'
    return
    
  if not (len(opts)):
    print 'FirstNet.py --trainingFiles=files.csv --valdiationFiles=files.csv'
    return
    
  for opt, arg in opts:
    if opt == '--trainingFiles':
      trainingFileNamesCSV = arg
    elif opt == '--valdiationFiles':
      validationFileNamesCSV = arg
      
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  headAndNeckTrainSet = HeadAndNeckDataset.HeadAndNeckDataset(trainingFileNamesCSV)

  for i in range(len(headAndNeckTrainSet)):
    sample = headAndNeckTrainSet[i]

  net = Net()
  #net.to(device)
  #trainNet(net)

if __name__ == "__main__":
  main(sys.argv[1:])  
