import os.path
import unicodecsv as csv
from torch.utils.data import Dataset
from medpy.io import load, save
import torch
import numpy as np

class HeadAndNeckDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None, loadOnInstantiation=True):

        self.transform = transform
        self.loadOnInstantiation = loadOnInstantiation
        csvtrainingFiles =  open(csv_file, 'rb')
        try:        
          trianingCSVFileReader = csv.reader(csvtrainingFiles, delimiter=';', encoding='iso8859_15')
          if (self.loadOnInstantiation):
            self.loadedIgSamples = {}
          else:
            self.dataFileList = []
            self.labelFileList = []
            self.maskFileList = []
          
          idx = 0
          for trainingFilePath in trianingCSVFileReader:
            imgFiles = []
            maskFiles = []
            labelFiles = []
            i = 0
            while (True):
              trainingFileName = trainingFilePath[0] + '/img' + str(i) + '.nii.gz'
              if (os.path.isfile(trainingFileName)):
                imgFiles.append(trainingFileName)
              else:
                break
                
              maskFileName = trainingFilePath[0] + '/mask' + str(i) + '.nii.gz'
              if (os.path.isfile(maskFileName)):
                maskFiles.append(maskFileName)
              labelsFileName = trainingFilePath[0] + '/struct' + str(i) + '.nii.gz'  
              if (os.path.isfile(labelsFileName)):
                labelFiles.append(labelsFileName)
              i=i+1
            
            if (self.loadOnInstantiation):
              sample = self.loadData(imgFiles, maskFiles, labelFiles)
              self.loadedIgSamples[idx] = sample
              idx = idx + 1
            else:
              self.dataFileList.append(imgFiles)
              self.labelFileList.append(labelFiles)
              self.maskFileList.append(maskFiles)
          
          if (self.loadOnInstantiation):
            self.length = len(self.loadedIgSamples)
          else:
            self.length = len(self.dataFileList)
        finally:
          csvtrainingFiles.close()
          

    def __len__(self):
        return self.length
    
    def loadData(self, trainingFileNames, maskFileNames, labelsFileNames):
      imgData = []
      for trainingFileName in trainingFileNames:
        imgNii, imgHeader = load(trainingFileName)
        imgData.append(imgNii)
      imgData = np.stack(imgData)
      imgData = imgData - imgData.mean()
      imgData = imgData / imgData.std()
        
      maskData = []
      if (len(trainingFileNames) == len(maskFileNames)):
        for maskFileName in maskFileNames:
          maskNii, maskHeader = load(maskFileName)
          maskData.append(maskNii)
        maskData = np.stack(maskData)
      
      labelData = []
      if (len(trainingFileNames) == len(labelsFileNames)):
        for labelsFileName in labelsFileNames:
          labelsNii, labelsHeader = load(labelsFileName)
          labelData.append(labelsNii)
        labelData = np.stack(labelData)

      sample = {'image': imgData, 'label': labelData, 'mask': maskData}
      if self.transform:
        sample = self.transform(sample)
      return sample  
    
    def __getitem__(self, idx):
        
        ##works currently only for single thread
        if self.loadOnInstantiation:
          sample = self.loadedIgSamples[idx]
        else:
          trainingFileNames = self.dataFileList[idx]
          maskFileNames = self.maskFileList[idx]
          labelsFileNames = self.labelFileList[idx]
          sample = self.loadData(trainingFileNames, maskFileNames, labelsFileNames)
          if self.transform:
            sample = self.transform(sample)

        return sample

def saveData(data, filename):
  save(data, filename)
      
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, mask = sample['image'], sample['label'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
#         image = image.transpose((2, 0, 1))
#         label = label.transpose((2, 0, 1))
#         mask = mask.transpose((2, 0, 1))
        labelTorch = torch.tensor([1])
        if(len(label) > 0):
          labelTorch = torch.from_numpy(label)
          
        maskTorch = torch.tensor([1])
        if(len(mask) > 0):
          maskTorch = torch.from_numpy(mask)
          
          
        return {'image': torch.from_numpy(image),
                'label': labelTorch,
                'mask': maskTorch}      
        