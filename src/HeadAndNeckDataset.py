import os
import unicodecsv as csv
from torch.utils.data import Dataset
import SimpleITK as sitk
import torch
import numpy as np

class HeadAndNeckDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None, loadOnInstantiation=True):

        self.transform = transform
        self.loadOnInstantiation = loadOnInstantiation
        csvtrainingFiles =  open(csv_file, 'rb')
        self.meansAndStds = {}
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
              trainingFileName = trainingFilePath[0] + os.path.sep +'img' + str(i) + '.nii.gz'
              if (os.path.isfile(trainingFileName)):
                imgFiles.append(trainingFileName)
              else:
                trainingFileName = trainingFilePath[0] + os.path.sep +'img' + str(i) + '.nrrd'
                if (os.path.isfile(trainingFileName)):
                  imgFiles.append(trainingFileName)
                else:
                  break
                
              maskFileName = trainingFilePath[0] + os.path.sep + 'mask' + str(i) + '.nii.gz'
              if (os.path.isfile(maskFileName)):
                maskFiles.append(maskFileName)
              maskFileName = trainingFilePath[0] + os.path.sep + 'mask' + str(i) + '.nrrd'
              if (os.path.isfile(maskFileName)):
                maskFiles.append(maskFileName)
              
              labelsFileName = trainingFilePath[0] + os.path.sep + 'struct' + str(i) + '.nii.gz'  
              if (os.path.isfile(labelsFileName)):
                labelFiles.append(labelsFileName)
              labelsFileName = trainingFilePath[0] + os.path.sep + 'struct' + str(i) + '.nrrd'  
              if (os.path.isfile(labelsFileName)):
                labelFiles.append(labelsFileName)
              i=i+1
            
            self.channels = len(imgFiles)
            if (self.loadOnInstantiation):
              sample = self.loadData(imgFiles, maskFiles, labelFiles, idx)
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
    
    def getChannels(self):
      return self.channels;
    
    def loadData(self, trainingFileNames, maskFileNames, labelsFileNames, idx):
      imgData = []
      for trainingFileName in trainingFileNames:
        #https://na-mic.org/w/images/a/a7/SimpleITK_with_Slicer_HansJohnson.pdf
        tmp = sitk.ReadImage(str(trainingFileName))
        imgNii = sitk.GetArrayFromImage(tmp)
#         imgNii, imgHeader = load(trainingFileName)
        imgData.append(imgNii)
      imgData = np.stack(imgData).astype('float32')
      imgData = imgData[:,:(imgData.shape[1]/2)*2,:(imgData.shape[2]/2)*2,:(imgData.shape[3]/2)*2]
      
      
        
      maskData = []
      if (len(trainingFileNames) == len(maskFileNames)):
        for maskFileName in maskFileNames:
          tmp = sitk.ReadImage(str(maskFileName))
          maskNii = sitk.GetArrayFromImage(tmp)
          maskData.append(maskNii)
        maskData = np.stack(maskData)
        
#make dimensions even; otehrwise there are probs with average pooling and upsampling         
        maskData = maskData[:,:(maskData.shape[1]/2)*2,:(maskData.shape[2]/2)*2,:(maskData.shape[3]/2)*2]
        
        imgMean = imgData[maskData > 0].mean()
        imgData = imgData - imgMean
        imgStd = imgData[maskData > 0].std()
        imgData = imgData / imgStd
        imgData[maskData == 0] = 0
        self.meansAndStds[idx] = (imgMean, imgStd)
      else:
        imgMean = imgData.mean()
        imgData = imgData - imgMean
        imgStd = imgData.std()
        imgData = imgData / imgStd
        self.meansAndStds[idx] = (imgMean, imgStd)
      
      labelData = []
      if (len(trainingFileNames) == len(labelsFileNames)):
        for labelsFileName in labelsFileNames:
          tmp = sitk.ReadImage(str(labelsFileName))
          labelsNii = sitk.GetArrayFromImage(tmp)
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
          sample = self.loadData(trainingFileNames, maskFileNames, labelsFileNames, idx)

        return sample

    def saveData(self, data, path,  filename, idx = -1):
      if idx > -1:
        (imgMean, imgStd) = self.meansAndStds[idx]
        data = data * imgStd
        data = data + imgMean
      if not os.path.isdir(path):
        os.makedirs(path)
      sitk.WriteImage(data, path + os.path.sep + filename)
      
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
        