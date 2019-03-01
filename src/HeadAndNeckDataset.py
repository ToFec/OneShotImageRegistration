import os
import unicodecsv as csv
from torch.utils.data import Dataset
import SimpleITK as sitk
import torch
import numpy as np
import Utils
from eval.LandmarkHandler import PointReader
import Options

class HeadAndNeckDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None, loadOnInstantiation=True, smoothFilter=None):

        self.transform = transform
        self.smooth = smoothFilter
        self.loadOnInstantiation = loadOnInstantiation
        csvtrainingFiles =  open(csv_file, 'rb')
        self.meansAndStds = {}
        self.spacings = {}
        self.origins = {}
        self.directionCosines = {}
        try:        
          trianingCSVFileReader = csv.reader(csvtrainingFiles, delimiter=';', encoding='iso8859_15')
          if (self.loadOnInstantiation):
            self.loadedIgSamples = {}
          
          self.dataFileList = []
          self.labelFileList = []
          self.maskFileList = []
          self.landMarkFileList = []
          
          idx = 0
          for trainingFilePath in trianingCSVFileReader:
            imgFiles = []
            maskFiles = []
            landMarkFiles = []
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
              
              landmarkFileName = trainingFilePath[0] + os.path.sep + str(i) + '0.pts'
              if (os.path.isfile(landmarkFileName)):
                landMarkFiles.append(landmarkFileName)
              
              labelsFileName = trainingFilePath[0] + os.path.sep + 'struct' + str(i) + '.nii.gz'  
              if (os.path.isfile(labelsFileName)):
                labelFiles.append(labelsFileName)
              labelsFileName = trainingFilePath[0] + os.path.sep + 'struct' + str(i) + '.nrrd'  
              if (os.path.isfile(labelsFileName)):
                labelFiles.append(labelsFileName)
              i=i+1
            
            self.dataFileList.append(imgFiles)
            self.labelFileList.append(labelFiles)
            self.maskFileList.append(maskFiles)
            self.landMarkFileList.append(landMarkFiles)
            
            self.channels = len(imgFiles)
            if (self.loadOnInstantiation):
              sample = self.loadData(idx)
              self.loadedIgSamples[idx] = sample
              idx = idx + 1

          
          self.length = len(self.dataFileList)
        finally:
          csvtrainingFiles.close()

    def __len__(self):
        return self.length
    
    def getChannels(self):
      return self.channels;
    
    def loadData(self, idx):
      imgData = []
      spacing = []
      imgSize = []
      trainingFileNames = self.dataFileList[idx]
      maskFileNames = self.maskFileList[idx]
      labelsFileNames = self.labelFileList[idx]
      landMarkFileNames = self.landMarkFileList[idx]
      
      for trainingFileName in trainingFileNames:
        #https://na-mic.org/w/images/a/a7/SimpleITK_with_Slicer_HansJohnson.pdf
        tmp = sitk.ReadImage(str(trainingFileName))
        imgNii = sitk.GetArrayFromImage(tmp)
        curSpacing = tmp.GetSpacing()
        currImgSize = tmp.GetSize()
        if (len(spacing)):
          if(curSpacing != spacing):
            print('Error: input images do not have same voxel spacing.')
        else:
          spacing = curSpacing
          
        if (len(imgSize)):
          if(currImgSize != imgSize):
            print('Error: input images do not have same size.')
        else:
          imgSize = currImgSize
          
        imgData.append(imgNii)
        self.spacings[idx] = spacing
        self.origins[idx] = tmp.GetOrigin()
        self.directionCosines[idx] = tmp.GetDirection()
        
      imgData = np.stack(imgData).astype('float32')
      
      nuOfDownSampleLayers = Options.netDepth - 1
      nuOfDownSampleSteps = len(Options.downSampleRates) -1
      timesDividableByTwo = 2**(nuOfDownSampleLayers + nuOfDownSampleSteps)
      imgData = imgData[:,:int((imgData.shape[1]/timesDividableByTwo)*timesDividableByTwo),:int(imgData.shape[2]/timesDividableByTwo)*timesDividableByTwo,:int(imgData.shape[3]/timesDividableByTwo)*timesDividableByTwo]
      
#       imgData[imgData < -1000] = -1000
#       imgData[imgData > 100] = 100
      
      maskData = []
      if (len(trainingFileNames) == len(maskFileNames)):
        for maskFileName in maskFileNames:
          tmp = sitk.ReadImage(str(maskFileName))
          maskNii = sitk.GetArrayFromImage(tmp)
          maskData.append(maskNii)
        maskData = np.stack(maskData)
        
#make dimensions even; otehrwise there are probs with average pooling and upsampling         
        maskData = maskData[:,:(maskData.shape[1]/timesDividableByTwo)*timesDividableByTwo,:(maskData.shape[2]/timesDividableByTwo)*timesDividableByTwo,:(maskData.shape[3]/timesDividableByTwo)*timesDividableByTwo]
        
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
        labelData = labelData[:,:(labelData.shape[1]/timesDividableByTwo)*timesDividableByTwo,:(labelData.shape[2]/timesDividableByTwo)*timesDividableByTwo,:(labelData.shape[3]/timesDividableByTwo)*timesDividableByTwo]
      
      landmarkData = []
      if (len(trainingFileNames) == len(landMarkFileNames)):
        pr = PointReader()
        for lmFileName in landMarkFileNames:
          points = pr.loadData(lmFileName)
          landmarkData.append(points)
      
      sample = {'image': imgData, 'label': labelData, 'mask': maskData, 'landmarks': landmarkData}
      if self.transform:
        sample = self.transform(sample)
      if self.smooth:
        sample = self.smooth(sample)
      return sample  
    
    def __getitem__(self, idx):
        
        ##works currently only for single thread
        if self.loadOnInstantiation:
          sample = self.loadedIgSamples[idx]
        else:
          sample = self.loadData(idx)

        return sample

    def saveData(self, data, path,  filename, idx = -1, meanSubtract = True):
      if idx > -1:
        if (data.GetPixelIDValue() < 12 and meanSubtract):
          (imgMean, imgStd) = self.meansAndStds[idx]
          data = data * imgStd
          data = data + imgMean
        data.SetSpacing(self.spacings[idx])
        data.SetOrigin(self.origins[idx])
        data.SetDirection(self.directionCosines[idx])
      if not os.path.isdir(path):
        os.makedirs(path)
      sitk.WriteImage(data, path + os.path.sep + filename)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, mask, landmarkData = sample['image'], sample['label'], sample['mask'], sample['landmarks']

        labelTorch = torch.tensor([1])
        if(len(label) > 0):
          labelTorch = torch.from_numpy(label)
          
        maskTorch = torch.tensor([1])
        if(len(mask) > 0):
          maskTorch = torch.from_numpy(mask)
          
          
        return {'image': torch.from_numpy(image),
                'label': labelTorch,
                'mask': maskTorch,
                'landmarks': landmarkData}

class SmoothImage(object):

    def __call__(self, sample):
      image, label, mask, landmarkData = sample['image'], sample['label'], sample['mask'], sample['landmarks']

      for i in range(0,image.shape[0]):
        imgToSmooth = image[i,]
        image[i,] = Utils.smoothArray3D(imgToSmooth, image.device, 1, 0.5, 3)

      return {'image': image,
                'label': label,
                'mask': mask,
                'landmarks': landmarkData}               
        