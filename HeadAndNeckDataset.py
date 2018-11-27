import os
import os.path
import unicodecsv as csv
from torch.utils.data import Dataset
import nibabel as nib
import torch
from theano.typed_list.basic import length

class HeadAndNeckDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        csvtrainingFiles =  open(csv_file, 'rb')
        try:        
          trianingCSVFileReader = csv.reader(csvtrainingFiles, delimiter=';', encoding='iso8859_15')
          self.dataFileList = []
          self.labelFileList = []
          self.maskFileList = []
          for trainingFilePath in trianingCSVFileReader:
            imgFiles = []
            maskFiles = []
            labelFiles = []
            for i in range(10):
              trainingFileName = trainingFilePath[0] + '/img' + str(i) + '.nii.gz'
              if (os.path.isfile(trainingFileName)):
                imgFiles.append(trainingFileName)
                
              maskFileName = trainingFilePath[0] + '/mask' + str(i) + '.nii.gz'
              if (os.path.isfile(maskFileName)):
                maskFiles.append(maskFileName)
              labelsFileName = trainingFilePath[0] + '/struct' + str(i) + '.nii.gz'  
              if (os.path.isfile(labelsFileName)):
                labelFiles.append(labelsFileName)
              
            self.dataFileList.append(imgFiles)
            self.labelFileList.append(labelFiles)
            self.maskFileList.append(maskFiles)
            
        finally:
          csvtrainingFiles.close()
          

    def __len__(self):
        return len(self.dataFileList)

    def __getitem__(self, idx):
        
        trainingFileNames = self.dataFileList[idx]
        maskFileNames = self.maskFileList[idx]
        labelsFileNames = self.labelFileList[idx]
        
        for trainingFileName in trainingFileNames:
          imgNii = nib.load(trainingFileName)
          floatData = imgNii.get_fdata()
          maskData = np.concatenate(floatData)
          
        if (len(trainingFileNames) == len(maskFileNames)):
          for maskFileName in maskFileNames:
            maskNii = nib.load(maskFileName)
            floatData= maskNii.get_fdata()
          
        if (len(trainingFileNames) == len(labelsFileNames)):
          for labelsFileName in labelsFileNames:
            labelsNii = nib.load(labelsFileName)
            floatData = labelsNii.get_fdata()
        

        sample = {'image': imgData, 'label': labelData, 'mask': maskData}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
      
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, mask = sample['image'], sample['label'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label),
                'mask': torch.from_numpy(mask)}      
        
        
        
        