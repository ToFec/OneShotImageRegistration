import os
import unicodecsv as csv
from torch.utils.data import Dataset
import nibabel as nib
import torch

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
            trainingFileName = trainingFilePath[0] + '/img.nii.gz'
            maskFileName = trainingFilePath[0] + '/maskBody.nii.gz'
            labelsFileName = trainingFilePath[0] + '/structureSet.nii.gz'
            
            self.dataFileList.append(trainingFileName)
            self.labelFileList.append(labelsFileName)
            self.maskFileList.append(maskFileName)
            
        finally:
          csvtrainingFiles.close()
          

    def __len__(self):
        return len(self.dataFileList)

    def __getitem__(self, idx):
        
        trainingFileName = self.dataFileList[idx]
        maskFileName = self.maskFileList[idx]
        labelsFileName = self.labelFileList[idx]
        imgNii = nib.load(trainingFileName)
        maskNii = nib.load(maskFileName)
        labelsNii = nib.load(labelsFileName)
   
        imgData = imgNii.get_fdata()
        labelData = labelsNii.get_fdata()
        maskData= maskNii.get_fdata()
        

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
        
        
        
        