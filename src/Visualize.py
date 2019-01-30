import matplotlib.pyplot as plt

def plotDataset(dataset):
  nuOfimg=len(dataset)
  nuOfImgPerAxes=max(1,round(nuOfimg/2,0))
  for i in range(0, nuOfimg):
      plt.subplot(2,nuOfImgPerAxes,i+1)
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
          plt.imshow(sample['label'][:,:,slice],cmap='jet', alpha=0.5,)
  plt.show(block=False)
  
def plotImageData(imageData, blockFig=False):
  nuOfimg=imageData.shape[0]
  nuOfImgPerAxes=max(1,round(nuOfimg/2,0))
  plt.figure()
  for i in range(0, nuOfimg):
      plt.subplot(2,nuOfImgPerAxes,i+1)
      sample = imageData[i,]
      if (sample.dim() == 4):
        slice = int(sample.shape[3] / 2)
        plt.imshow(sample[0,:,:,slice],cmap='gray')
      elif (sample.dim() == 3):
        slice = int(sample.shape[2] / 2)
        plt.imshow(sample[:,:,slice],cmap='gray')
  plt.show(block=blockFig)