import sys, getopt
import SimpleITK as sitk
import numpy as np

import matplotlib.pyplot as plt

def main(argv):
  
  try:
    opts, args = getopt.getopt(argv, '', ['jcFile=', 'output=', 'mask=', 'cropBorder='])
  except getopt.GetoptError, e:
    print(e)
    return
    
  maskName=None
  cropBorder = 0
  for opt, arg in opts:
    if opt == '--jcFile':
      jcFile = arg
    elif opt == '--output':
      outputFileName = arg
    elif opt == '--mask':
      maskName = str(arg)   
    elif opt == '--cropBorder':
      cropBorder = int(arg)          

  jacobiImg = sitk.ReadImage(str(jcFile))
  jacobiImgData = sitk.GetArrayFromImage(jacobiImg)
  jacobiImgData = jacobiImgData[cropBorder:jacobiImgData.shape[0]-cropBorder,cropBorder:jacobiImgData.shape[1]-cropBorder,cropBorder:jacobiImgData.shape[2]-cropBorder]
  
  if maskName is not None:
    maskImg = sitk.ReadImage(maskName)
    maskData = sitk.GetArrayFromImage(maskImg) 
    maskData = maskData[cropBorder:maskData.shape[0]-cropBorder,cropBorder:maskData.shape[1]-cropBorder,cropBorder:maskData.shape[2]-cropBorder] 
    jacobiImgData = jacobiImgData[maskData > 0]
  
  mean= jacobiImgData.mean()
  min = jacobiImgData.min()
  max = jacobiImgData.max()
  std = jacobiImgData.std()
  median = np.median(jacobiImgData)
  
#   n, bins, patches = plt.hist(jacobiImgData.flatten(), 400, density=True)
#   plt.show()
  
  jacobiImgData[jacobiImgData[:] < 0] = 0
  
  negativeFragction = float(jacobiImgData.size - np.count_nonzero(jacobiImgData)) / float(jacobiImgData.size)
  

  seperator = ';'
  resultFile = open(outputFileName,'a', buffering=0)
  resultFile.write(str(negativeFragction) + seperator + str(mean) + seperator + str(std) + seperator + str(median) + seperator + str(min) + seperator + str(max) + seperator + str(jacobiImgData.size - np.count_nonzero(jacobiImgData)) + '\n')
  resultFile.close()


  
if __name__ == "__main__":
  main(sys.argv[1:]) 