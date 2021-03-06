import sys, getopt
import SimpleITK as sitk
import numpy as np

import medpy.metric as medMet


class BinaryImageEvaluator():
  def calculateDice(self, img0, img1):
    dc = medMet.binary.dc(img0, img1)
    return dc
  
  def calculateHausdorffDistance(self, img0, img1, spacing):
    hd = medMet.binary.hd(img0, img1, spacing)
    return hd
  
  def calculateASSD(self, img0, img1, spacing):
    assd = medMet.binary.assd(img0, img1, spacing)
    return assd


def loadImage(fileName):
    imgITK = sitk.ReadImage(str(fileName))
    img = sitk.GetArrayFromImage(imgITK)
    spacing = imgITK.GetSpacing()
    return img, (spacing[2], spacing[1], spacing[0])

def main(argv):
  try:
    opts, args = getopt.getopt(argv, 'f:m:o:c', ['file0=', 'file1=', 'outputFileName=', 'cropZ'])
  except getopt.GetoptError:
    return

  outputfile = None
  cropZ = False
  for opt, arg in opts:
    if opt in ("-f", "--file0"):
      filename0 = arg
    elif opt in ("-m", "--file1"):
      filename1 = arg
    elif opt in ("-o", "--outputFileName"):
      outputfile = arg
    elif opt in ("-c", "--cropZ"):
      cropZ = True
      
  img0, spacing = loadImage(filename0)
  img1, _ = loadImage(filename1)
  
  if outputfile is not None:
    logFile = open(outputfile,'a', buffering=0)
  
    if img0.shape == img1.shape and img0.max() > 0 and img1.max() > 0:
      bie = BinaryImageEvaluator()
      
      if cropZ:
        indices0 = np.where(img0 > 0)
        indices1 = np.where(img1 > 0)
        upperbound = np.min([np.max(indices0[0]),np.max(indices1[0])])
        lowerBound = np.max([np.min(indices0[0]),np.min(indices1[0])])
        img0 = img0[lowerBound:upperbound,:,:]
        img1 = img1[lowerBound:upperbound,:,:]
      
      hd = bie.calculateHausdorffDistance(img0, img1, spacing)
      dc = bie.calculateDice(img0, img1)
      assd = bie.calculateASSD(img0, img1, spacing)
  
      logFile.write(str(dc) + ';' + str(hd) + ';' + str(assd))
    else:
      logFile.write('')
    logFile.write('\n')  
    logFile.close()
      
      
  

if __name__ == '__main__':
  main(sys.argv[1:])