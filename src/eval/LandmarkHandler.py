import sys, getopt, os
import SimpleITK as sitk
import numpy as np
from math import sqrt

class PointReader():
  def loadData(self, filename):
    currPoints = []
    pointFile = open(filename,'r') 
    filename, file_extension = os.path.splitext(filename)
    if file_extension == '.fcsv':#slicer datasets
      for line in pointFile:
        pointsStr = line.split(',')
        point = (np.float32(pointsStr[1]), np.float32(pointsStr[2]), np.float32(pointsStr[3]))
        currPoints.append(point)      
    elif file_extension == '.pts':#popi datasets
      for line in pointFile:
        pointsStr = line.split( )
        point = (np.float32(pointsStr[0]), np.float32(pointsStr[1]), np.float32(pointsStr[2]))
        currPoints.append(point)
    elif file_extension == '.txt':#dirlab datasetes
      for line in pointFile:
        pointsStr = line.split('\t')
        point = (np.float32(pointsStr[0]), np.float32(pointsStr[1]), np.float32(pointsStr[2]))
        currPoints.append(point)
    return currPoints
  
  def saveData(self, filename, points):
    pointFile = open(filename,'w') 
    for point in points:
      pointFile.write(str(np.float32(point[0])[0]) + ' ' + str(np.float32(point[1])[0]) + ' ' + str(np.float32(point[2])[0]) + '\n') ##little hack to print torch tensors and numpy values
    pointFile.close()

  def saveDataFcsv(self, filename, points):
    pointFile = open(filename,'w')
    i = 0 
    for point in points:
      pointFile.write('Landmark' + str(i) + ',' + str(point[0]) + ',' + str(point[1]) + ',' + str(point[2]) + '\n')
      i+=1
    pointFile.close()
  
  def saveDataFcsvSlicer(self, filename, points):
    pointFile = open(filename,'w')
    i = 0 
    for point in points:
      pointFile.write('Landmark' + str(i) + ',' + str(point[0] * -1) + ',' + str(point[1] * -1) + ',' + str(point[2]) + '\n')
      i+=1
    pointFile.close()     

class PointProcessor():

  def calculatePointDistance(self, filepath0, filepath1):
    i = 0
    origFiles = []
    deformedFiles = []
    while (True):
      landmarkFileName = filepath0 + os.path.sep + str(i) + '0.pts'
      if (os.path.isfile(landmarkFileName)):
        origFiles.append(landmarkFileName)
      else:
        break
      
      landmarkFileName = filepath1 + os.path.sep + str(i) + '0deformed.pts'
      if (os.path.isfile(landmarkFileName)):
        deformedFiles.append(landmarkFileName)

      i = i + 1

    if (len(origFiles) != len(deformedFiles)):
      deformedFiles = origFiles
      
    pr = PointReader()
    distances = [] 
    for pIdx in range(-1, i - 1):
      origPoints = pr.loadData(origFiles[pIdx])
      deformedPoints = pr.loadData(deformedFiles[pIdx+1])
      distances.append( self.calculatePointSetDistance(origPoints, deformedPoints) )
      
    return distances

  def calculatePointSetDistance(self, points0, points1):
    meanDistance = 0.0
    for idx in range(0,len(points0)):
      point0 = points0[idx]
      point1 = points1[idx]
      distance = 0.0
      for dim in range(0,len(point0)):
        diff = point0[dim] - point1[dim]
        diff *= diff
        distance += diff
      meanDistance += sqrt(distance)
    meanDistance /= len(points0)
    return meanDistance

##expects a vector field that points from input to output
## ATTENTION: the neural net outputs a vector field that points from output to input
  def deformPoints(self, filePath, defFieldFileName, referenceImg = 0):
    
    defFieldITK = sitk.ReadImage(str(defFieldFileName))
    defField = sitk.GetArrayFromImage(defFieldITK)
    if (referenceImg != 0):
      refImgItk = sitk.ReadImage(str(referenceImg))
      defFieldSpacing = refImgItk.GetSpacing()
      defFieldOrigin = refImgItk.GetOrigin()
    else:
      defFieldSpacing = defFieldITK.GetSpacing()
      defFieldOrigin = defFieldITK.GetOrigin()
      
    defField = np.moveaxis(defField, 0, 2)
    defField = np.moveaxis(defField, 0, 1)
    
    pr = PointReader()
    points = pr.loadData(filePath)
    
    newPoints = self.deformPointsWithField(points, defField, defFieldOrigin, defFieldSpacing)
    filename, file_extension = os.path.splitext(filePath)  
    pr.saveDataFcsv(filename + 'deformed.fcsv', newPoints)
  
  def deformPointsWithField(self, points, defField, defFieldOrigin, defFieldSpacing):
    
    newPoints = []
    for point in points:

      idx0 = []
      idx1 = []
      part0 = []
      part1 = []
      for idx in range(0,3):
        ijk = (point[idx] - defFieldOrigin[idx]) / defFieldSpacing[idx]
        firstIdx = int(ijk)
        idx0.append( firstIdx )
        idx1.append( firstIdx + 1 )
        firsPart = ijk - firstIdx
        part1.append( firsPart )
        part0.append( 1 - firsPart )
      
      defVec = (0,0,0)
      if (idx0[0] > 0 and idx0[1] > 0 and idx0[2] > 0 and idx1[0] < defField.shape[0] and idx1[1] < defField.shape[1] and idx1[2] < defField.shape[2]):
        defVec = part0[0] * part0[1] * part0[2] * defField[idx0[0],idx0[1],idx0[2],:] + part0[0] * part0[1] * part1[2] * defField[idx0[0],idx0[1],idx1[2],:] + \
        part0[0] * part1[1] * part0[2] * defField[idx0[0],idx1[1],idx0[2],:] + part1[0] * part0[1] * part0[2] * defField[idx1[0],idx0[1],idx0[2],:] + \
        part0[0] * part1[1] * part1[2] * defField[idx0[0],idx1[1],idx1[2],:] + part1[0] * part1[1] * part0[2] * defField[idx1[0],idx1[1],idx0[2],:] + \
        part1[0] * part0[1] * part1[2] * defField[idx1[0],idx0[1],idx1[2],:] + part1[0] * part1[1] * part1[2] * defField[idx1[0],idx1[1],idx1[2],:]
      
      pointPosNew = (point[0] + defVec[0], point[1] + defVec[1], point[2] + defVec[2])
      newPoints.append(pointPosNew)
    
    return newPoints    
    
      
  def convertToFcsv(self, filepath):
    i = 0
    pr = PointReader()
    while (True):
      pointFileName = filepath + os.path.sep + str(i) + '0.pts'
      if (os.path.isfile(pointFileName)):
        points = pr.loadData(pointFileName)
        pr.saveDataFcsvSlicer(filepath + os.path.sep + str(i) + '0.fcsv', points)
      else:
        break
      i=i+1  
  
  def correctPointPositions(self, filepath, referenceImg):
    imgFilePath = filepath + os.path.sep + referenceImg
    if (os.path.isfile(imgFilePath)):
      reader = sitk.ImageFileReader()
      reader.SetFileName(imgFilePath)
      reader.ReadImageInformation()
      imgSize = reader.GetSize()
      imgSpacing = reader.GetSpacing()
      imgOrigin = reader.GetOrigin()
      
      ySizeHalf = imgSize[1] / 2.0
      offset = imgOrigin[1] + ySizeHalf * imgSpacing[1]
      if (imgSize[1] % 2 == 0):
        offset -= imgSpacing[1] / 2
      
      
      i = 0
      pr = PointReader()
      while (True):
        pointFileName = filepath + os.path.sep + str(i) + '0.pts'
        if (os.path.isfile(pointFileName)):
          newPoints = []
          points = pr.loadData(pointFileName)
          for point in points:
            newPoint = (point[0], point[1] + offset, point[2])
            newPoints.append(newPoint)
          pr.saveData(filepath + os.path.sep + str(i) + '00.pts', newPoints)
          pr.saveDataFcsv(filepath + os.path.sep + str(i) + '00.fcsv', newPoints)
        else:
          break
        i=i+1
      
  
def main(argv):
  try:
    opts, args = getopt.getopt(argv, '', ['path0=', 'refImg=', 'correctPos', 'path1=', 'calcDiff', 'outputPath=' ])
  except getopt.GetoptError, e:
    print(e)
    return
  
  filepath0 = ''
  filepath1 = ''
  outputPath = '.'
  calcDiff = False
  correctPos = False  
  for opt, arg in opts:
    if opt == '--path0':
      filepath0 = arg
    elif opt == '--correctPos':
      correctPos = True
    elif opt == '--calcDiff':
      calcDiff = True      
    elif opt == '--path1':
      filepath1 = arg
    elif opt == '--outputPath':
      outputPath = arg
    elif opt == '--refImg':
      referenceImg = arg    
        
  pointProcessor = PointProcessor()   
  if (correctPos):
    pointProcessor.correctPointPositions(filepath0, referenceImg)
  elif(calcDiff):
    distances = pointProcessor.calculatePointDistance(filepath0, filepath1)
    logfile = outputPath + os.path.sep + 'distances.csv'    
    logFile = open(logfile,'w', buffering=0)
    for dist in distances:
      logFile.write(str(dist) + ';')
      
    logFile.close()
      
      
  else:
    pointProcessor.convertToFcsv(filepath0)
  
  
if __name__ == "__main__":
  main(sys.argv[1:])    