import sys, getopt, os
import SimpleITK as sitk
import numpy as np
from math import sqrt
import torch
from fileinput import filename
from scipy.interpolate import griddata
from __builtin__ import file

class PointReader():
  
  def loadData(self, filename):
    currPoints = []
    pointFile = open(filename,'r') 
    filename, file_extension = os.path.splitext(filename)
    if file_extension == '.fcsv':#slicer datasets
      for line in pointFile:
        pointsStr = line.split(',')
        point = (np.float32(pointsStr[1])*-1, np.float32(pointsStr[2])*-1, np.float32(pointsStr[3]))
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
  
  def saveDataTensor(self, filename, points, seperator=' '):
    pointFile = open(filename,'w')
    for point in points:
      pointFile.write(str(np.float32(point[0])[0]) + seperator + str(np.float32(point[1])[0]) + seperator + str(np.float32(point[2])[0]) + '\n')
    pointFile.close()
    
  def saveDataList(self, filename, points, seperator=' '):
    pointFile = open(filename,'w')
    for point in points:
      pointFile.write(str(point[0]) + seperator + str(point[1]) + seperator + str(point[2]) + '\n')
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
    pointsAreLists = False
    if len(points) > 0:
      if type(points[0][0]) == list:
        pointsAreLists = True
         
    for point in points:
      if pointsAreLists:
        pointFile.write('Landmark' + str(i) + ',' + str(np.float32(point[0])[0] * -1) + ',' + str(np.float32(point[1])[0] * -1) + ',' + str(np.float32(point[2])[0]) + '\n')
      else:
        pointFile.write('Landmark' + str(i) + ',' + str(np.float32(point[0]) * -1) + ',' + str(np.float32(point[1]) * -1) + ',' + str(np.float32(point[2])) + '\n')
      i+=1
    pointFile.close()     

class PointProcessor():

  def getFileName(self, path, fileName):
    landmarkFileName = path + os.path.sep + fileName + '.pts'
    if (os.path.isfile(landmarkFileName)):
      return landmarkFileName
    landmarkFileName = path + os.path.sep + fileName + '.txt'
    if (os.path.isfile(landmarkFileName)):
      return landmarkFileName
    landmarkFileName = path + os.path.sep + fileName + '.fcsv'
    if (os.path.isfile(landmarkFileName)):
      return landmarkFileName
    return None

  def calculatePointDistance(self, filepath0, filepath1):
    i = 0
    origFiles = []
    deformedFiles = []
    if os.path.isfile(filepath0) and os.path.isfile(filepath1):
      origFiles.append(filepath0)
      deformedFiles.append(filepath1)
      i = i + 1
    else:
      while (True):
        landmarkFileName = self.getFileName(filepath0,str(i) + '0')
        if (landmarkFileName is not None):
          origFiles.append(landmarkFileName)
        else:
          break
        
        landmarkFileName = self.getFileName(filepath1, str(i) + '0deformed')
        if (landmarkFileName is not None):
          deformedFiles.append(landmarkFileName)
  
        i = i + 1

    if (len(origFiles) != len(deformedFiles)):
      deformedFiles = origFiles
      
    pr = PointReader()
    distances = [] 
    stds = []
    for pIdx in range(-1, i - 1):
      origPoints = pr.loadData(origFiles[pIdx])
      deformedPoints = pr.loadData(deformedFiles[pIdx+1])
      mean, std = self.calculatePointSetDistance(origPoints, deformedPoints)
      distances.append( mean )
      stds.append(std)
      
    return distances, stds

  def calculatePointSetDistance(self, points0, points1):
    distances = []
    for idx in range(0,len(points0)):
      point0 = points0[idx]
      point1 = points1[idx]
      distance = 0.0
      for dim in range(0,len(point0)):
        diff = point0[dim] - point1[dim]
        diff *= diff
        distance += diff
      currDist = sqrt(distance)
      distances.append(currDist)
      print(currDist)
    meanDistance = np.mean(distances)
    std = np.std(distances)
    return meanDistance, std

##expects a vector field that points from input to output
## ATTENTION: the neural net outputs a vector field that points from output to input
  def deformPoints(self, filePath, defFieldFileName, referenceImg = 0):
    
    defFieldITK = sitk.ReadImage(str(defFieldFileName))
    defField = sitk.GetArrayFromImage(defFieldITK)
    if (referenceImg != 0):
      refImgItk = sitk.ReadImage(str(referenceImg))
      defFieldSpacing = refImgItk.GetSpacing()
      defFieldOrigin = refImgItk.GetOrigin()
      defFieldDirection = refImgItk.GetDirection()
    else:
      defFieldSpacing = defFieldITK.GetSpacing()
      defFieldOrigin = defFieldITK.GetOrigin()
      defFieldDirection = defFieldITK.GetDirection()
      
    defField = np.moveaxis(defField, 0, 2)
    defField = np.moveaxis(defField, 0, 1)
    
    pr = PointReader()
    points = pr.loadData(filePath)
    
    newPoints = self.deformPointsWithField(points, defField, defFieldOrigin, defFieldSpacing, defFieldDirection)
    filename, file_extension = os.path.splitext(filePath)  
    pr.saveDataList(filename + 'deformed.pts', newPoints)
  
  def deformPointsWithField(self, points, defField, defFieldOrigin, defFieldSpacing, direction):
    
    newPoints = []
    for point in points:

      idx0 = []
      idx1 = []
      part0 = []
      part1 = []
      pointMinusOrigin = [point[0]- defFieldOrigin[0],
                          point[1]- defFieldOrigin[1],
                          point[2]- defFieldOrigin[2]]
      #so far we consider only axis aligend images 
      idxCoords = []
      for idx in range(0,3):
        ijk = pointMinusOrigin[idx] / (defFieldSpacing[idx] * direction[idx*3 + idx])
        idxCoords.append(ijk)
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
      
      pointPosNew=[0.0,0.0,0.0]
      for idx in range(0,3):
        pointPosNew[idx] = ((point[idx] + defVec[idx]))# * defFieldSpacing[idx] * direction[idx*3 + idx]) + defFieldOrigin[idx]
      
      newPoints.append(pointPosNew)
    
    return newPoints    
    
      
  def convertToFcsv(self, filepath):
    pr = PointReader()
    if (os.path.isfile(filepath)):
      points = pr.loadData(filepath)
      filePathName, _ = os.path.splitext(filepath)
      pr.saveDataFcsvSlicer(filePathName + '.fcsv', points)
    else:
      i = 0
      while (True):
        pointFileName = filepath + os.path.sep + str(i) + '0.pts'
        if (os.path.isfile(pointFileName)):
          points = pr.loadData(pointFileName)
          pr.saveDataFcsvSlicer(filepath + os.path.sep + str(i) + '0.fcsv', points)
        else:
          break
        i=i+1  
  
  def convertTxtToPts(self, filename, referenceImg):
    sitkImg = sitk.ReadImage(referenceImg)
    pr = PointReader()
    newPoints = []
    points = pr.loadData(filename)
    for point in points:
      pointWorldCoord = sitkImg.TransformIndexToPhysicalPoint(sitk.VectorInt64([int(point[0]), int(point[1]), int(point[2])]))
      newPoints.append(([pointWorldCoord[0]], [pointWorldCoord[1]], [pointWorldCoord[2]]))
    return newPoints
  
  def convertPtsToTxt(self, filename, referenceImg):
    sitkImg = sitk.ReadImage(referenceImg)
    pr = PointReader()
    newPoints = []
    points = pr.loadData(filename)
    for point in points:
      pointWorldCoord = sitkImg.TransformPhysicalPointToIndex(sitk.VectorDouble([int(point[0]), int(point[1]), int(point[2])]))
      newPoints.append(([pointWorldCoord[0]], [pointWorldCoord[1]], [pointWorldCoord[2]]))
    return newPoints
  
  def convertPtsToTxt2(self, filename, referenceImg):
    sitkImg = sitk.ReadImage(referenceImg)
    pr = PointReader()
    newPoints = []
    points = pr.loadData(filename)
    for point in points:
      pointWorldCoord = sitkImg.TransformPhysicalPointToIndex(sitk.VectorDouble([int(point[0]), int(point[1]), int(point[2])]))
      newPoints.append((pointWorldCoord[0], pointWorldCoord[1], pointWorldCoord[2]))
    return newPoints  

  def convertPoints(self, filename, referenceImg):
    pr = PointReader()
    filePathName, _ = os.path.splitext(filename)
    if '.pts' in filename:
      newPoints = self.convertPtsToTxt(filename, referenceImg)
      fileNameNew = filePathName + '.txt'
      pr.saveDataTensor(fileNameNew, newPoints, '\t')
    elif '.txt' in filename:
      newPoints = self.convertTxtToPts(filename, referenceImg)
      fileNameNew = filePathName + '.pts'
      pr.saveDataTensor(fileNameNew, newPoints)
    else:
      return
  
  def correctPointPositions(self, filepath, referenceImg):
    imgFilePath = filepath + os.path.sep + referenceImg
    if (os.path.isfile(imgFilePath)):
      sitkImg = sitk.ReadImage(imgFilePath)
      
      imgSize = sitkImg.GetSize()
      imgSpacing = sitkImg.GetSpacing()
      imgOrigin = sitkImg.GetOrigin()
      
      ySizeHalf = imgSize[1] / 2.0
      offset = imgOrigin[1] + ySizeHalf * imgSpacing[1]
      if (imgSize[1] % 2 == 0):
        offset -= imgSpacing[1] / 2
        
      i = 0
      pr = PointReader()
      while (True):
        pointFileName = self.getFileName(filepath, str(i) + '0')
        if pointFileName is not None:
          newPoints = []
          points = pr.loadData(pointFileName)
          if '.pts' in pointFileName:
            for point in points:
              newPoint = ([point[0]], [point[1] + offset], [point[2]])
              newPoints.append(newPoint)
          pr.saveDataTensor(filepath + os.path.sep + str(i) + '00.pts', newPoints)
          pr.saveDataFcsvSlicer(filepath + os.path.sep + str(i) + '00.fcsv', newPoints)
        else:
          break
        i=i+1
    else:
      print('reference imgage not found')
  
def main(argv):
  try:
    opts, args = getopt.getopt(argv, '', ['path0=', 'refImg=', 'correctPos', 'path1=', 'calcDiff', 'outputPath=', 'convert', 'deformPoints','visErr' ])
  except getopt.GetoptError as e:#python3
    print(e)
    return
  
  filepath0 = ''
  filepath1 = ''
  outputPath = '.'
  calcDiff = False
  correctPos = False  
  convertPoints = False
  deformPoints = False
  visualiseError = False
  referenceImg = 0
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
    elif opt == '--convert':
      convertPoints = True
    elif opt == '--deformPoints':
      deformPoints = True    
    elif opt == '--refImg':
      referenceImg = arg    
    elif opt == '--visErr':
      visualiseError = True 
        
  pointProcessor = PointProcessor()   
  if (correctPos):
    pointProcessor.correctPointPositions(filepath0, referenceImg)
  elif(convertPoints):
    pointProcessor.convertPoints(filepath0, referenceImg)
  elif(calcDiff):
    distances, stds = pointProcessor.calculatePointDistance(filepath0, filepath1)
    logfile = outputPath + os.path.sep + 'distances.csv'    
    logFile = open(logfile,'a', buffering=0)
    for dist, std in zip(distances, stds):
      logFile.write(str(dist) + ';' + str(std))
    logFile.write('\n')  
    logFile.close()
      
  elif(deformPoints): ##ATTENTION: the def field points from output to input therefore we need no take the target landmarks and deform them
    pointProcessor.deformPoints(filepath0, filepath1, referenceImg) 
  elif(visualiseError):
    pointsIdx = pointProcessor.convertPtsToTxt2(filepath0, referenceImg)
    pointDiffs = []
    pointDiffFile = open(filepath1,'r') 
    filename, file_extension = os.path.splitext(referenceImg)
    for line in pointDiffFile:
      pointsStr = line.split(';')
      pointDiff = np.float32(pointsStr[0])
      pointDiffs.append(pointDiff)
    
    sitkImg = sitk.ReadImage(referenceImg)
    grid_x, grid_y, grid_z = np.mgrid[0:sitkImg.GetSize()[0], 0:sitkImg.GetSize()[1],0:sitkImg.GetSize()[2]]
    grid = griddata(pointsIdx, pointDiffs, (grid_x, grid_y, grid_z), method='linear')
    errrImg = sitk.GetImageFromArray( np.swapaxes(grid, -1, 0) )
    errrImg.SetSpacing( sitkImg.GetSpacing() )
    errrImg.SetOrigin( sitkImg.GetOrigin() )
    errrImg.SetDirection( sitkImg.GetDirection() )
    sitkImg = sitk.WriteImage(errrImg, filename + "Error.nrrd")
    
  else:
    pointProcessor.convertToFcsv(filepath0)
  
  
if __name__ == "__main__":
  main(sys.argv[1:])    