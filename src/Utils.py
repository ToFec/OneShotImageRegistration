import numpy as np
import SimpleITK as sitk
def deform(inputVol, x1, y1, z1):
  ##http://simpleitk.github.io/SimpleITK-Notebooks/01_Image_Basics.html
  imgToDeform = sitk.GetImageFromArray(inputVol)
  defField = np.stack([x1, y1, z1])
  defField = np.moveaxis(defField, 0, -1)
  itkVecImg = sitk.GetImageFromArray(defField.astype('f8'))
  displacement = sitk.DisplacementFieldTransform(itkVecImg)

  resampler = sitk.ResampleImageFilter()
  resampler.SetReferenceImage(imgToDeform)
  resampler.SetInterpolator(sitk.sitkLinear)
  resampler.SetDefaultPixelValue(0)
  resampler.SetTransform(displacement)
  moving = resampler.Execute(imgToDeform)
  
  deformedVol = sitk.GetArrayFromImage(moving) 
  
  return deformedVol