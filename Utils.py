import numpy as np
import SimpleITK as sitk
def deform(inputVol, x1, y1, z1):
  deformedVol = np.zeros(inputVol.shape)
  ##http://simpleitk.github.io/SimpleITK-Notebooks/01_Image_Basics.html
  imgToDeform = sitk.GetImageFromArray(inputVol)
  defField = np.stack(x1, y1, z1)
  displacement = sitk.DisplacementFieldTransform(sitk.GetImageFromArray(defField.astype('f8')))

  resampler = sitk.ResampleImageFilter()
  resampler.SetReferenceImage(inputVol)
  resampler.SetInterpolator(sitk.sitkLinear)
  resampler.SetDefaultPixelValue(0)
  resampler.SetTransform(displacement)
  moving = resampler.Execute(inputVol)

  m = inputVol.shape[0]
  n = inputVol.shape[1]
  o = inputVol.shape[2]
  for k in xrange(o):
    for j in xrange(n):
      for i in xrange(m):
        x = np.floor(x1[i,j,k])
        y = np.floor(y1[i,j,k])
        z = np.floor(z1[i,j,k])
        dx = x1[i,j,k] - x
        dy = y1[i,j,k] - y
        dz = z1[i,j,k] - z

        x = np.array(x + j,dtype=np.int)
        y = np.array(y + i,dtype=np.int)
        z = np.array(z + k,dtype=np.int)
        deformedVol[i, j, k] = (1.0 - dx) * (1.0 - dy) * (1.0 - dz) * inputVol[np.min(np.array([np.max(np.array([y, 0])), m - 1])), np.min(np.array([np.max(np.array([x, 0])), n - 1])), np.min(np.array([np.max(np.array([z, 0])), o - 1]))] \
            + (1.0 - dx) * dy * (1.0 - dz) * inputVol[np.min(np.array([np.max(np.array([y + 1, 0])), m - 1])), np.min(np.array([np.max(np.array([x, 0])), n - 1])), np.min(np.array([np.max(np.array([z, 0])), o - 1]))] \
            + dx * (1.0 - dy) * (1.0 - dz) * inputVol[np.min(np.array([np.max(np.array([y, 0])), m - 1])), np.min(np.array([np.max(np.array([x + 1, 0])), n - 1])), np.min(np.array([np.max(np.array([z, 0])), o - 1]))] \
            + (1.0 - dx) * (1.0 - dy) * dz * inputVol[np.min(np.array([np.max(np.array([y, 0])), m - 1])), np.min(np.array([np.max(np.array([x, 0])), n - 1])), np.min(np.array([np.max(np.array([z + 1, 0])), o - 1]))] \
            + dx * dy * (1.0 - dz) * inputVol[np.min(np.array([np.max(np.array([y + 1, 0])), m - 1])), np.min(np.array([np.max(np.array([x + 1, 0])), n - 1])), np.min(np.array([np.max(np.array([z, 0])), o - 1]))] \
            + (1.0 - dx) * dy * dz * inputVol[np.min(np.array([np.max(np.array([y + 1, 0])), m - 1])), np.min(np.array([np.max(np.array([x, 0])), n - 1])), np.min(np.array([np.max(np.array([z + 1, 0])), o - 1]))] \
            + dx * (1.0 - dy) * dz * inputVol[np.min(np.array([np.max(np.array([y, 0])), m - 1])), np.min(np.array([np.max(np.array([x + 1, 0])), n - 1])), np.min(np.array([np.max(np.array([z + 1, 0])), o - 1]))] \
            + dx * dy * dz * inputVol[np.min(np.array([np.max(np.array([y + 1, 0])), m - 1])), np.min(np.array([np.max(np.array([x + 1, 0])), n - 1])), np.min(np.array([np.max(np.array([z + 1, 0])), o - 1]))]
  
  deformedVol = sitk.GetArrayFromImage(image)          
  return deformedVol
