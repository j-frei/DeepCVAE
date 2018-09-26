import SimpleITK as sitk
import numpy as np

def resampleImage(img,resolution,spacings):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing(spacings)
    resampler.SetSize(resolution)
    return resampler.Execute(img)

def normalize(npv):
    mi = npv.min()
    ma = npv.max()
    return (npv - mi) / (ma - mi)

def readNormalizedVolumeByPath(vol_path,config):
    in_img = sitk.ReadImage(vol_path)
    resampled = resampleImage(in_img,config.get('resolution',(128,128,128)),config.get('spacings',(1.5,1.5,1.5)))
    resampled_np = sitk.GetArrayFromImage(resampled)
    return normalize(resampled_np).astype("float32")
