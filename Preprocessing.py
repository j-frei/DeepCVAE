import SimpleITK as sitk
import numpy as np

def resampleImageByAtlas(img,atlas):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(atlas)
    resampler.SetSize(atlas.GetSize())
    return resampler.Execute(img)

def normalize(npv):
    mi = npv.min()
    ma = npv.max()
    return (npv - mi) / (ma - mi)

def readNormalizedVolumeByPath(vol_path,atlas):
    in_img = sitk.ReadImage(vol_path)
    resampled = resampleImageByAtlas(in_img,atlas)
    resampled_np = sitk.GetArrayFromImage(resampled)
    return normalize(resampled_np).astype("float32")

def readNormalizedAtlasAndITKAtlas(vol_path):
    in_img = sitk.ReadImage(vol_path)
    resampled_np = sitk.GetArrayFromImage(in_img)
    return normalize(resampled_np).astype("float32"),in_img
