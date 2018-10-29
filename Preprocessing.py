import SimpleITK as sitk
import numpy as np

def readNormalizedVolumeByPath(vol_path,atlas):
    in_img = sitk.ReadImage(vol_path)
    resampled = resampleImageByAtlas(in_img,atlas)
    resampled_np = fromITKtoNormalizedNumpy(resampled)
    return normalize(resampled_np)

def readNormalizedAtlasAndITKAtlas(vol_path):
    in_img = sitk.ReadImage(vol_path)
    atlas_np = fromITKtoNormalizedNumpy(in_img)
    return normalize(atlas_np),in_img

def fromITKtoNormalizedNumpy(itk_vol):
    # sitk has xyz in GetPixel
    vol_np = sitk.GetArrayFromImage(itk_vol)
    # swap axes from default zyx (numpy) to xyz
    vol_np = np.transpose(vol_np,(2,1,0))
    # apply directions
    vol_dirs = np.array(itk_vol.GetDirection()).reshape(3,3)
    vol_np = np.flip(vol_np,[ a for a in range(3) if vol_dirs[a,a] == -1.])
    vol_np = normalize(vol_np)
    return vol_np.astype(np.float32)

def resampleImageByAtlas(img,atlas):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(atlas)
    resampler.SetSize(atlas.GetSize())
    return resampler.Execute(img)

def normalize(npv):
    mi = npv.min()
    ma = npv.max()
    return (npv - mi) / (ma - mi)
