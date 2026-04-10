import torchio as tio
import SimpleITK as sitk

def normalization(input_img):
    print('Running Normalization')
    znorm = tio.ZNormalization(masking_method=None)
    normalized_image = znorm(input_img)
    sitk.WriteImage(normalized_image, 'normalized_image.nii.gz')
