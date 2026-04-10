import SimpleITK as sitk

def bias_correction(input_path, output_path): # Bias Field Correction (N4ITK algorithm)
    print("Running Bias Correction")
    input_img = sitk.ReadImage(input_path, sitk.sitkFloat32)
    mask_img = sitk.OtsuThreshold(input_img, 0, 1, 200)
    shrinkFactor = 1
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4
    corrected_image = corrector.Execute(input_img, mask_img)
    log_bias_field = corrector.GetLogBiasFieldAsImage(input_img)
    corrected_image_fr = corrected_image / sitk.Exp(log_bias_field)
    sitk.WriteImage(corrected_image_fr, output_path)