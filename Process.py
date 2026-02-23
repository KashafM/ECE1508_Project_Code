from nilearn import plotting

# view sample data
nii_file = './dataset/training_data1_v2/BraTS-GLI-00005-100/BraTS-GLI-00005-100-t2w.nii.gz'
plotting.plot_anat(nii_file)
plotting.show()

# Bias Field Correction (N4ITK algorithm)

# Skull Stripping (SynthStrip)

# Resample (128x128x128)

# Z-Score Normalization