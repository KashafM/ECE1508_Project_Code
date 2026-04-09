# Segmentation Models for Brain Tumor Segmentation

## Project Overview


Our project aims to investigate segmentation models for brain tumor segmentation using the BraTS 2024 dataset for Post-Treatment Glioma.

### Dataset: BraTS 2024 Post-Treatment Glioma Dataset
- 1621 Individual Brain Scans from MRI machines
- Each brain scan has a resolution of 182 x 208 x 182
- Each Brain Scan has 4 Images (T1, Post Contrast T1 Weighted, T2 Weighted, T2 FLAIR)
- Each Brain Scan has a segmentation mask with 5 possible labels (Normal Tissue, Necrotic Tumor Core, Tumor Infiltration and Edema, Enhancing Tumor Core and Resection Cavity)
- All files are stored in .nii (Neuroimaging Informatics Technology Initiative) files which is standard for MRI scans

### Preprocessing
- Downsampling: To reduce the size of the dataset to allow for training with the limited computational resources available, the dataset was downsampled by a factor of 2 from 182 x 208 x 182 to 91 x 109 x 91
  - Mean-Pooling: Inputs were downsampled using mean-pooling with a filter of 2x2x2. Labels were downsampled by max-pooling with a filter of 2x2x2 
  - Gaussian-Pooling: Gaussian filter was first applied to inputs and then the inputs were subsampled. Labels were downsampled by max-pooling with a filter of 2x2x2
- Other Preprocessing: Due to the computational cost of our pre-processing pipeline (took >1 min per image to preprocess which scales to over 100 hours of total preprocessing), data was not preprocessed. Here is the pipeline we would've implemented if we had greater computational resources
  - Bias Field Correction: Removes low-frequency noise from MRI images
  - Skull Stripping: Removes non-brain tissue such as the skull from input images
  - Intensity normalization

### Model tested
- The 3D-UNet was the model of choice for this project as it has been shown to perform well on medical segmentation tasks in the past
- A diagram showing the architecture of the 3D-Unet used is shown below

### Variations of Models Tested
- Number of features on first layer: 8 vs 16
  - Number of features doubles after every subsequent encoder
- Number of encoder/decoder layers: 3 or 4:   
- Downsampling method: Mean-Pooling and Gaussian-Pooling
- Normalization method: Batch and Instance Normalization
- Batch Size: 4 or 16


### Results
- Comparison of training and validation curves
- Sensitivity Analysis
  - 8 models with different number of features, number of encoder layers nad downsampling method
- Batch Size and Normalization Comparison




## Description of Files



### FileProcessing+TrainTestValidationSplit.py
- Processes files from BraTS 2024 database for post-treatment glioma
- Converts .nii files into arrays
- Downsamples inputs using mean-pooling or gaussian filter + sub-sampling to reduce the size of the image by a factor of 8
- Downsamples labels using max-pooling to reduce the size of the image by a factor of 8
- Saves inputs and labels for each scan into a .h5 database file and moves it to another folder
- After all brain scans have been converted into .h5 files they are split into train, test, validation datasets (each in their own folder)


### Inputs+LabelVisualization.ipynb
- Takes one brain scan and visualizes the inputs and labels for one z-slice of the 3d image
- Four inputs: T1 (t1n), Post Contrast T1 Weighted (t1c), T2 Weighted (t2w), T2 FLAIR (t2f)
- Five labels for output: Normal Tissue, Necrotic Tumor Core, Tumor Infiltration and Edema, Enhancing Tumor Core, Resection Cavity

### FilePreprocessing.py
- Pipeline for implementing pre-processing for the images

### Training.ipynb
- To be used in Google Colab
- Moves files into local colab folder to allow for easier retrieval
- Includes 3D-UNet model whose parameters were adjusted (number of features, number of encoder/decoder layers)
- Uses DICE loss for training and DICE score as the main evaluation metric
- Visualizes prediction and ground truth using 2d slices and 3d plot


 
