'''
File Organization Script and Train Test Validation Script

Steps to run:
1. Go to this link https://www.synapse.org/Synapse:syn59059776 and download 'BraTS-PTG supplementary demographic information and metadata.csv',
'BraTS2024-BraTS-GLI-TrainingData.zip' and BraTS2024-BraTS-GLI-AdditionalTrainingData.zip'
2. Unzip the zip files and put the folders and the .csv file inside of the directory specified by the variable baseDir
3. Run the script to get downsampled database files

File Organization Script

Files from BraTS dataset are in hierarchal structure (1500+ folders (one for each brain scan)) making them hard to access.
Each of these folders contains 5 .nii.gz (gzip) files, 4 for the mr images and 1 for the segmentation mask.
This script unzip
For each brain scan there are five .nii.gz files

1. Check excel file that references name of brain scan
2. using name of brain scan, navigate to folder
3. unzip all files in folder
4. use nilearn to convert files from .nii to array + use smaller data types to save on space
5. save arrays to .h5 file with two datasets "mr": 4 mr scans "seg": segmentation mask
6. move .h5 file to new folder trainingData
7. delete all unzipped .nii files
8. repeat until all files are unzipped, converted to arrays and deleted


Splits .h5 files between Train, Test and Validation

Splits brain scans into separate train, test and validation folders based on the split percentage

Script
1. Check excel file that references name of brain scan and get list of brain scans
2. Randomize list of names
3. Split list into train,test,validation sets based on split percentage
4. Add Train,Test, Validation folders
5. Move scans in train set to Train folder, scans in test set to Test folder, scans in validation set to Validation folder
'''

#for opening and processing .csv file
import pandas as pd
#for array operations
import numpy as np
#for storing arrays into database files
import h5py
#for opening, moving, unzipping, deleting files using terminal commands
import os
#for reading .nii files
from nilearn import image
#for mean pooling
import torch
import torch.nn.functional as F
#for gaussian pooling
from scipy.ndimage import gaussian_filter
#for randomizing array indicess
import random



### File Processing Script

#Change this folder to the folder you are putting this file in
baseDir = '/Users/aidanthompson/Desktop/DeepLearning/Project/Training'
os.chdir(baseDir)

#destination for .h5 files 
#for mean pooled downsampled data: downsampledTrainingDataMean
#for gaussian filter downsampled data: downsampledTrainingDataBlur
destination = '/Users/aidanthompson/Desktop/DeepLearning/Project/Training/downsampledTrainingDataBlur'

#reading .csv file (you can download this file from the BraTS database)
df = pd.read_csv("BraTS-PTG supplementary demographic information and metadata.csv")

subjectID = df["BraTS Subject ID"]
folder = df["Train/Test/Validation "]
site = df["Site"]

#looping through files in training and training_additional
for i,fileName in enumerate(subjectID):
    if folder[i]== "Train":
        path_ext = "training_data1_v2"+"/"+fileName +"/" 
    elif folder[i] == "Train-additional":
        path_ext = "training_data_additional/"+fileName + "/" 
    else:
        continue
    
    #going to path
    os.chdir(path_ext)

    #unzipping all files: 4 files for inputs and 1 file for segmentation
    os.system('gunzip ' + fileName + '-seg.nii.gz')
    os.system('gunzip ' + fileName + '-t1c.nii.gz')
    os.system('gunzip ' + fileName + '-t1n.nii.gz')
    os.system('gunzip ' + fileName + '-t2f.nii.gz')
    os.system('gunzip ' + fileName + '-t2w.nii.gz')

    #segmentation data
    file = fileName + '-seg.nii'
    img = image.load_img(file)
    segData = img.get_fdata()

    '''
    Downsampling
    
    Reduces size of image from 182 x 208 x 182 to 91 x 109 x 91

    TWO METHODS FOR INPUTS:
        MEAN POOLING (2x2x2 filter)
        BLUR POOLING (GAUSSIAN FILTER + SUBSAMPLING)
    METHOD FOR OUTPUT/LABEL:
        MAX POOLING (2x2x2 filter)
    '''

    #maxpooling labels
    segData = torch.from_numpy(segData).float()
    segData = segData.unsqueeze(0).unsqueeze(0)
    segData = F.max_pool3d(segData, kernel_size=2)
    #converting to integer 8 to conserve space as labels are integers from 0-4
    segData = segData.squeeze().numpy().astype(np.int8)

    #meanpooling or blur pooling inputs
    #mr data files: 4 images
    list = ['-t1c.nii','-t1n.nii','-t2f.nii','-t2w.nii']
    #numpy array to store downsampled data
    mrData = np.zeros((91,109,91,4),dtype=np.float32)

    for i, ext in enumerate(list):
        file = fileName + ext
        img = image.load_img(file)
        data = img.get_fdata()

        #downsample data by meanpooling

        # data = torch.from_numpy(data).float()
        # data = data.unsqueeze(0).unsqueeze(0)
        # data = F.avg_pool3d(data,kernel_size=2)
        # data = data.squeeze().numpy()
        # mrData[:,:,:,i] = data 

        #downsample data by blur pooling (gaussian blur + taking every other data point)
        blurred_data = gaussian_filter(data, sigma=0.7)
        downsampled_data = blurred_data[::2, ::2, ::2]
        mrData[:,:,:,i] = downsampled_data

    #creating dataset file from segmentation and mr (.hdf5 file)
    with h5py.File(fileName + '.h5', 'w') as hdf_file:
        hdf_file.create_dataset('mr',data=mrData ) #input dataset (91 x 109 x 91 x 4)
        hdf_file.create_dataset('seg', data=segData) #label dataset (91 x 109 x 91)
    
    #move file to new folder
    os.system('mv '+fileName + '.h5 ' + destination)

    #delete all files in folder: conserves space on disk
    os.system('rm *')

    #go back to base directory
    os.chdir(baseDir)


### Train, Test, Validation Split Script

#split for train,test and validation datasets
train = 0.6
val = 0.2
test = 0.2

#reading metadata file and getting a list of file names
fileList = []
df = pd.read_csv("BraTS-PTG supplementary demographic information and metadata.csv")
subjectID = df["BraTS Subject ID"]
folder = df["Train/Test/Validation "]

for i,fileName in enumerate(subjectID):
    if folder[i] == "Train" or folder[i] == "Train-additional":
        fileList.append(fileName)

#shuffle file list
random.shuffle(fileList)

#indices for train, val, and test data
endTrain = round(train*len(fileList))
endVal = round(val*len(fileList))+endTrain
end = len(fileList)

#indexing file List based on percentages
trainList = fileList[0:endTrain]
valList = fileList[endTrain:endVal]
testList = fileList[endVal:end]

print(f"Train List length: {len(trainList)} \n Validation List length: {len(valList)} \n Test List length: {len(testList)}")


#creating folders for Train, Validation and Test
baseDir = '/Users/aidanthompson/Desktop/DeepLearning/Project/Training/downsampledTrainingDataBlur'
os.chdir(baseDir) #go to base directory
os.system('mkdir Train')
os.system('mkdir Validation')
os.system('mkdir Test')

#moving processed .h5 dataset files into train, validation and test folders
destination = '/Users/aidanthompson/Desktop/DeepLearning/Project/Training/downsampledTrainingDataBlur/Train'
for train in trainList:
    os.system('mv ' + train + '.h5 ' + destination)

print("Finished moving files to Train folder")

destination = '/Users/aidanthompson/Desktop/DeepLearning/Project/Training/downsampledTrainingDataBlur/Validation'
for val in valList:
    os.system('mv ' + val + '.h5 ' + destination)

print("Finished moving files to Validation folder")


destination = '/Users/aidanthompson/Desktop/DeepLearning/Project/Training/downsampledTrainingDataBlur/Test'
for test in testList:
    os.system('mv ' + test + '.h5 ' + destination)

print("Finished moving files to Test folder")