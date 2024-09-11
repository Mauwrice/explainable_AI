## Introduction

This repo extends the image-only binary outcome prediction framework implemented in https://github.com/Jonas-Bra/xAI_stroke_3d by moving to the Ontram models that allow for running various experiments using semi-structured Deep Transformation models (such as presented in https://arxiv.org/abs/2206.13302). 

The images are represented by 3D-DWI acquired within 3 days of hospital admission. After pre-processing, each 3D image has dimension 128x128x28x1 with zero mean and unit variance. Tabular data include Age, sex, hypertension, prior stroke, smoking, atrial fibrillation, coronary heart disease (CHD), prior transient ischemic attack (TIA), diabetes, hypercholesterolemia, NIHSS baseline score and pre-stroke mRS information. 

The binary outcome is prepresented by a dichotomized modified Rankin Scale (mRS) at 90 days after stroke. 
 
 ## Methods (to do)

- explain and introduce CNN and attributon maps 

## Model Fit

### File:

- `Ontram_fit_10foldCV_CIBLSX`:

### Key changes:

- Architecture change to `ontram`
    - define seperate models for image & tabular data

- Read-data: `split_data_tabular()`  
    - extended to read in tabular data 

- Augmentation: `train_preprocessing()`
    -  zoom, rotate, shift, flip volume operations

## Results assembly

### File:

- `result_assembly_ontram_CIBLSX.ipynb`

### Key-changes

- use `split_data_tabular()` to read in tabular data & image data
- use `predict_ontram()` to predict class 1 probability based on image & tabular input

## Occluison: Heatmaps generation 

### Files:

- `occlusion_all_models_slider_CIBLSX.ipynb`
    - heatmap visualisation for indiviual patients

- `generate_all_heatmaps_CIBLSX.ipynb`
    - create .png, .pdf for all patients  
    - visualize different types heatmaps (max, average, ....)

### Key-changes:

- `volume_occlusion_tabular()`
    - extend `volume_occlusion()` function to include tabular data for the creation of occlusion based heatmaps



## Setting up the environment and starting training
We have documented this process in the file [README-SETUP-ENVIRONMENT.md](README-SETUP-ENVIRONMENT.md)



