# Data preparation

## Data pahts
The Docker container for the image stroke_perfusion:2.2.0-publish will start in the working directory /tf/notebooks. The rest of the filesystem and additionally mounted directories are accessible via unix paths.

As defined in the run script, [README-SETUP-ENVIRONMENT.md](README-SETUP-ENVIRONMENT.md), the parameter `--mount type=bind,source="$(pwd)", target=/tf/notebooks` mounts the local directory where the command was executed from, into the Docker container under the path `/tf/notebooks`. This means when we create a directory `data` in the current directory (local file system) it will be available under `/tf/notebooks/data/` in the Docker container as absolute path or `./data` as relative path.

Note: *`"$(pwd)` could be replaced with a fixed path, depending on the environment.*

## Transform the dataset with [split_data.ipynb](split_data.ipynb)

Define two directories here:
* `IMG_DIR`: The input directory that contains the following assets:
  * A `.h5` HDF file that contains the image data with additional metadata.
  * A `.csv` comma separated value file that contains the tabular data.
* `OUTPUT_DIR`: This will hold the outputs used for further training:
  * `prepocessed_dicom_3d.npy` The patients brain volume extracted from the `.h5` file. Contains all patients.
  * `10Fold_ids_V3_mhs.csv` 
    * The file assigning each patient to a `train`, `test` or `val` split for each training fold. 
    * A mapping between `p_idx` and `p_id` to map the patients index and the patients id between the voxels and the tabular data.

### .h5 data structure
It is expected that the .h5 file contains the following structure:

`X`: The image voxels per patient with dimension `(128, 128, 28)` per patient.  
`Y_img`: Not used. A label indicating **TODO: fill in what it actually does** per voxel slice. (28 slices)  
`Y_pat`: Indicating if the patient had a TIA (trans ischemic attack).  
`pat`:   The patients id.

The following call will read the corresponding data.
```py
with h5py.File(path_img, "r") as h5:
    X_in = h5["X"][:]
    Y_img = h5["Y_img"][:]
    Y_pat = h5["Y_pat"][:]
    pat = h5["pat"][:]
```

### .csv data structure
**TODO: Fill out all columns:** Contains the following columns:  
"p_id"  
"mrs3"  
"age"  
"sexm"  
"nihss_baseline"  
"mrs_before"  
"stroke_beforey"  
"tia_beforey"  
"ich_beforey"    
"rf_hypertoniay"  
"rf_diabetesy"  
"rf_hypercholesterolemiay"  
"rf_smokery"  
"rf_atrial_fibrillationy"  
"rf_chdy"  
**"one-missing-colum"**

#### Target variable Y_new
The binary target variable, **0**, and **1** gets created by splitting the modified ranking scale (mrs3) outcome after three months in the following groups: 

**0: favorable**: contains 0, 1, and 2  

**1: unfavorable**: contains 3, 4, 5 and 6

### Training splits
Since stratified cross-validation is applied we split the dataset into 10 folds. Where different seeds are used to generate different versions of splits:
* V0: `StratifiedKFold(n_splits=10, shuffle=True, random_state=100`
* V1: `StratifiedKFold(n_splits=10, shuffle=True, random_state=999)`
* V2: `StratifiedKFold(n_splits=10, shuffle=True, random_state=500)`
* V3: `StratifiedKFold(n_splits=10, shuffle=True, random_state=200)`

### Model versions **TODO: passt das hier hin?**

| **Model Name**     | **Ensembels <br> per split** | **# Folds** | **Activation Function** | **Description**|
| ------             | --------              | ---------   | -----------------                  | ------------------- |
| andrea_split       |                         |             |                                    | Splits as per paper: TODO: DOI und Titel                                                                           |
| 10Fold_sigmoid     | 5                       | 10          | sigmoid                            | stratified by the binarized mrs (mrs &gt; or &lt;= 2)                                                              |
| 10Fold_sigmoid_V0  | 5                       | 10          | sigmoid                            | renamed version of 10Fold_sigmoid                                                                                  |
| 10Fold_softmax_V0  | 5                       | 10          | softmax                            | same Folds as 10Fold_sigmoid                                                                                       |
| 10Fold_softmax_V1  | 10                      | 10          | softmax                            | stratified via mrs                                                                                                 |
| 10Fold_sigmoid_V1  | 10                      | 10          | sigmoid                            | same folds as 10Fold_softmax_V1                                                                                    |
| 10Fold_sigmoid_V2  | 5                       | 10          | sigmoid                            | stratified by the binarized mrs (mrs > or <= 2), <br>different seed than 10Fold_sigmoid_V0                   |
| 10Fold_sigmoid_V2f | 5                       | 10          | sigmoid                            | Same as 10Fold_sigmoid_V2 but with flatten Layer                                                                   |
| 10Fold_signoid_V3  | 5                       | 10          | sigmoid                            | stratified by the binarized mrs (mrs > or <= 2), <br>removed TIA patients, <br>different seed than V0 and V2 |

## Validation of outcome distribution
The distribution of outcomes will be visualized per split.

## Validation of mapping
There is a function to validate that the mapping between patients is correct.