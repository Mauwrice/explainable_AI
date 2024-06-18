# Assemble the Results of Model Training
This document discusses on how to assemble and analyze the models trained in the previous step.  
The analysis is facilitated via the notebook [result_assembly_fin.ipynb](result_assembly_fin.ipynb).


## File Paths
Here we define two variables. 
* `INPUT_DIR`
  * Defines the location where the model weights, trained by [README-03-FIT-MODEL.md](README-03-FIT-MODEL.md), will be loaded from.
* `OUTPUT_DIR`
  * Defines the location where the analysis results will be saved to. This includes the loss curves and model performance metrics.

## Model Definition
One defines the parameters `version`, `model_version`, 
 `comp_mode`, `model_nrs`, and `which_splits` such that they correspond to the values utilized to train the model one whishes to analyze.

## Model Analysis
The model is recreated according to the previously defined parameters and the weights are loaded. 

> [!IMPORTANT]  
> TODO: Why tune the weights?

Now for every split and every member of the ensemble predictions for the binary mrs outcome are made.

## Classification Threshold Calculation
On the validation data the calssification threshold is calculated and then applied to the test data. The standard deviation of prediction is normalized and then serves as a measure of uncertainty.

## Result Analysis
The previously generated result data is now visualized. 

This includes the following results:
* Boxplots of the mean log-odds ratio of the `estimated average betas` per ``variable``. For:
  * `means_per_parameter`
  * `means_per_parameter_w`
* The model evaluation metrics as tables, for the untuned and tuned Transformation models.
* The performance metrics on the test set.
* The training and validation loss curves gathered during training.