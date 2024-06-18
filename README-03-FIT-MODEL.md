# Model Fit
In this document we discuss on how to fit the Transformation model. The training is coordinated via the notebook [DeepTransformationModel_fit.ipynb](DeepTransformationModel_fit.ipynb)

## File Paths
Here we define two variables. 
* `INPUT_DIR`
  * Defines the location where the data, preprocessed by [README-02-PREPARE-DATA.md](README-02-PREPARE-DATA.md), is stored.
  * Defines the location where the pretrained CNN weights for the `CIBLSX` model version can be loaded from. If no model has been pretrained than new weights will be initialized.
* `OUTPUT_DIR`
  * Defines the location where the best model checkpoints will be saved to.

The notebook assumes that it is run from `OUTPUT_DIR`.

## Model Versions
The variable `version` can take two values:
* `CIBLSX`: Complex Intercept, Linear Shift -- image data and tabular data
* `CIB`: Complex Intercept -- image data only

One can define the the current model version via the `model_version` indicator. This helps to track experiments. Compatibility mode can be triggered via `comp_mode`. This is a boolean. 

If one wants to add an additional version model that contains a differing number of models in the ensemble than defined in , it is possible to modify this in the `version_setup` function in `functions_read_data.py`. Here it is also possible to modify the file names defined according to the previous data preparation step.

## Fitting the Model
The training loop will utilize the data splits as defined in the previous step, [README-02-PREPARE-DATA.md](README-02-PREPARE-DATA.md). 

As described in the paper **TODO: REFERENCE** the linear shift part of the model converges to the parameters as if the tabular data would be fitted utilizing logistic regression. This is valid in the case when using logistic activation function. Before training on each fold, first a logistic regression is fitted on the folds data and those parameters are then included as seed weights before the models training.

During training, the model will be checkpointing the progress and save the best version of the weights. Early stopping is enabled on the validation loss with a patience of `60` while restoring the previously best weights for each ensemble member and data split.

**Note:** During training only the model weights are saved, not the model. One requires to initialize the model first to load the corresponding weights. This functionality is included in further steps.