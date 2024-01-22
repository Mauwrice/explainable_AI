
from __future__ import print_function


import os
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical


#ontram functions
from k_ontram_functions.ontram import ontram
from k_ontram_functions.ontram_loss import ontram_loss
from k_ontram_functions.ontram_metrics import ontram_acc, ontram_auc
from k_ontram_functions.ontram_predict import predict_ontram, get_parameters




import numpy as np
import ipywidgets as widgets

from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, Layout, AppLayout
from IPython.display import display
from termcolor import colored

import functions_occlusion as oc
import functions_gradcam as gc
import functions_plot_heatmap as phm

import os 
import pandas as pd
import numpy as np
import h5py
import pickle as pkl

#import functions_read_data as rdat
# Tensorflow/Keras
import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.keras import layers
import pickle as pkl

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from tensorflow.keras.callbacks import ModelCheckpoint

def read_and_split_img_data_andrea_maurice(path_img, path_tab, path_splits, split, check_print = True):   
    # path_img: path to image data
    # path_tab: path to tabular data
    # path_splits: path to splitting definition
    # split: which split to use (1,2,3,4,5,6)
    # check_print: print shapes of data
     
    ## read image data
    with h5py.File(path_img, "r") as h5:
    # with h5py.File(IMG_DIR2 + 'dicom-3d.h5', "r") as h5:
    # both images are the same
        X_in = h5["X"][:]
        Y_img = h5["Y_img"][:]
        Y_pat = h5["Y_pat"][:]
        pat = h5["pat"][:]
    
    X_in = np.expand_dims(X_in, axis = 4)
    ## read tabular data
    dat = pd.read_csv(path_tab, sep=",")       

    ## read splitting file
    andrea_splits = pd.read_csv(path_splits, 
                                sep='\,', header = None, engine = 'python', 
                                usecols = [1,2,3]).apply(lambda x: x.str.replace(r"\"",""))
    andrea_splits.columns = andrea_splits.iloc[0]
    andrea_splits.drop(index=0, inplace=True)
    andrea_splits = andrea_splits.astype({'idx': 'int32', 'spl': 'int32'})
    splitx = andrea_splits.loc[andrea_splits['spl']==split]        

    
    ## extract X and Y and split into train, val, test
    n = []
    for p in pat:
        if p in dat.p_id.values:
            n.append(p)
    n = len(n)

    # match image and tabular data
    X = np.zeros((n, X_in.shape[1], X_in.shape[2], X_in.shape[3], X_in.shape[4]))
    X_tab = np.zeros((n, 13))
    Y_mrs = np.zeros((n))
    Y_eventtia = np.zeros((n))
    p_id = np.zeros((n))

    i = 0
    for j, p in enumerate(pat):
        if p in dat.p_id.values:
            k = np.where(dat.p_id.values == p)[0]
            X_tab[i,:] = dat.loc[k,["age", "sexm", "nihss_baseline", "mrs_before",
                                   "stroke_beforey", "tia_beforey", "ich_beforey", 
                                   "rf_hypertoniay", "rf_diabetesy", "rf_hypercholesterolemiay", 
                                   "rf_smokery", "rf_atrial_fibrillationy", "rf_chdy"]]
            X[i] = X_in[j]
            p_id[i] = pat[j]
            Y_eventtia[i] = Y_pat[j]
            Y_mrs[i] = dat.loc[k, "mrs3"]
            i += 1        
        
    ## all mrs <= 2 are favorable all higher unfavorable
    Y_new = []
    for element in Y_mrs:
        if element in [0,1,2]:
            Y_new.append(0)
        else:
            Y_new.append(1)
    Y_new = np.array(Y_new)
   
    X = np.squeeze(X)
    X = np.float32(X)

    train_idx = splitx["idx"][splitx['type'] == "train"].to_numpy() -1 
    valid_idx = splitx["idx"][splitx['type'] == "val"].to_numpy() - 1 
    test_idx = splitx["idx"][splitx['type'] == "test"].to_numpy() - 1 

    X_train = X[train_idx]
    X_valid = X[valid_idx]
    X_test = X[test_idx]
    
    Y_train = Y_new[train_idx]
    Y_valid = Y_new[valid_idx]
    Y_test = Y_new[test_idx]
         
    X_tab_train = X_tab[train_idx]
    X_tab_valid = X_tab[valid_idx]
    X_tab_test = X_tab[test_idx]
        
        
    ## safe data in table
    results = pd.DataFrame(
        {"p_idx": test_idx+1,
         "p_id": p_id[test_idx],
         "mrs": Y_mrs[test_idx],
         "unfavorable": Y_test
        }
    )
    
    return (X_train, X_valid, X_test, X_tab_train, X_tab_valid, X_tab_test), (Y_train, Y_valid, Y_test), results

# Model for the intercept function: C = number of classes
def mod_baseline(C):
    mod = keras.Sequential(name = "mod_baseline")
    mod.add(keras.Input(shape = (1, )))
    mod.add(keras.layers.Dense(C - 1, activation = "linear", use_bias = False))
    return mod

# Model for linear shift terms
def mod_linear_shift(x):
    mod = keras.Sequential(name = "mod_linear_shift")
    mod.add(keras.Input(shape = (x, )))
    mod.add(keras.layers.Dense(1, activation = "linear", use_bias = False))
    return mod

# Model for complex shift terms
def mod_complex_shift(x):
    mod = keras.Sequential(name = "mod_complex_shift")
    mod.add(keras.Input(shape = (x, )))
    mod.add(keras.layers.Dense(8, activation = "relu"))
    mod.add(keras.layers.Dense(8, activation = "relu"))
    mod.add(keras.layers.Dense(1, activation = "linear", use_bias = False))
    return mod  

def img_model_linear(input_shape, output_shape, activation = "linear"):
    initializer = keras.initializers.he_normal(seed = 2202)
    in_ = keras.Input(shape = input_shape)
    x = keras.layers.Convolution3D(32, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(in_)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Convolution3D(32, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Convolution3D(64, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Convolution3D(64, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation = 'relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation = 'relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    out_ = keras.layers.Dense(output_shape, activation = activation, use_bias = False)(x) 
    nn_im = keras.Model(inputs = in_, outputs = out_)
    return nn_im

def img_model_linear_final(input_shape, output_shape, activation = "linear"):
    initializer = keras.initializers.he_normal(seed = 2202)
    in_ = keras.Input(shape = input_shape)

    # conv block 0
    x = keras.layers.Convolution3D(32, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(in_)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # conv block 1
    x = keras.layers.Convolution3D(32, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # conv block 2
    x = keras.layers.Convolution3D(64, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # conv block 3
    x = keras.layers.Convolution3D(64, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # cnn to flat connection
    x = layers.GlobalAveragePooling3D()(x) 
    
    # flat block
    x = keras.layers.Dense(128, activation = 'relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation = 'relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    out_ = keras.layers.Dense(output_shape, activation = activation, use_bias = False)(x) 
    nn_im = keras.Model(inputs = in_, outputs = out_)
    return nn_im


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inv_sigmoid(x):
    return np.log(x / (1 - x))




def read_and_split_img_data_andrea_maurice2(path_img, path_tab, path_splits, split, check_print = True):   
    # path_img: path to image data
    # path_tab: path to tabular data
    # path_splits: path to splitting definition
    # split: which split to use (1,2,3,4,5,6)
    # check_print: print shapes of data
     
    ## read image data
    with h5py.File(path_img, "r") as h5:
    # with h5py.File(IMG_DIR2 + 'dicom-3d.h5', "r") as h5:
    # both images are the same
        X_in = h5["X"][:]
        Y_img = h5["Y_img"][:]
        Y_pat = h5["Y_pat"][:]
        pat = h5["pat"][:]
    
    X_in = np.expand_dims(X_in, axis = 4)
    ## read tabular data
    dat = pd.read_csv(path_tab, sep=",")       

    ## read splitting file
    andrea_splits = pd.read_csv(path_splits, 
                                sep='\,', header = None, engine = 'python', 
                                usecols = [1,2,3]).apply(lambda x: x.str.replace(r"\"",""))
    andrea_splits.columns = andrea_splits.iloc[0]
    andrea_splits.drop(index=0, inplace=True)
    andrea_splits = andrea_splits.astype({'idx': 'int32', 'spl': 'int32'})
    splitx = andrea_splits.loc[andrea_splits['spl']==split]        

    
    ## extract X and Y and split into train, val, test
    n = []
    for p in pat:
        if p in dat.p_id.values:
            n.append(p)
    n = len(n)

    # match image and tabular data
    X = np.zeros((n, X_in.shape[1], X_in.shape[2], X_in.shape[3], X_in.shape[4]))
    X_tab = np.zeros((n, 13))
    Y_mrs = np.zeros((n))
    Y_eventtia = np.zeros((n))
    p_id = np.zeros((n))

    i = 0
    for j, p in enumerate(pat):
        if p in dat.p_id.values:
            k = np.where(dat.p_id.values == p)[0]
            X_tab[i,:] = dat.loc[k,["age", "sexm", "nihss_baseline", "mrs_before",
                                   "stroke_beforey", "tia_beforey", "ich_beforey", 
                                   "rf_hypertoniay", "rf_diabetesy", "rf_hypercholesterolemiay", 
                                   "rf_smokery", "rf_atrial_fibrillationy", "rf_chdy"]]
            X[i] = X_in[j]
            p_id[i] = pat[j]
            Y_eventtia[i] = Y_pat[j]
            Y_mrs[i] = dat.loc[k, "mrs3"]
            i += 1        
        
    ## all mrs <= 2 are favorable all higher unfavorable
    Y_new = []
    for element in Y_mrs:
        if element in [0,1,2]:
            Y_new.append(0)
        else:
            Y_new.append(1)
    Y_new = np.array(Y_new)
    p_idxx = np.arange(0, len(p_id))+1
   
    X = np.squeeze(X)
    X = np.float32(X)

    train_idx = splitx["idx"][splitx['type'] == "train"].to_numpy() -1 
    valid_idx = splitx["idx"][splitx['type'] == "val"].to_numpy() - 1 
    test_idx = splitx["idx"][splitx['type'] == "test"].to_numpy() - 1 

    train_idxx = splitx["idx"][splitx['type'] == "train"].to_numpy() 
    valid_idxx = splitx["idx"][splitx['type'] == "val"].to_numpy()
    test_idxx = splitx["idx"][splitx['type'] == "test"].to_numpy()

    X_train = X[train_idx]
    X_valid = X[valid_idx]
    X_test = X[test_idx]
    
    Y_train = Y_new[train_idx]
    Y_valid = Y_new[valid_idx]
    Y_test = Y_new[test_idx]
         
    X_tab_train = X_tab[train_idx]
    X_tab_valid = X_tab[valid_idx]
    X_tab_test = X_tab[test_idx]
        
        
    ## safe data in table
    results2 = pd.DataFrame(
        {"p_idx": test_idx+1,
         "p_id": p_id[test_idx],
         "mrs": Y_mrs[test_idx],
         "unfavorable": Y_test
       }
    )
    
    results = pd.DataFrame(
        {"p_idx": p_idxx,
        "p_id": p_id,
         "mrs": Y_mrs,
         "unfavorable": Y_new 
        }
    )

    results2["status"] = np.where(results2["p_idx"].isin(train_idxx), "train",
                   np.where(results2["p_idx"].isin(valid_idxx), "valid",
                   np.where(results2["p_idx"].isin(test_idxx), "test", None)))

      
    return (train_idx, valid_idx, test_idx, X, X_train, X_valid, X_test, X_tab_train, X_tab_valid, X_tab_test), (Y_train, Y_valid, Y_test), results, results2














# Occlusion Heatmap Calculation:
# Calculates the heatmap for a given volume and models
# For each model, the heatmap is calculated and then averaged over all models
#
# Returns the heatmap, the original volume, the coordinates of the maximum heatmap 
#  slice and the standard deviation of the heatmaps
def volume_occlusion(volume, res_tab, 
                     occlusion_size, 
                     cnn, model_names,
                     normalize = True,
                     both_directions = False,
                     invert_hm = "pred_class",
                     model_mode = "mean",
                     occlusion_stride = None,
                     input_shape = (128,128,28,1)):
    # volume: np array in shape of input_shape
    # res_tab: dataframe with results of all models
    # occlusion_size: scalar or 3 element array, if scalar, occlusion is cubic
    # cnn: keras model
    # model_names: list of model names, to load weights
    # normalize: bool, if True, heatmap is normalized to [0,1] (after each model, and after averaging)
    # both_directions: bool, if True, heatmap is calculated for positive and negative prediction impact, if False,
    #           heatmap is cut off at the non-occluded prediction probability and only negative impact is shown
    # invert_hm: string, one of ["pred_class", "always", "never"], if "pred_class", heatmap is inverted if
    #           class 1 is predicted, if "always", heatmap is always inverted, if "never", heatmap is never inverted
    # model_mode: string, one of ["mean", "median", "max", "weighted"], defines how the heatmaps of the different models are combined
    # occlusion_stride: tuple, shape of input volume, if None, stride is set to minimum of occlusion_size
    # input_shape: tuple, shape of input volume
    
    ## Check input
    valid_modes = ["mean", "median", "max", "weighted"]
    if model_mode not in valid_modes:
        raise ValueError("volume_occlusion: model_mode must be one of %r." % valid_modes)
    
    valid_inverts = ["pred_class", "always", "never"]
    if invert_hm not in valid_inverts:
        raise ValueError("volume_occlusion: invert_hm must be one of %r." % valid_inverts)

    if not isinstance(model_names, list):
        model_names = [model_names]
    
    volume = volume.reshape(input_shape)
    
    if len(occlusion_size) == 1:
        occlusion_size = np.array([occlusion_size, occlusion_size, occlusion_size])
    elif len(occlusion_size) != 3:
        raise ValueError('occluson_size must be a scalar or a 3 element array')
  
    if occlusion_stride is None:
        occlusion_stride = np.repeat(np.min(occlusion_size),3)
    elif len(occlusion_stride) == 1:
        occlusion_stride = np.repeat(occlusion_stride,3)
       
  
    if np.any(occlusion_stride > occlusion_size):
        raise ValueError('stride must be smaller or equal size')
    
    if np.any(occlusion_stride == occlusion_size):
        if (not (volume.shape[0] / occlusion_size)[0].is_integer() or
            not (volume.shape[1] / occlusion_size)[1].is_integer() or 
            not (volume.shape[2] / occlusion_size)[2].is_integer()):
            
            raise ValueError('size does not work with this volume')
    elif np.any(occlusion_stride != occlusion_size):
        if (((volume.shape[0]-occlusion_size[0]) % occlusion_stride[0]) != 0 or 
            ((volume.shape[1]-occlusion_size[1]) % occlusion_stride[1]) != 0 or
            ((volume.shape[2]-occlusion_size[2]) % occlusion_stride[2]) != 0):
        
            raise ValueError('shape and size do not match')
    
    y_pred_class = "y_pred_class_avg"
    if model_mode == "weighted":
        weights = res_tab.loc[:, res_tab.columns.str.startswith("weight")].to_numpy().squeeze()
        y_pred_class += "_w"
        model_names = list(np.array(model_names)[weights > 0])
        weights = weights[weights>0] 

    ## loop over models
    h_l = []
    for model_name in model_names:
        cnn.load_weights(model_name)
        
        heatmap_prob_sum = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), np.float32)
        heatmap_occ_n = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), np.float32)
        
        ## Generate all possible occlusions
        X = []
        xyz = []
        for n, (x, y, z, vol_float) in enumerate(oc.iter_occlusion(
                volume, size = occlusion_size, stride = occlusion_stride)):
            X.append(vol_float.reshape(volume.shape[0], volume.shape[1], volume.shape[2], 1))
            xyz.append((x,y,z))
        
        X = np.array(X)
        out = 1-sigmoid(cnn.predict(X))
        out = out.squeeze()       

        ## Add predictions to heatmap and count number of predictions per voxel
        for i in range(len(xyz)):
            x,y,z = xyz[i]
            heatmap_prob_sum[x:x + occlusion_size[0], y:y + occlusion_size[1], z:z + occlusion_size[2]] += out[i]
            heatmap_occ_n[x:x + occlusion_size[0], y:y + occlusion_size[1], z:z + occlusion_size[2]] += 1

        hm = heatmap_prob_sum / heatmap_occ_n # calculate average probability per voxel
        
        ## Get cutoff, invert heatmap if necessary and normalize
        cut_off = res_tab["y_pred_model_" + model_name[-4:-3]][0]
    
        if (res_tab[y_pred_class][0] == 0 and invert_hm == "pred_class" and not both_directions) or (
            invert_hm == "never" and not both_directions): 
            hm[hm < cut_off] = cut_off
        elif (res_tab[y_pred_class][0] == 1 and invert_hm == "pred_class" and not both_directions) or (
            invert_hm == "always" and not both_directions):
            hm[hm > cut_off] = cut_off
        elif both_directions:
            hm = hm - cut_off
        
        if normalize and not both_directions:
            hm = ((hm - hm.min())/(hm.max()-hm.min()))
        elif normalize and both_directions:
            hm_min_max = [np.min(hm), np.max(hm)]
            hm_abs_max = np.max(np.abs(hm_min_max))
            hm = hm / hm_abs_max
        
        h_l.append(hm)
        
    ## Average over all models
    h_l = np.array(h_l)
    h_l = np.expand_dims(h_l, axis = -1)
    if model_mode == "mean":
        heatmap = np.mean(h_l, axis = 0)
    elif model_mode == "median":
        heatmap = np.median(h_l, axis = 0)
    elif model_mode == "max":
        heatmap = np.max(h_l, axis = 0)
    elif model_mode == "weighted":
        heatmap = np.average(h_l, axis=0, weights=weights)
        
    if normalize and not both_directions:
        heatmap = ((heatmap - heatmap.min())/(heatmap.max()-heatmap.min()))
    elif normalize and both_directions:
        heatmap_min_max = [np.min(heatmap), np.max(heatmap)]
        heatmap_abs_max = np.max(np.abs(heatmap_min_max))
        heatmap = heatmap / heatmap_abs_max
        
    if invert_hm == "pred_class" and res_tab[y_pred_class][0] == 1:
        heatmap = 1 - heatmap  
    elif invert_hm == "always":
        heatmap = 1 - heatmap
        
    ## Get maximum heatmap slice and standard deviation of heatmaps
    target_shape = h_l.shape[:-1]
    max_hm_slice = np.array(np.unravel_index(h_l.reshape(target_shape).reshape(len(h_l), -1).argmax(axis = 1), 
                                             h_l.reshape(target_shape).shape[1:])).transpose()
    hm_mean_std = np.sqrt(np.mean(np.var(h_l, axis = 0)))
    
    return heatmap, volume, max_hm_slice, hm_mean_std

# generates a occlusion heatmap plot/slider for a given patient id
# first row shows the average heatmap over each direction
# second row shows a slider to select the heatmap slice (default is the maximum heatmap slice)
# third row shows the original image with the same slider as in the second row
# additionally some meta information is printed
# 
# all heatmaps can be provided in the same order as X_in or will be generated if None
# if None then the model arguments are required 
# if the heatmap is provided then pred_hm_only does only change the colorbar

def occlusion_interactive_plot(p_id, occ_size, occ_stride,
                               cnn, all_results, pat, X_in,
                               generate_model_name, num_models,
                               pat_dat,
                               pred_hm_only=True,
                               heatmaps = None):
    # p_id: patient id
    # occ_size: size of the occlusion window
    # occ_stride: stride of the occlusion window (if None then occ_stride = occ_size)
    # cnn: the cnn model (weights must not be loaded)
    # all_results: the result table 
    # pat: the patient ids of the images (same order as X_in)
    # X_in: the images (same order as pat)
    # generate_model_name: function to generate the model names
    # num_models: number of models per fold
    # pat_dat: the patient data table
    # pred_hm_only: if True then the heatmap is only plotted for the predicted class
    #               if False then the positive and negative heatmap is plotted
    # heatmaps: if None then the heatmaps are generated, otherwise the heatmaps must be provided (same order as X_in)
    
    p_ids = [p_id]
    (res_table, res_images, res_model_names) = gc.get_img_and_models(
        p_ids, results = all_results, pats = pat, imgs = X_in, 
        gen_model_name = generate_model_name,
        num_models = num_models)
    
    print("patient id: ", res_table.p_id[0])
    print("age: ", pat_dat[pat_dat["p_id"] == res_table.p_id[0]]["age"].values[0])
    print("true mrs: ", res_table.mrs[0])
    print("true class: ", res_table.unfavorable[0])
    print(colored("pred class: "+str(res_table.y_pred_class[0]), 
                'green' if res_table["pred_correct"][0] == True else 'red'))
    print("pred prob (class 1): ", res_table.y_pred_trafo_avg[0])
    print("Threshold: ", res_table.threshold[0])
    print("pred uncertainty: ", res_table.y_pred_unc[0])
    # print("heatmap unc. last layer: ", res_table.y_pred_unc[0])
      
    ## Generate heatmap
    if pred_hm_only:
        invert_hm = "pred_class"
        both_directions = False
        cmap = "jet"
        hm_positive=True
    else:
        invert_hm = "never"
        both_directions = True
        cmap = "bwr"
        hm_positive=False   
    
    if heatmaps is None:
        (heatmap, resized_img, max_hm_slice, hm_mean_std) =  volume_occlusion(
            volume = res_images, 
            res_tab = res_table, 
            occlusion_size = np.array(occ_size), 
            cnn = cnn,
            invert_hm=invert_hm,
            both_directions=both_directions,
            model_names = res_model_names[0],
            occlusion_stride = occ_stride)
    else:
        heatmap = heatmaps[np.argwhere(pat == p_id).squeeze()]
        resized_img = res_images[0]
    
    slices = np.unravel_index(heatmap.argmax(), heatmap.shape)
    print("max slices:", (slices[2], slices[0], slices[1]))
    
    ## Plot Heatmap Average
    phm.plot_heatmap(np.squeeze(resized_img, axis=-1), np.squeeze(heatmap, axis=-1),
                version = "overlay",
                mode = "avg",
                hm_colormap=cmap,
                hm_positive=hm_positive,
                colorbar=True)

    ## Plot Heatmap Slider
    def slicer(axi_slider, cor_slider, sag_slider):
        phm.plot_heatmap(np.squeeze(resized_img, axis=-1), np.squeeze(heatmap, axis=-1),
                version = "overlay",
                mode = "def",
                slices = (cor_slider,sag_slider,axi_slider),
                hm_colormap=cmap,
                hm_positive=hm_positive,
                colorbar=True)
        phm.plot_heatmap(np.squeeze(resized_img, axis=-1), np.squeeze(heatmap, axis=-1),
                version = "original",
                mode = "def",
                slices=(cor_slider,sag_slider,axi_slider),
                hm_colormap=cmap,
                hm_positive=hm_positive,
                slice_line=True)

    w=interactive(
        slicer, 
        axi_slider=widgets.IntSlider(value=slices[2],min=0,max=27,step=1), 
        cor_slider=widgets.IntSlider(value=slices[0],min=0,max=127,step=1), 
        sag_slider=widgets.IntSlider(value=slices[1],min=0,max=127,step=1))

    slider_layout = Layout(display='flex', flex_flow='row', 
                        justify_content='space-between', align_items='center',
                        width='9.2in')
    images_layout = Layout(display='flex', flex_flow='row', 
                        justify_content='space-between', align_items='center',
                        width='15', height='15')

    display(VBox([
        HBox([w.children[0],w.children[1], w.children[2]], layout=slider_layout),
        HBox([w.children[3]], layout=images_layout)
    ]))      
    w.update()





##########################################################################################

def volume_occlusion_tabular(volume, res_tab, tabular_df,
                     occlusion_size, 
                     model_names,
                     normalize = True,
                     both_directions = False,
                     invert_hm = "pred_class",
                     model_mode = "mean",
                     occlusion_stride = None,
                     input_shape = (128,128,28,1)):
    # volume: np array in shape of input_shape
    # res_tab: dataframe with results of all models
    # occlusion_size: scalar or 3 element array, if scalar, occlusion is cubic
    # cnn: keras model
    # model_names: list of model names, to load weights
    # normalize: bool, if True, heatmap is normalized to [0,1] (after each model, and after averaging)
    # both_directions: bool, if True, heatmap is calculated for positive and negative prediction impact, if False,
    #           heatmap is cut off at the non-occluded prediction probability and only negative impact is shown
    # invert_hm: string, one of ["pred_class", "always", "never"], if "pred_class", heatmap is inverted if
    #           class 1 is predicted, if "always", heatmap is always inverted, if "never", heatmap is never inverted
    # model_mode: string, one of ["mean", "median", "max"], defines how the heatmaps of the different models are combined
    # occlusion_stride: scalar, stride of occlusion, if None, stride is set to minimum of occlusion_size
    # input_shape: tuple, shape of input volume
    
    ## Check input
    valid_modes = ["mean", "median", "max", "weighted"]
    if model_mode not in valid_modes:
        raise ValueError("volume_occlusion: model_mode must be one of %r." % valid_modes)
    
    valid_inverts = ["pred_class", "always", "never"]
    if invert_hm not in valid_inverts:
        raise ValueError("volume_occlusion: invert_hm must be one of %r." % valid_inverts)

    if not isinstance(model_names, list):
        model_names = [model_names]
    
    volume = volume.reshape(input_shape)
    
    if len(occlusion_size) == 1:
        occlusion_size = np.array([occlusion_size, occlusion_size, occlusion_size])
    elif len(occlusion_size) != 3:
        raise ValueError('occluson_size must be a scalar or a 3 element array')
  
    if occlusion_stride is None:
        occlusion_stride = np.repeat(np.min(occlusion_size),3)
    elif len(occlusion_stride) == 1:
        occlusion_stride = np.repeat(occlusion_stride,3)
       
  
    if np.any(occlusion_stride > occlusion_size):
        raise ValueError('stride must be smaller or equal size')
    
    if np.any(occlusion_stride == occlusion_size):
        if (not (volume.shape[0] / occlusion_size)[0].is_integer() or
            not (volume.shape[1] / occlusion_size)[1].is_integer() or 
            not (volume.shape[2] / occlusion_size)[2].is_integer()):
            
            raise ValueError('size does not work with this volume')
    elif np.any(occlusion_stride != occlusion_size):
        if (((volume.shape[0]-occlusion_size[0]) % occlusion_stride[0]) != 0 or 
            ((volume.shape[1]-occlusion_size[1]) % occlusion_stride[1]) != 0 or
            ((volume.shape[2]-occlusion_size[2]) % occlusion_stride[2]) != 0):
        
            raise ValueError('shape and size do not match')
    
    y_pred_class = "y_pred_class_avg"
    if model_mode == "weighted":
        weights = res_tab.loc[:, res_tab.columns.str.startswith("weight")].to_numpy().squeeze()
        y_pred_class += "_w"
        model_names = model_names[weights>0]
        weights = weights[weights>0] 

    ## loop over models
    h_l = []

    for model_name in model_names:
        input_dim = (128, 128, 28, 1)
        output_dim = 1
        batch_size = 6
        C = 2 

        mbl = img_model_linear_final(input_dim, output_dim)
        mls = mod_linear_shift(13)
        cnn = ontram(mbl, mls)             

        cnn.compile(optimizer=keras.optimizers.Adam(learning_rate=5*1e-5),
                                        loss=ontram_loss(C, batch_size),
                                        metrics=[ontram_acc(C, batch_size)])

        cnn.load_weights(model_name)
        
        heatmap_prob_sum = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), np.float32)
        heatmap_occ_n = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), np.float32)
        
        ## Generate all possible occlusions
        X = []
        xyz = []
        for n, (x, y, z, vol_float) in enumerate(oc.iter_occlusion(
                volume, size = occlusion_size, stride = occlusion_stride)):
            X.append(vol_float.reshape(volume.shape[0], volume.shape[1], volume.shape[2], 1))
            xyz.append((x,y,z))
        
        X = np.array(X)

        filtered_df = tabular_df[tabular_df['patient_id'] == res_tab['p_id'][0]].drop('patient_id', axis=1).values
        X_tab_occ = np.tile(filtered_df, (len(X), 1))

        occ_dataset_pred = ((X, X_tab_occ))
        preds = cnn.predict(occ_dataset_pred)
        out = 1-sigmoid(preds[:,0]-preds[:,1])
                
        ## Add predictions to heatmap and count number of predictions per voxel
        for i in range(len(xyz)):
            x,y,z = xyz[i]
            heatmap_prob_sum[x:x + occlusion_size[0], y:y + occlusion_size[1], z:z + occlusion_size[2]] += out[i]
            heatmap_occ_n[x:x + occlusion_size[0], y:y + occlusion_size[1], z:z + occlusion_size[2]] += 1

        hm = heatmap_prob_sum / heatmap_occ_n # calculate average probability per voxel
        
        ## Get cutoff, invert heatmap if necessary and normalize
        cut_off = res_tab["y_pred_model_" + model_name[-4:-3]][0]
                   
        if (res_tab[y_pred_class][0] == 0 and invert_hm == "pred_class" and not both_directions) or (
            invert_hm == "never" and not both_directions): 
            hm[hm < cut_off] = cut_off
        elif (res_tab[y_pred_class][0] == 1 and invert_hm == "pred_class" and not both_directions) or (
            invert_hm == "always" and not both_directions):
            hm[hm > cut_off] = cut_off
              
        
        elif both_directions:
            hm = hm - cut_off
        
        if normalize and not both_directions:
            hm = ((hm - hm.min())/(hm.max()-hm.min()))
        elif normalize and both_directions:
            hm_min_max = [np.min(hm), np.max(hm)]
            hm_abs_max = np.max(np.abs(hm_min_max))
            hm = hm / hm_abs_max
        
        h_l.append(hm)
        
    ## Average over all models
    h_l = np.array(h_l)
    h_l = np.expand_dims(h_l, axis = -1)
    if model_mode == "mean":
        heatmap = np.mean(h_l, axis = 0)
    elif model_mode == "median":
        heatmap = np.median(h_l, axis = 0)
    elif model_mode == "max":
        heatmap = np.max(h_l, axis = 0)

    elif model_mode == "weighted":
        heatmap = np.average(h_l, axis=0, weights=weights)

        
    if normalize and not both_directions:
        heatmap = ((heatmap - heatmap.min())/(heatmap.max()-heatmap.min()))
    elif normalize and both_directions:
        heatmap_min_max = [np.min(heatmap), np.max(heatmap)]
        heatmap_abs_max = np.max(np.abs(heatmap_min_max))
        heatmap = heatmap / heatmap_abs_max
        
    if invert_hm == "pred_class" and res_tab[y_pred_class][0] == 1:
        heatmap = 1 - heatmap  
    elif invert_hm == "always":
        heatmap = 1 - heatmap
        
    ## Get maximum heatmap slice and standard deviation of heatmaps
    target_shape = h_l.shape[:-1]
    max_hm_slice = np.array(np.unravel_index(h_l.reshape(target_shape).reshape(len(h_l), -1).argmax(axis = 1), 
                                             h_l.reshape(target_shape).shape[1:])).transpose()
    hm_mean_std = np.sqrt(np.mean(np.var(h_l, axis = 0)))
    
    return heatmap, volume, max_hm_slice, hm_mean_std
