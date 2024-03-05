import numpy as np
import tensorflow as tf
import functions_metrics as fm


# Occlusion Iteration: 
# Generatess all possible occlusions of a given size and stride for a given volume
# stride must be smaller or equal size
#
# Returns the starting coordinates of the occlusion and the occluded volume
def iter_occlusion(volume, size, stride):
    # volume: np array in shape 128, 128, 64, 1
    # size: 3 element array or tuple
    # stride: 3 element array or tuple

    occlusion_center = np.full((size[0], size[1], size[2], 1), [0.5], np.float32)

    for x in range(0, volume.shape[0]-size[0]+1, stride[0]):
        for y in range(0, volume.shape[1]-size[1]+1, stride[1]):
            for z in range(0, volume.shape[2]-size[2]+1, stride[2]):
                tmp = volume.copy()

                tmp[x:x + size[0], y:y + size[1], z:z + size[2]] = occlusion_center

                yield x, y, z, tmp


# Occlusion Heatmap Calculation:
# Calculates the heatmap for a given volume and models
# For each model, the heatmap is calculated and then averaged over all models
#
# Returns the heatmap, the original volume, the coordinates of the maximum heatmap 
#  slice and the standard deviation of the heatmaps
def volume_occlusion(volume, res_tab,
                     occlusion_size, 
                     cnn, model_names,
                     tabular_df = None,
                     normalize = True,
                     both_directions = False,
                     invert_hm = "pred_class",
                     model_mode = "mean",
                     occlusion_stride = None,
                     input_shape = (128,128,28,1),
                     reset_cut_off = False):
    # volume: np array in shape of input_shape
    # res_tab: dataframe with results of all models
    # tabular_df: dataframe with normalized tabular data, is only needed when models use tabular data
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
        model_names = list(np.array(model_names)[weights > 0])
        weights = weights[weights>0] 
  
    ## loop over models
    h_l = []
    for model_name in model_names:
        cnn.load_weights(model_name)
        
        heatmap_prob_sum = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), np.float32)
        heatmap_occ_n = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), np.float32)

        # for n, (x, y, z, vol_float) in tqdm.tqdm(enumerate(iter_occlusion(volume, size = occlusion_size, stride = occlusion_stride))):
        #     X = vol_float.reshape(1, volume.shape[0], volume.shape[1], volume.shape[2], 1)
        #     out = model.predict(X)

        #     heatmap_prob_sum[x:x + occlusion_size[0], y:y + occlusion_size[1], z:z + occlusion_size[2]] += out[0]
        #     heatmap_occ_n[x:x + occlusion_size[0], y:y + occlusion_size[1], z:z + occlusion_size[2]] += 1

        ## Faster Implementation
        
        ## Generate all possible occlusions
        X = []
        xyz = []
        for n, (x, y, z, vol_float) in enumerate(iter_occlusion(
                volume, size = occlusion_size, stride = occlusion_stride)):
            X.append(vol_float.reshape(volume.shape[0], volume.shape[1], volume.shape[2], 1))
            xyz.append((x,y,z))
        
        X = np.array(X)
        if "ontram" in cnn.name and not isinstance(cnn.input, list):
            out = 1-fm.sigmoid(cnn.predict(X))
        elif "ontram" in cnn.name and isinstance(cnn.input, list):  
            filtered_df = tabular_df[tabular_df['p_id'] == res_tab['p_id'][0]].drop('p_id', axis=1).values
            X_tab_occ = np.tile(filtered_df, (len(X), 1))

            occ_dataset_pred = ((X, X_tab_occ))
            preds = cnn.predict(occ_dataset_pred)
            out = 1-fm.sigmoid(preds[:,0]-preds[:,1])
        else:
            out = cnn.predict(X) # do prediction for all occlusions at once 

        out = out.squeeze()
        
        ## Add predictions to heatmap and count number of predictions per voxel
        for i in range(len(xyz)):
            x,y,z = xyz[i]
            heatmap_prob_sum[x:x + occlusion_size[0], y:y + occlusion_size[1], z:z + occlusion_size[2]] += out[i]
            heatmap_occ_n[x:x + occlusion_size[0], y:y + occlusion_size[1], z:z + occlusion_size[2]] += 1

        hm = heatmap_prob_sum / heatmap_occ_n # calculate average probability per voxel
        
        ## Get cutoff, invert heatmap if necessary and normalize
        cut_off = res_tab["y_pred_model_" + model_name[-4:-3]][0]
        if (reset_cut_off):
            occ_dataset_pred = ((np.expand_dims(volume, axis=0), filtered_df))
            preds = cnn.predict(occ_dataset_pred)
            cut_off = 1-fm.sigmoid(preds[:,0]-preds[:,1])
    
        hm = hm - cut_off
        if (res_tab[y_pred_class][0] == 0 and invert_hm == "pred_class" and not both_directions) or (
            invert_hm == "never" and not both_directions): 
            hm[hm < 0] = 0
        elif (res_tab[y_pred_class][0] == 1 and invert_hm == "pred_class" and not both_directions) or (
            invert_hm == "always" and not both_directions):
            hm[hm > 0] = 0
        
        if normalize:
            hm = fm.normalize_heatmap(hm, both_directions=both_directions)
        
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
        
    if normalize:
        heatmap = fm.normalize_heatmap(heatmap, both_directions=both_directions)
        
    # invert at the end else inversion is done on unnormalized heatmap
    if ((invert_hm == "pred_class" and res_tab[y_pred_class][0] == 1 and normalize) or
        (invert_hm == "always" and normalize)):
        heatmap = 1 - heatmap  
    elif (both_directions or 
          (invert_hm == "pred_class" and res_tab[y_pred_class][0] == 1 and not normalize) or
          (invert_hm == "always" and not normalize)): 
        # inversion so interpretation is same as gradcam (positive heatmap => unfavorable, neg hm => favorable)
        heatmap = heatmap * -1
        
    ## Get maximum heatmap slice and standard deviation of heatmaps
    target_shape = h_l.shape[:-1]
    max_hm_slice = np.array(np.unravel_index(h_l.reshape(target_shape).reshape(len(h_l), -1).argmax(axis = 1), 
                                             h_l.reshape(target_shape).shape[1:])).transpose()
    if model_mode == "weighted":
        hm_mean_std = np.sqrt(np.mean(fm.wght_variance(h_l, weights = weights, axis = 0)))
    else:
        hm_mean_std = np.sqrt(np.mean(np.var(h_l, axis = 0)))
    
    return heatmap, volume, max_hm_slice, hm_mean_std, h_l

