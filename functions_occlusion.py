import numpy as np

# Occlusion Iteration: 
# Generatess all possible occlusions of a given size and stride for a given volume
# stride must be smaller or equal size
#
# Returns the starting coordinates of the occlusion and the occluded volume
def iter_occlusion(volume, size, stride):
    # volume: np array in shape 128, 128, 64, 1
    # size: 3 element array or tuple
    # stride: scalar

    occlusion_center = np.full((size[0], size[1], size[2], 1), [0.5], np.float32)

    for x in range(0, volume.shape[0]-size[0]+1, stride):
        for y in range(0, volume.shape[1]-size[1]+1, stride):
            for z in range(0, volume.shape[2]-size[2]+1, stride):
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
    valid_modes = ["mean", "median", "max"]
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
        occlusion_stride = np.min(occlusion_size)
    elif any(occlusion_stride > occlusion_size):
        raise ValueError('stride must be smaller or equal size')
    
    if any(occlusion_stride == occlusion_size):
        if (not (volume.shape[0] / occlusion_size)[0].is_integer() or
            not (volume.shape[1] / occlusion_size)[1].is_integer() or 
            not (volume.shape[2] / occlusion_size)[2].is_integer()):
            
            raise ValueError('size does not work with this volume')
    elif any(occlusion_stride != occlusion_size):
        if (((volume.shape[0]-occlusion_size[0]) % occlusion_stride) != 0 or 
            ((volume.shape[1]-occlusion_size[1]) % occlusion_stride) != 0 or
            ((volume.shape[2]-occlusion_size[2]) % occlusion_stride) != 0):
        
            raise ValueError('shape and size do not match')

    # num_occlusion =  int(np.prod(((np.array(volume.shape[0:3]) - occlusion_size) / occlusion_stride) + 1))
    # print('number of occlusions per model: ', num_occlusion)
    
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
        out = cnn.predict(X) # do prediction for all occlusions at once 
        
        ## Add predictions to heatmap and count number of predictions per voxel
        for i in range(len(xyz)):
            x,y,z = xyz[i]
            heatmap_prob_sum[x:x + occlusion_size[0], y:y + occlusion_size[1], z:z + occlusion_size[2]] += out[i,0]
            heatmap_occ_n[x:x + occlusion_size[0], y:y + occlusion_size[1], z:z + occlusion_size[2]] += 1

        hm = heatmap_prob_sum / heatmap_occ_n # calculate average probability per voxel
        
        ## Get cutoff, invert heatmap if necessary and normalize
        cut_off = res_tab["y_pred_model_" + model_name[-5:-3]][0]
    
        if (res_tab["y_pred_class"][0] == 0 and invert_hm == "pred_class" and not both_directions) or (
            invert_hm == "never" and not both_directions): 
            hm[hm < cut_off] = cut_off
        elif (res_tab["y_pred_class"][0] == 1 and invert_hm == "pred_class" and not both_directions) or (
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
        
    if normalize and not both_directions:
        heatmap = ((heatmap - heatmap.min())/(heatmap.max()-heatmap.min()))
    elif normalize and both_directions:
        heatmap_min_max = [np.min(heatmap), np.max(heatmap)]
        heatmap_abs_max = np.max(np.abs(heatmap_min_max))
        heatmap = heatmap / heatmap_abs_max
        
    if invert_hm == "pred_class" and res_tab["y_pred_class"][0] == 1:
        heatmap = 1 - heatmap  
    elif invert_hm == "always":
        heatmap = 1 - heatmap
        
    ## Get maximum heatmap slice and standard deviation of heatmaps
    target_shape = h_l.shape[:-1]
    max_hm_slice = np.array(np.unravel_index(h_l.reshape(target_shape).reshape(len(h_l), -1).argmax(axis = 1), 
                                             h_l.reshape(target_shape).shape[1:])).transpose()
    hm_mean_std = np.sqrt(np.mean(np.var(h_l, axis = 0)))
    
    return heatmap, volume, max_hm_slice, hm_mean_std

