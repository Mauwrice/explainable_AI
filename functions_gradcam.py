import tensorflow as tf
import numpy as np

from skimage.transform import resize


### GradCam for specific layer
def grad_cam_3d(img, model_3d, layer, pred_index=None, 
                inv_hm=False, gcplusplus=True):
    # img: 3d image
    # model_3d: 3d cnn model with loaded weights
    # layer: layer name of model_3d where gradcam should be applied
    # pred_index: output channel when sigmoid should always be 0, when softmax 0 favorable, 1 unfavorable
    # inv_hm: invert heatmap
    # gcplusplus: use gradcam++ instead of gradcam (only positive values)
    
    # First, we create a MODEL that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    # Match correct output layer of the CNN part (dependent on model architecture)
    if model_3d.name == "cnn_3d_":
        grad_model = tf.keras.models.Model([model_3d.inputs], 
            [model_3d.get_layer(layer).output, model_3d.output])
    elif model_3d.name == "mod_ontram":
        grad_model = tf.keras.models.Model([model_3d.inputs], 
            [model_3d.get_layer(layer).output, model_3d.get_layer("dense_complex_intercept").output])
    
    # Then, we compute the GRADIENT for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        # check for right model variant
        if pred_index is None and model_3d.name == "mod_ontram":
            pred_index = 0
        elif pred_index is None or model_3d.layers[-1].get_config()["activation"] == "sigmoid":
                pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index] # when sigmoid, pred_index must be None or 0

    # This is the gradient of the output neuron
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)[0]
    
    # Average gradients spatially
    weights = tf.reduce_mean(grads, axis=(0, 1, 2)) # pooled grads
    
    # Build a ponderated map of filters according to gradients importance
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    output = conv_outputs[0]    
    
    # slower implementation:
#     cam = np.zeros(output.shape[0:3], dtype=np.float32)
#     for index, w in enumerate(weights):
#         cam += w * output[:, :, :, index]

#     capi=resize(cam,(img.shape[1:]))
#     capi = np.maximum(capi,0)
#     heatmap = (capi - capi.min()) / (capi.max() - capi.min())
#     resized_img = img.reshape(img.shape[1:])

    # faster implementation:
    heatmap = output @ weights[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = resize(heatmap, img.shape[1:])
    
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    if inv_hm:
        heatmap = heatmap * (-1)
    
    if gcplusplus:
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    else:
        # normalize heatmap between -1 and 1 (absolute max is then -1 or 1)
        heatmap_min_max = [tf.math.reduce_min(heatmap), tf.math.reduce_max(heatmap)]
        heatmap_abs_max = tf.math.reduce_max(tf.math.abs(heatmap_min_max))
        heatmap = heatmap / heatmap_abs_max
    heatmap = heatmap.numpy()
    resized_img = img.reshape(img.shape[1:])
    
    return heatmap, resized_img


### GradCam for multiple layers (mean, median or max of all layers can be used)
def multi_layers_grad_cam_3d(img, model_3d, layers, mode = "mean", 
                             normalize = True, pred_index=None, 
                             invert_hm="none", gcpp_hm="all"):
    # img: 3d image
    # model_3d: 3d cnn model with loaded weights
    # layers: list of layer names of model_3d where gradcam should be applied
    # mode: mean, median or max, how to combine the heatmaps of the layers
    # normalize: normalize heatmaps between 0 and 1
    # pred_index: output channel when sigmoid should always be 0, when softmax 0 favorable, 1 unfavorable
    # invert_hm: one of "none", "all", "last", 
    #      if "all" then all heatmaps are inverted, if "last" then only last heatmap is inverted
    # gcpp_hm: one of "all", "none", "last", 
    #      if "all" then all heatmaps are positive, if "none" then all heatmaps are negative,
    #      if "last" then only last heatmap is positive (gradcam++)
    
    
    ## Check input
    valid_modes = ["mean", "median", "max"]
    if mode not in valid_modes:
        raise ValueError("multi_layers_grad_cam_3d: mode must be one of %r." % valid_modes)
    valid_hm_inverts = ["none", "all", "last"]
    if invert_hm not in valid_hm_inverts:
        raise ValueError("multi_layers_grad_cam_3d: invert_hm must be one of %r." % valid_hm_inverts)
    valid_hm_gcpp = ["all", "none", "last"]
    # all: all layers use gradcam++, 
    # none: no layer uses gradcam++, 
    # last: only last layer uses gradcam++
    if gcpp_hm not in valid_hm_gcpp:
        raise ValueError("multi_layers_grad_cam_3d: gcpp_hm must be one of %r." % valid_hm_gcpp)
        
    if not isinstance(layers, list):
        layers = [layers]
    
    ## Apply gradcam to all layers
    h_l = []
    for i, layer in enumerate(layers):
        
        # check if gradcam++ should be used
        gcpp = True
        if (i != len(layers)-1 and gcpp_hm == "last") or gcpp_hm == "none":
            gcpp = False 
        
        if (i == len(layers)-1 and invert_hm == "last") or invert_hm == "all":
            heatmap, resized_img = grad_cam_3d(
                img = img, model_3d = model_3d , layer = layer, 
                pred_index=pred_index, inv_hm=True, gcplusplus=gcpp)
        else:
            heatmap, resized_img = grad_cam_3d(
                img = img, model_3d = model_3d , layer = layer, 
                pred_index=pred_index, inv_hm=False, gcplusplus=gcpp)
        h_l.append(heatmap)
        
    h_l = np.array(h_l)
    
    if gcpp_hm == "last":
        # if all, then everything is already positive
        # if none, then absolute is not applied
        h_l = np.abs(h_l)
    
    ## Combine heatmaps
    if mode == "mean":
        heatmap = np.mean(h_l, axis = 0)
    elif mode == "median":
        heatmap = np.median(h_l, axis = 0)
    elif mode == "max":
        heatmap = np.max(h_l, axis = 0) 
        
    ## Normalize heatmap
    if normalize and gcpp_hm in ["last", "all"] and heatmap.max() != 0:
        heatmap = ((heatmap - heatmap.min())/(heatmap.max()-heatmap.min()))
    elif normalize and gcpp_hm == "none" and heatmap.max() != 0:
        heatmap_min_max = [tf.math.reduce_min(heatmap), tf.math.reduce_max(heatmap)]
        heatmap_abs_max = tf.math.reduce_max(tf.math.abs(heatmap_min_max))
        heatmap = heatmap / heatmap_abs_max
        heatmap = heatmap.numpy()
    elif not normalize:
        raise ValueError("Something went wrong with normalization in multi_layers_grad_cam_3d")
    
    return (heatmap, resized_img)

# Applies grad cams to multiple models (and layers)
#
# Returns combined heatmap, the original image, the slice with the highest 
#   heatmap value and the mean std of the heatmaps
def multi_models_grad_cam_3d(img, cnn, model_names, layers, 
                             model_mode = "mean", layer_mode = "mean", 
                             normalize = True, pred_index=None,
                             invert_hm="none", gcpp_hm="last"):
    # img: 3d image
    # cnn: 3d cnn model without loaded weights
    # model_names: list of model names (with path) of cnn, to load weights
    # layers: list of layer names of model_3d where gradcam should be applied
    # model_mode: mean, median or max, how to combine the heatmaps of the models
    # layer_mode: mean, median or max, how to combine the heatmaps of the layers
    # normalize: normalize heatmaps between 0 and 1
    # pred_index: output channel when sigmoid should always be 0, when softmax 0 favorable, 1 unfavorable
    # invert_hm: one of "none", "all", "last",
    #     if "all" then all heatmaps are inverted, if "last" then only last heatmap is inverted
    # gcpp_hm: one of "all", "none", "last", 
    #     if "all" then all heatmaps are positive, if "none" then all heatmaps are negative,
    #     if "last" then only last heatmap is positive (gradcam++)
    
    ## Check input
    valid_modes = ["mean", "median", "max"]
    if model_mode not in valid_modes:
        raise ValueError("multi_models_grad_cam_3d: model_mode must be one of %r." % valid_modes)
        
    if not isinstance(layers, list):
        layers = [layers]
    
    ## Load weights and apply gradcam to all models
    h_l = []
    for model_name in model_names:
        cnn.load_weights(model_name)
        heatmap, resized_img = multi_layers_grad_cam_3d(
            img = img, model_3d = cnn , layers = layers, mode = layer_mode, normalize = normalize, 
            pred_index=pred_index, invert_hm=invert_hm, gcpp_hm=gcpp_hm)
        h_l.append(heatmap)
    
    ## Combine heatmaps of all models
    h_l = np.array(h_l)
    if model_mode == "mean":
        heatmap = np.mean(h_l, axis = 0)
    elif model_mode == "median":
        heatmap = np.median(h_l, axis = 0)
    elif model_mode == "max":
        heatmap = np.max(h_l, axis = 0)
        
    ## Normalize heatmap
    if normalize and gcpp_hm in ["last", "all"] and heatmap.max() != 0:
        heatmap = ((heatmap - heatmap.min())/(heatmap.max()-heatmap.min()))
    elif normalize and gcpp_hm == "none" and heatmap.max() != 0:
        heatmap_min_max = [tf.math.reduce_min(heatmap), tf.math.reduce_max(heatmap)]
        heatmap_abs_max = tf.math.reduce_max(tf.math.abs(heatmap_min_max))
        heatmap = heatmap / heatmap_abs_max
        heatmap = heatmap.numpy()
    elif not normalize:
        raise ValueError("Something went wrong with normalization in multi_models_grad_cam_3d")
        
    ## Extract max slice and mean std of heatmap
    target_shape = h_l.shape[:-1]
    max_hm_slice = np.array(np.unravel_index(h_l.reshape(target_shape).reshape(len(h_l), -1).argmax(axis = 1), 
                                             h_l.reshape(target_shape).shape[1:])).transpose()
    hm_mean_std = np.sqrt(np.mean(np.var(h_l, axis = 0)))
        
    return heatmap, resized_img, max_hm_slice, hm_mean_std


# Prepares data in order to use multi_models_grad_cam_3d and volume_occlusion
#
# Returns a dataframe with all results, an array of the images and a list of model names 
#  for all patients in p_ids
def get_img_and_models(p_ids, results, pats, imgs, gen_model_name, num_models = 5):
    # p_ids: list of patient ids
    # results: dataframe with results of all patients
    # pats: array with patient ids of all images (must be same order as imgs!!!)
    # imgs: array with all images (must be same order as pats!!!)
    # gen_model_name: function that generates model names for a given test split and model number
    # num_models: number of models per test split
    
    imgs = np.expand_dims(imgs, axis = -1)
    
    # extract table with all matches for p_ids
    res_tab = results[results.p_id.isin(p_ids)].sort_values(["p_id", "test_split"]).reset_index()
    
    res_imgs = []
    res_mod_names = []
    
    res_test_splits = list(res_tab.test_split)
    for i, p_id in enumerate(list(res_tab.p_id)):
        index = np.argwhere(pats == p_id).squeeze()
        res_imgs.append(imgs[index])
        
        # generate model names
        res_mod_names.append(
            [gen_model_name(res_test_splits[i], j) for j in range(num_models)]
        )
            
    return (res_tab, np.array(res_imgs), res_mod_names)

  
    