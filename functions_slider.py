from __future__ import print_function

import numpy as np
import ipywidgets as widgets

from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, Layout, AppLayout
from IPython.display import display
from termcolor import colored

import functions_gradcam as gc
import functions_occlusion as oc
import functions_plot_heatmap as phm

# generates a gradcam heatmap plot/slider for a given patient id
# first row shows the average heatmap over each direction
# second row shows a slider to select the heatmap slice (default is the maximum heatmap slice)
# third row shows the original image with the same slider as in the second row
# additionally some meta information is printed
# 
# all heatmaps can be provided in the same order as X_in or will be generated if None
# if None then the model arguments are required 
# if the heatmap is provided then pred_hm_only does only change the colorbar
def gradcam_interactive_plot(p_id, vis_layers,
                             cnn, all_results, pat, X_in,
                             generate_model_name, num_models,
                             pat_dat,
                             pred_hm_only=True, 
                             y_pred_cl = "y_pred_class_avg",
                             normalize_hm = True,
                             model_mode = "mean",
                             heatmaps = None):
    # p_id: patient id
    # vis_layers: the layers for which the heatmap is generated, should be last layer
    # cnn: the cnn model (weights must not be loaded)
    # all_results: the result table 
    # pat: the patient ids of the images (same order as X_in)
    # X_in: the images (same order as pat)
    # generate_model_name: function to generate the model names
    # num_models: number of models per fold
    # pat_dat: the patient data table
    # pred_hm_only: if True then the heatmap is only plotted for the predicted class
    #               if False then the positive and negative heatmap is plotted
    # y_pred_cl: the column name of the predicted class
    # normalize_hm: if True then the heatmap is normalized
    # model_mode: the mode for the model (mean, median, max or weighted)
    # heatmaps: if None then the heatmaps are generated, otherwise the heatmaps must be provided (same order as X_in)
    
    p_ids = [p_id]
    (res_table, res_images, res_model_names) = gc.get_img_and_models(
        p_ids, results = all_results, pats = pat, imgs = X_in, 
        gen_model_name = generate_model_name,
        num_models = num_models)
    
    if model_mode == "weighted":
        y_pred_cl = "y_pred_class_avg_w"
        pred_co = "pred_correct_w"
        y_pred_prob = "y_pred_trafo_avg_w"
        y_pred_u = "y_pred_unc_w"
    else:
        y_pred_cl = "y_pred_class_avg"
        pred_co = "pred_correct"
        y_pred_prob = "y_pred_trafo_avg"
        y_pred_u = "y_pred_unc"

    
    print("patient id: ", res_table.p_id[0])
    print("age: ", pat_dat[pat_dat["p_id"] == res_table.p_id[0]]["age"].values[0])
    print("true mrs: ", res_table.mrs[0])
    print("true class: ", res_table.unfavorable[0])
    print(colored("pred class: "+str(res_table[y_pred_cl][0]), 
                'green' if res_table[pred_co][0] == True else 'red'))
    print("pred prob (class 1): ", res_table[y_pred_prob][0])
    print("pred uncertainty: ", res_table[y_pred_u][0])
    
    ## Generate heatmap
    if pred_hm_only:
        invert_hm = "all" if res_table[y_pred_cl][0] == 0 else "none"
        pos_hm = "last"
        cmap = "jet"
        hm_positive=True
    else:
        invert_hm = "none"
        pos_hm = "none"
        cmap = "bwr"
        hm_positive=False
    
    if heatmaps is None:        
        heatmap, resized_img, max_hm_slice, hm_mean_std, all_heatmaps = gc.multi_models_grad_cam_3d(
            img = np.expand_dims(res_images[0], axis = 0), 
            cnn = cnn,
            model_names = res_model_names[0],
            layers = vis_layers,
            model_mode = model_mode,
            pred_index = 0,
            invert_hm = invert_hm,
            pos_hm = pos_hm,
            # model weigths are only used when model_mode = "weighted"
            model_weights = res_table[0:1].reset_index(drop = True).loc[:, 
                res_table.columns.str.startswith("weight")].to_numpy().squeeze(),
            normalize = normalize_hm)
    else:
        heatmap = heatmaps[np.argwhere(pat == p_id).squeeze()]
        resized_img = res_images[0]

    slices = np.unravel_index(heatmap.argmax(), heatmap.shape)
    print("max slices:", (slices[2], slices[0], slices[1]))
    
    ## Plot Heatmap Average
    phm.plot_heatmap(resized_img, heatmap,
                version = "overlay",
                mode = "avg",
                hm_colormap=cmap,
                hm_positive=hm_positive,
                colorbar=True)

    ## Plot Heatmap Slider
    def slicer(axi_slider, cor_slider, sag_slider):
        phm.plot_heatmap(resized_img, heatmap,
                version = "overlay",
                mode = "def",
                slices = (cor_slider,sag_slider,axi_slider),
                hm_colormap=cmap,
                hm_positive=hm_positive,
                colorbar=True)
        phm.plot_heatmap(resized_img, heatmap,
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
                               y_pred_cl = "y_pred_class_avg",
                               normalize_hm = True,
                               model_mode = "mean",
                               pat_norm_table = None,
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
    # y_pred_cl: the column name of the predicted class
    # normalize_hm: if True then the heatmap is normalized
    # model_mode: the mode for the model (mean, median, max or weighted)
    # heatmaps: if None then the heatmaps are generated, otherwise the heatmaps must be provided (same order as X_in)
    
    ####### redundant bc also in gradcam slicer => make function
    p_ids = [p_id]
    (res_table, res_images, res_model_names) = gc.get_img_and_models(
        p_ids, results = all_results, pats = pat, imgs = X_in, 
        gen_model_name = generate_model_name,
        num_models = num_models)
    
    if model_mode == "weighted":
        y_pred_cl = "y_pred_class_avg_w"
        pred_co = "pred_correct_w"
        y_pred_prob = "y_pred_trafo_avg_w"
        y_pred_u = "y_pred_unc_w"
    else:
        y_pred_cl = "y_pred_class_avg"
        pred_co = "pred_correct"
        y_pred_prob = "y_pred_trafo_avg"
        y_pred_u = "y_pred_unc"

    
    print("patient id: ", res_table.p_id[0])
    print("age: ", pat_dat[pat_dat["p_id"] == res_table.p_id[0]]["age"].values[0])
    print("true mrs: ", res_table.mrs[0])
    print("true class: ", res_table.unfavorable[0])
    print(colored("pred class: "+str(res_table[y_pred_cl][0]), 
                'green' if res_table[pred_co][0] == True else 'red'))
    print("pred prob (class 1): ", res_table[y_pred_prob][0])
    print("pred uncertainty: ", res_table[y_pred_u][0])
      
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

    ###### only this part is different to gradcam
    
    if heatmaps is None:
        (heatmap, resized_img, max_hm_slice, hm_mean_std, all_heatmaps) =  oc.volume_occlusion(
            volume = res_images, 
            res_tab = res_table, 
            occlusion_size = np.array(occ_size), 
            cnn = cnn,
            invert_hm=invert_hm,
            tabular_df=pat_norm_table,
            model_mode = model_mode,
            both_directions=both_directions,
            model_names = res_model_names[0],
            normalize = normalize_hm,
            occlusion_stride = occ_stride)
    else:
        heatmap = heatmaps[np.argwhere(pat == p_id).squeeze()]
        resized_img = res_images[0]
    
    slices = np.unravel_index(heatmap.argmax(), heatmap.shape)
    print("max slices:", (slices[2], slices[0], slices[1]))
    

    #### reduntant bc also in gradcam slicer => make function
    ## Plot Heatmap Average
    phm.plot_heatmap(resized_img, heatmap,
                version = "overlay",
                mode = "avg",
                hm_colormap=cmap,
                hm_positive=hm_positive,
                colorbar=True)

    ## Plot Heatmap Slider
    def slicer(axi_slider, cor_slider, sag_slider):
        phm.plot_heatmap(resized_img, heatmap,
                version = "overlay",
                mode = "def",
                slices = (cor_slider,sag_slider,axi_slider),
                hm_colormap=cmap,
                hm_positive=hm_positive,
                colorbar=True)
        phm.plot_heatmap(resized_img, heatmap,
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