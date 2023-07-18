import warnings
import skimage

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import functions_gradcam as gc

def plot_gradcam(resized_img, heatmap, 
                 version = "overlay",
                 mode = "avg",
                 slices = None,
                 orig_max = False,
                 heatmap_threshold = None,
                 negative = False,
                 pic_size = (128,128),
                 show = True,
                 add_plot = None,
                 hm_positive = True,
                 hm_colormap = "jet",
                 colorbar = False,
                 slice_line = False,
                 return_fig = False):
    # mode: avg => averages (mean) of heatmaps and image. avg_heatmap_correction can be applied for heatmap average
    #       max => extracts slice with highest activation for heatmap and image
    #       def => extracts defined slices. if not defined by "slices", than extract middle slice of each view
    # slices: should be none or a tuple of shape 3. defines which slice to take when mode == "def" (cor, sag, ax)
    # orig_max: if True and mode is "max" then the slice with the brightest point is selected
    # heatmap_threshold: if not None than should be between 0 and 1. At which proportion of the heatmap values, 
    #                     the heatmap should be set to 0. This can reduce noise of GradCam. "gradcam threshold"
    # add_plot: if not NULL, a tuple of (current_row, total_rows) must be given (current_row starts counting with 0)
    # hm_positive: if True only positive Values will be shown (should be normal if gradcam++) 
    # slice_line: if True a line will be drawn in the image to show which slice is selected
    
    valid_versions = ["overlay", "original", "activation"]
    valid_modes = ["avg", "max", "def"]
    if version not in valid_versions:
        raise ValueError("plot_gradcam: version must be one of %r." % valid_versions)
    if mode not in valid_modes:
        raise ValueError("plot_gradcam: mode must be one of %r." % valid_modes)
    if heatmap_threshold is not None:
        if not (heatmap_threshold < 1 and heatmap_threshold > 0):
            raise ValueError("plot_gradcam: if heatmap_threshold is not None than must be between 0 and 1")
    
    if slices is None and mode == "def":
        warnings.warn("plot_gradcam: slices are not defined but mode is set to def. Plot slices (64,64,14)")
        slices = (64,64,14)
    elif slices is not None and mode in ["avg", "max"]:
        warnings.warn("plot_gradcam: slices are defined but mode is not set to def. Ignore value of slice!")

    if mode == "max" and orig_max == False:
        slices = np.unravel_index(heatmap.argmax(), heatmap.shape)
    elif mode == "max" and orig_max == True:
        slices = np.unravel_index(resized_img.argmax(), resized_img.shape)
        
    if negative:
        resized_img = np.negative(resized_img)

    if add_plot is None:
        fig = plt.figure(figsize = (15,15))
    
    # Define all plots (image and heatmap even if not used)
    # Also define captions
    if mode == "avg":
        img_ax = np.mean(resized_img, axis = 2)
        map_ax = np.mean(heatmap, axis = 2)
        img_cor = skimage.transform.resize(np.rot90(np.mean(resized_img, axis = 0)), output_shape=pic_size)
        map_cor = skimage.transform.resize(np.rot90(np.mean(heatmap, axis = 0)), output_shape=pic_size)
        img_sag = skimage.transform.resize(np.fliplr(np.rot90(np.mean(resized_img, axis = 1))), output_shape=pic_size)
        map_sag = skimage.transform.resize(np.fliplr(np.rot90(np.mean(heatmap, axis = 1))), output_shape=pic_size)
        captions = ["Axial Avg", "Coronal Avg", "Sagital Avg"]            
    else:
        img_ax = resized_img[:,:,slices[2]]
        map_ax = heatmap[:,:,slices[2]]
        img_cor = skimage.transform.resize(np.rot90(resized_img[slices[0],:,: ]), output_shape=pic_size)
        map_cor = skimage.transform.resize(np.rot90(heatmap[slices[0],:,:]), output_shape=pic_size)
        img_sag = skimage.transform.resize(np.fliplr(np.rot90(resized_img[:,slices[1],: ])), output_shape=pic_size)
        map_sag = skimage.transform.resize(np.fliplr(np.rot90(heatmap[:,slices[1],:])), output_shape=pic_size)
        captions = ["Ax. Slice: " + str(slices[2]), "Cor. Slice: " + str(slices[0]), "Sag. Slice: " + str(slices[1])]
        
    # apply heatmap_threshold if it is defined (GradCam threshold)
    if heatmap_threshold is not None:
        ax_th = np.max(map_ax)*heatmap_threshold
        map_ax[map_ax < ax_th] = 0
        cor_th = np.max(map_cor)*heatmap_threshold
        map_cor[map_cor < cor_th] = 0
        sag_th = np.max(map_sag)*heatmap_threshold
        map_sag[map_sag < sag_th] = 0

    images = [img_ax, img_cor, img_sag]
    heatmaps = [map_ax, map_cor, map_sag]
    
    images_min = np.array(images).min()
    images_max = np.array(images).max()
    
    if hm_positive:
        heatmaps_min = np.array(heatmaps).min()
        heatmaps_max = np.array(heatmaps).max()
    else:
        abs_max = np.abs(np.array(heatmaps)).max()
        heatmaps_min = -abs_max
        heatmaps_max = abs_max
        
    
    # For all three views plot desired version and add caption
    if slice_line and mode != "avg":
        h_line = (0,2,2)
        v_line = (1,1,0)
        
    for i in range(3):
        if add_plot is None:
            ax = plt.subplot(1,3, i+1)
        elif len(add_plot) == 2:
            ax = plt.subplot(add_plot[1], 3, (add_plot[0]*3)+(i+1))
            
        if version in ["overlay", "original"]:
            ax.imshow(images[i], cmap='gray', vmin = images_min, vmax = images_max)
        if version in ["overlay", "activation"]:
            ax.imshow(heatmaps[i], alpha=0.4, cmap=hm_colormap, vmin = heatmaps_min, vmax = heatmaps_max)
        if slice_line and mode != "avg":
            ax.axhline(y=slices[h_line[i]] if i == 0 else
                       heatmap.shape[0]-np.round(slices[h_line[i]]*heatmap.shape[0]/heatmap.shape[2]), 
                       color='r', linestyle='-')
            ax.axvline(x=slices[v_line[i]] if i != 2 else 
                       heatmap.shape[1]-slices[v_line[i]], 
                       color='r', linestyle='-')
        ax.set_title(captions[i])
        plt.axis('off')
    
    if colorbar and version != "original":
        axins = inset_axes(
            ax,
            width="5%",  # width: 5% of parent_bbox width
            height="100%",  # height: 50%
            loc="lower left",
            bbox_to_anchor=(1.01, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        plt.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=heatmaps_min, vmax=heatmaps_max, clip=False), 
                cmap=hm_colormap), 
            cax=axins,
            label='',
            ticks=np.trunc(np.linspace(heatmaps_min, heatmaps_max, 5)*100)/100)
        
    if show:
        plt.show()
    if return_fig:
        return fig, ax
        
        
# Plot function: Last conv layer, average over all conv layer and original
def plot_gradcams_last_avg_org(res_table, vis_layers, res_images, res_model_names, model_3d,
                               layer_mode, heatmap_mode, save_path, save_name, save = True, hm_of_pred = True):
    
    if "sigmoid" in str(model_3d.layers[-1].activation):
        pred_idx = 0
    elif "softmax" in str(model_3d.layers[-1].activation):
        pred_idx = 1
    
    if len(res_table["p_id"]) != res_table["p_id"].nunique():
        add_testset = True
    else:
        add_testset = False
    
    for j in range(len(res_table)):   
        plot_per_iter = 1
        plot_at_end = 1
        layer_iter = 1 #len(vis_layers)
        num_rows = layer_iter*plot_per_iter + plot_per_iter + plot_at_end
        width = 15

        start_text = 0.08
        end_text = 0.89
        text_pos = np.flip(np.linspace(
            start_text+(plot_at_end/num_rows)+0.2/(num_rows-plot_at_end), 
            end_text-0.2/(num_rows-plot_at_end), 
            layer_iter+1))

        fig = plt.figure(figsize = (width,num_rows*width/3))

        plt.gcf().text(0.14, end_text+3/num_rows/18, "p_id:        " + str(round(res_table["p_id"][j])), fontsize=16)
        plt.gcf().text(0.14, end_text+2/num_rows/18, "true_mrs:    " + str(round(res_table["mrs"][j])), fontsize=16)
        plt.gcf().text(0.14, end_text+1/num_rows/18, "true class:  " + str(res_table["unfavorable"][j]), fontsize=16)
        plt.gcf().text(0.4, end_text+3/num_rows/18, "pred class:          " + str(res_table["y_pred_class"][j]), fontsize=16,
                      fontweight = "bold", color = ("red" if res_table["pred_correct"][j] == False else "black"))
        plt.gcf().text(0.4, end_text+2/num_rows/18, "pred prob (class 1): " + str(round(res_table["y_pred_trafo_avg"][j], 3)), fontsize=16)
        plt.gcf().text(0.4, end_text+1/num_rows/18, "pred uncertainty:    " + str(round(res_table["y_pred_unc"][j], 3)), fontsize=16)
        if "heatmap_std_avg_layer" in res_table:
            plt.gcf().text(0.66, end_text+3/num_rows/18, 
                           "heatmap unc. avg layer: " + str(round(res_table["heatmap_unc_avg_layer"][j], 3)), fontsize=16)
            plt.gcf().text(0.66, end_text+2/num_rows/18, 
                           "heatmap unc. last layer: " + str(round(res_table["heatmap_unc_last_layer"][j], 3)), fontsize=16)
        
        # check predicted class
        if res_table["y_pred_class"][j] == 0 and hm_of_pred == True:
            invert_last_layer = "last"
        else:
            invert_last_layer = "none"

        # last layer
        plt.gcf().text(0.1, text_pos[0], "Layer: " + vis_layers[-1], 
                       horizontalalignment='center', verticalalignment='center', fontsize=14, rotation = 90)

        heatmap, resized_img, max_hm_slice, hm_mean_std = gc.multi_models_grad_cam_3d(
                img = res_images[j:j+1], 
                cnn = model_3d,
                model_names = res_model_names[j],
                layers = vis_layers[-1],
                model_mode = layer_mode,
                layer_mode = layer_mode,
                pred_index = pred_idx,
                invert_hm = invert_last_layer)

        plot_gradcam(resized_img, heatmap,
                version = "overlay",
                mode = heatmap_mode,
                add_plot = (0,num_rows),
                show = False)


        # average over all layers
        heatmap, resized_img, max_hm_slice, hm_mean_std = gc.multi_models_grad_cam_3d(
                img = res_images[j:j+1], 
                cnn = model_3d,
                model_names = res_model_names[j],
                layers = vis_layers,
                model_mode = layer_mode,
                layer_mode = layer_mode,
                pred_index = pred_idx,
                invert_hm = invert_last_layer)

    #     print(layer_mode, "over all Layers")
        plt.gcf().text(0.1, text_pos[-1], layer_mode + " over all Layers", 
                       horizontalalignment='center', verticalalignment='center', fontsize=14, rotation = 90)
        plot_gradcam(resized_img, heatmap,
                    version = "overlay",
                    mode = heatmap_mode,
                    add_plot = (1,num_rows),
                    show = False)

    #     print("Original")
        plt.gcf().text(0.1, start_text+(plot_at_end/num_rows)/2, "Original", 
                       horizontalalignment='center', verticalalignment='center', fontsize=14, rotation = 90)
        plot_gradcam(resized_img, heatmap,
                    version = "original",
                    orig_max = True,
                    mode = heatmap_mode,
                    add_plot = (num_rows-1,num_rows),
                    show = False)

        plt.subplots_adjust(wspace=0.05, hspace=0.15)
        if save:
            plt.savefig(save_path + 'pat' + str(round(res_table["p_id"][j])) + '_' + 
                        ("testset_" + res_table["p_id"][j] + "_" if add_testset else "") +
                        save_name + '_last_and_all_layers_' + heatmap_mode + '.png')
            
# Plot function:  Last conv layer, average over all conv layer and original
def plot_gradcams_avg_max_org(pat_data, res_table, vis_layers, res_images, res_model_names, model_3d,
                               layer_mode, cmap, hm_positive,
                               save_path, save_name, save = True, hm_of_pred = True):
    
    if "sigmoid" in str(model_3d.layers[-1].activation):
        pred_idx = 0
    elif "softmax" in str(model_3d.layers[-1].activation):
        pred_idx = 1
    
    if len(res_table["p_id"]) != res_table["p_id"].nunique():
        add_testset = True
    else:
        add_testset = False
    
    for j in range(len(res_table)):   
        plot_per_iter = 1
        plot_at_end = 1
        layer_iter = 1 #len(vis_layers)
        num_rows = layer_iter*plot_per_iter + plot_per_iter + plot_at_end
        width = 15

        start_text = 0.08
        end_text = 0.89
        text_pos = np.flip(np.linspace(
            start_text+(plot_at_end/num_rows)+0.2/(num_rows-plot_at_end), 
            end_text-0.2/(num_rows-plot_at_end), 
            layer_iter+1))

        fig = plt.figure(figsize = (width,num_rows*width/3))

        plt.gcf().text(0.14, end_text+3/num_rows/18, "p_id: " + str(round(res_table["p_id"][j])), fontsize=16)
        plt.gcf().text(0.14, end_text+2/num_rows/18, "true_mrs: " + str(round(res_table["mrs"][j])), fontsize=16)
        plt.gcf().text(0.14, end_text+1/num_rows/18, "age: " + str(
            pat_data.age[pat_data["p_id"] == res_table["p_id"][j]].values[0]), fontsize=16)
        plt.gcf().text(0.4, end_text+3/num_rows/18, "true class: " + str(res_table["unfavorable"][j]), fontsize=16)
        plt.gcf().text(0.4, end_text+2/num_rows/18, "pred class: " + str(res_table["y_pred_class"][j]), fontsize=16,
                      fontweight = "bold", color = ("red" if res_table["pred_correct"][j] == False else "black"))
        plt.gcf().text(0.4, end_text+1/num_rows/18, "pred prob (class 1): " + str(round(res_table["y_pred_trafo_avg"][j], 3)), fontsize=16)
        plt.gcf().text(0.66, end_text+3/num_rows/18, "pred uncertainty: " + str(round(res_table["y_pred_unc"][j], 3)), fontsize=16)
        if "heatmap_unc_last_layer" in res_table:
            plt.gcf().text(0.66, end_text+2/num_rows/18, 
                           "heatmap unc. last layer: " + str(round(res_table["heatmap_unc_last_layer"][j], 3)), fontsize=16)
        
        # check predicted class
        if res_table["y_pred_class"][j] == 0 and hm_of_pred == True:
            invert_last_layer = "last"
        else:
            invert_last_layer = "none"

        # last layer
        plt.gcf().text(0.1, text_pos[0], "Average Heatmap", 
                       horizontalalignment='center', verticalalignment='center', fontsize=14, rotation = 90)

        heatmap, resized_img, max_hm_slice, hm_mean_std = gc.multi_models_grad_cam_3d(
                img = res_images[j:j+1], 
                cnn = model_3d,
                model_names = res_model_names[j],
                layers = vis_layers,
                model_mode = layer_mode,
                layer_mode = layer_mode,
                pred_index = pred_idx,
                invert_hm = invert_last_layer)

        plot_gradcam(resized_img, heatmap,
                version = "overlay",
                mode = "avg",
                hm_colormap=cmap,
                hm_positive=hm_positive,
                colorbar=True,
                add_plot = (0,num_rows),
                show=False)

        plt.gcf().text(0.1, text_pos[-1], "Max Heatmap Slice", 
                       horizontalalignment='center', verticalalignment='center', fontsize=14, rotation = 90)
        plot_gradcam(resized_img, heatmap,
                    version = "overlay",
                    mode = "max",
                    hm_colormap=cmap,
                    hm_positive=hm_positive,
                    colorbar=True,
                    add_plot = (1,num_rows),
                    show = False)

        plt.gcf().text(0.1, start_text+(plot_at_end/num_rows)/2, "Original", 
                       horizontalalignment='center', verticalalignment='center', fontsize=14, rotation = 90)
        plot_gradcam(resized_img, heatmap,
                    version = "original",
                    mode = "max",
                    orig_max = True,
                    hm_colormap=cmap,
                    hm_positive=hm_positive,
                    add_plot = (num_rows-1,num_rows),
                    show = False)

        plt.subplots_adjust(wspace=0.05, hspace=0.15)
        if save:
            plt.savefig(save_path + 'pat' + str(round(res_table["p_id"][j])) + '_' + 
                        ("testset_" + res_table["p_id"][j] + "_" if add_testset else "") +
                        save_name + '_last_layer_avg_max_orig.png')