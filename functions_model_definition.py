import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model

from k_ontram_functions.ontram import ontram
from k_ontram_functions.ontram_loss import ontram_loss
from k_ontram_functions.ontram_metrics import ontram_acc
import numpy as np

# Define the 3d cnn model for binary stroke classification      
# Consists of 4 convolutional blocks, 1 fully connected block and 1 output layer
def stroke_binary_3d(input_dim = (128, 128, 28,1), 
                     output_dim = 1,
                     layer_connection = "globalAveragePooling",
                     last_activation = "sigmoid"):
    # input_dim: tuple of integers, shape of input data
    # output_dim: integer, if 1 sigmoid or linear activation must be used, if 2 softmax activation must be used
    # layer_connection: string, either "flatten" or "globalAveragePooling"
    # last_activation: string, either "sigmoid", "linear" or "softmax"
    
    valid_layer_connection = ["flatten", "globalAveragePooling"]
    if layer_connection not in valid_layer_connection:
        raise ValueError("stroke_binary_3d: layer_connection must be one of %r." % valid_layer_connection)
    valid_activation = ["sigmoid", "linear", "softmax"]
    if last_activation not in valid_activation:
        raise ValueError("stroke_binary_3d: last_activation must be one of %r." % valid_activation)
           
    #input
    inputs = keras.Input(input_dim)
    
    # conv block 0
    x = layers.Convolution3D(32, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(inputs)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    # conv block 1
    x = layers.Convolution3D(32, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    # conv block 2
    x = layers.Convolution3D(64, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    # conv block 3
    x = layers.Convolution3D(64, kernel_size=(3, 3, 3), padding = 'same',activation = 'relu')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
       
    # cnn to flat connection
    if layer_connection == list(valid_layer_connection)[0]:
        x = layers.Flatten()(x)
    elif layer_connection == list(valid_layer_connection)[1]:
        x = layers.GlobalAveragePooling3D()(x) 
    
    # flat block
    x = layers.Dense(128, activation = 'relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation = 'relu')(x)
    x = layers.Dropout(0.3)(x)
    
    if last_activation == list(valid_activation)[0]:
        out = layers.Dense(units=output_dim, activation = last_activation)(x) # sigmoid
    elif last_activation == list(valid_activation)[1]:
        out = layers.Dense(units=output_dim, activation = last_activation, use_bias = False)(x) # linear
    elif last_activation == list(valid_activation)[2]:
        out = layers.Dense(units=output_dim, activation = last_activation)(x) # softmax (output_dim must be at least 2)
    
    # Define the model.
    model_3d = Model(inputs=inputs, outputs=out, name = "cnn_3d_")
    
    return model_3d

# Model for linear shift terms
def mod_linear_shift(x, weights = None):
    mod = keras.Sequential(name = "mod_linear_shift")
    mod.add(tf.keras.layers.Dense(1, activation="linear", use_bias=False, input_shape=(x,)))
    
    if weights is not None:
        mod.layers[0].set_weights([weights])
    
    return mod


# Model for complex intercept
def img_model_linear_final(input_shape, output_shape, activation = "linear"):
    
    in_ = keras.Input(shape = input_shape)

    # conv block 0
    x = keras.layers.Convolution3D(32, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu', name = "CIB_Conv3D0")(in_)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # conv block 1
    x = keras.layers.Convolution3D(32, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu', name = "CIB_Conv3D1")(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # conv block 2
    x = keras.layers.Convolution3D(64, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu', name = "CIB_Conv3D2")(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # conv block 3
    x = keras.layers.Convolution3D(64, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu', name = "CIB_Conv3D3")(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # cnn to flat connection
    x = keras.layers.GlobalAveragePooling3D()(x) 
    
    # flat block
    x = keras.layers.Dense(128, activation = 'relu', name = "CIB_Dense1")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation = 'relu', name = "CIB_Dense2")(x)
    x = keras.layers.Dropout(0.3)(x)
    out_ = keras.layers.Dense(output_shape, activation = activation, use_bias = False, 
                              name = "CIB_dense_complex_intercept")(x) 
    nn_im = keras.Model(inputs = in_, outputs = out_, name = "CIB_mod_complex_intercept")
    return nn_im

# Define the 3d cnn model parameters for binary stroke classification based on the current model version
def model_setup(version, input_dim = (128, 128, 28, 1), ):
    # version: string, model version, e.g. 10Fold_sigmoid_V0
    # input_dim: tuple of integers, shape of input data

    if "sigmoid" in version or "andrea_split" in version:
        last_activation = "sigmoid"
        output_dim = 1
        LOSS = "binary_crossentropy"
    elif "softmax" in version:
        last_activation = "softmax"
        output_dim = 2
        LOSS = tf.keras.losses.categorical_crossentropy
    elif "CIB" in version:
        last_activation = "linear"
        output_dim = 1
        LOSS = None
        
    if version.endswith("f"):
        layer_connection = "flatten"
    else:
        layer_connection = "globalAveragePooling"
        
    return input_dim, output_dim, LOSS, layer_connection, last_activation

def model_init(version, 
               output_dim,
               LOSS,
               layer_connection = None,
               last_activation = None,
               C = None,
               learning_rate = 5*1e-5,
               batch_size = 6,
               input_dim = (128, 128, 28, 1),
               input_dim_tab = None,
               weights_tab_init = None,
               cnn_weights_init_path = None):
    
    # version: string, model version, e.g. 10Fold_sigmoid_V0
    # output_dim: integer, if sigmoid, linear activation or ontram 1 must be used, if softmax then 2
    # LOSS: string or function, loss function
    # layer_connection: string, either "flatten" or "globalAveragePooling" or None
    # last_activation: string, either "sigmoid", "linear" or "softmax" or None
    # C: integer, number of classes for Ontram
    # learning_rate: float, learning rate for optimizer
    # input_dim: tuple of integers, shape of input data
    # input_dim_tab: tuple of integers, shape of input data for tabular data

    if ("sigmoid" or "softmax" or "andrea_split") in version:
        model_3d = stroke_binary_3d(input_dim = input_dim,
                               output_dim = output_dim,
                               layer_connection = layer_connection,
                               last_activation = last_activation)
        model_3d.compile(
            loss=LOSS,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["acc", tf.keras.metrics.AUC()])
    elif "CIBLSX" in version:
        mbl = img_model_linear_final(input_dim, output_dim)
        mls = mod_linear_shift(input_dim_tab, weights=weights_tab_init)     
        model_3d = ontram(mbl, mls)   
        
        if cnn_weights_init_path is not None:
            model_3d_cib = ontram(mbl) 
            model_3d_cib.load_weights(cnn_weights_init_path) 

            ciblsx_namelist = []
            for layer in model_3d.layers:
                ciblsx_namelist.append(layer.name)

            cib_namelist = []
            for layer in model_3d_cib.layers:
                cib_namelist.append(layer.name)

            cib_namelist_filter = [name for name in cib_namelist if "CIB" in name]

            ciblsx_indexlist = []
            for name in cib_namelist_filter:
                ciblsx_indexlist.append(ciblsx_namelist.index(name))

            cib_indexlist = []
            for name in cib_namelist_filter:
                cib_indexlist.append(cib_namelist.index(name))

            for old, new in zip(cib_indexlist,ciblsx_indexlist):
                model_3d.layers[new].set_weights(model_3d_cib.layers[old].get_weights())

        model_3d.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                                        loss=ontram_loss(C, batch_size),
                                        metrics=[ontram_acc(C, batch_size)])   
    
    elif "CIB" in version:
        mbl = img_model_linear_final(input_dim, output_dim)
        model_3d = ontram(mbl)             

        model_3d.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                                        loss=ontram_loss(C, batch_size),
                                        metrics=[ontram_acc(C, batch_size)])

    return model_3d

# Define the generate_model_name function based on model version, layer connection and last activation
# Returns a function that generates a model name based on the split and model number
def set_generate_model_name(model_version, layer_connection, last_activation, path, compatability_mode = False):
    # compatibility_mode: bool, if True, uses old naming convention

    def generate_model_name(which_split, model_nr):
        if not compatability_mode:
            if "CIBLSX" in path:
                return (path + "3D_CNN_avg_layer_binary_outcome_CIBLSX_split" + str(which_split) +
                    "_ens" + str(model_nr) + "_M" + str(model_version) + ".h5")
            elif "CIB" in path:
                return (path + "3D_CNN_avg_layer_binary_outcome_CIB_split" + str(which_split) + 
                    "_ens" + str(model_nr) + "_M" + str(model_version) + ".h5")

        # old naming convention
        elif compatability_mode:
            if last_activation == "linear" and "CIBLSX" in path:
                return (path + "3d_cnn_binary_model_split" + "CIB_LSX" + str(which_split) + 
                    "_normalized_avg_layer_paper_model_" + last_activation + 
                    "_activation_"  + str(model_version) + "_" + str(model_nr) + ".h5")
            elif last_activation == "linear" and "CIB" in path:
                return (path + "3d_cnn_binary_model_split" + "CIB" + str(which_split) + 
                    "_normalized_avg_layer_paper_model_" + last_activation + 
                    "_activation_"  + str(model_version) + "_" + str(model_nr) + ".h5")
            elif layer_connection == "globalAveragePooling":
                return (path + "3d_cnn_binary_model_split" + str(which_split) + 
                        "_unnormalized_avg_layer_paper_model_" + last_activation + 
                        "_activation_"  + str(model_version) + str(model_nr) + ".h5")
            elif layer_connection == "flatten":
                return (path + "3d_cnn_binary_model_split" + str(which_split) + 
                        "_unnormalized_flat_layer_paper_model_" + last_activation + 
                        "_activation_" + str(model_version) + str(model_nr) + ".h5")

    return generate_model_name


def get_last_conv_layer(model):
    vis_layers = [i.name for i in model.layers]
    vis_layers = [vis_layer for vis_layer in vis_layers if vis_layer.startswith("conv")]
    return vis_layers[-1]



def model_init_test(version, 
               output_dim,
               LOSS,
               layer_connection = None,
               last_activation = None,
               C = None,
               learning_rate = 5*1e-5,
               batch_size = 6,
               input_dim = (128, 128, 28, 1),
               input_dim_tab = None,
               weights_tab_init = None,
               cnn_weights_init = None):
    
    # version: string, model version, e.g. 10Fold_sigmoid_V0
    # output_dim: integer, if sigmoid, linear activation or ontram 1 must be used, if softmax then 2
    # LOSS: string or function, loss function
    # layer_connection: string, either "flatten" or "globalAveragePooling" or None
    # last_activation: string, either "sigmoid", "linear" or "softmax" or None
    # C: integer, number of classes for Ontram
    # learning_rate: float, learning rate for optimizer
    # input_dim: tuple of integers, shape of input data
    # input_dim_tab: tuple of integers, shape of input data for tabular data

    if ("sigmoid" or "softmax" or "andrea_split") in version:
        model_3d = stroke_binary_3d(input_dim = input_dim,
                               output_dim = output_dim,
                               layer_connection = layer_connection,
                               last_activation = last_activation)
        model_3d.compile(
            loss=LOSS,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["acc", tf.keras.metrics.AUC()])
    elif "CIBLSX" in version:
        mbl = img_model_linear_final(input_dim, output_dim)
        mls = mod_linear_shift(input_dim_tab, weights=None)
        model_3d = ontram(mbl, mls)   
        
        if cnn_weights_init is not None:
            model_3d.load_weights(cnn_weights_init)
        
        if weights_tab_init is not None:
            mls = mod_linear_shift(input_dim_tab, weights=weights_tab_init) 

        return model_3d

        model_3d.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                                        loss=ontram_loss(C, batch_size),
                                        metrics=[ontram_acc(C, batch_size)])
        
    elif "CIB" in version:
        mbl = img_model_linear_final(input_dim, output_dim)
        model_3d = ontram(mbl)             

        model_3d.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                                        loss=ontram_loss(C, batch_size),
                                        metrics=[ontram_acc(C, batch_size)])

    return model_3d

