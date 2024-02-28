import os
import h5py
# import zipfile
import pandas as pd
import numpy as np


# Read data as in the original paper:
# https://github.com/LucasKook/dtm-usz-stroke/blob/main/README.md
# not efficient when data is big
def read_and_split_img_data_andrea(path_img, path_tab, path_splits, split, check_print = True):   
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
    if check_print:
        print("image shape in: ", X_in.shape)
        print("image min, max, mean, std: ", X_in.min(), X_in.max(), X_in.mean(), X_in.std())
        
    
    ## read tabular data
    dat = pd.read_csv(path_tab, sep=",")
    if check_print:
        print("tabular shape in: ", dat.shape)
       

    ## read splitting file
    andrea_splits = pd.read_csv(path_splits, 
                                sep='\,', header = None, engine = 'python', 
                                usecols = [1,2,3]).apply(lambda x: x.str.replace(r"\"",""))
    andrea_splits.columns = andrea_splits.iloc[0]
    andrea_splits.drop(index=0, inplace=True)
    andrea_splits = andrea_splits.astype({'idx': 'int32', 'spl': 'int32'})
    splitx = andrea_splits.loc[andrea_splits['spl']==split]        
    if check_print:
        print("split file shape in: ", splitx.shape)
        
    
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
    if check_print:
        print("X tab out shape: ", X_tab.shape)
        print("Y mrs out shape: ", Y_mrs.shape)
        
        
    ## all mrs <= 2 are favorable all higher unfavorable
    Y_new = []
    for element in Y_mrs:
        if element in [0,1,2]:
            Y_new.append(0)
        else:
            Y_new.append(1)
    Y_new = np.array(Y_new)
    
    
    # # Split data into training set and test set "old"
    # X = np.squeeze(X)
    # X = np.float32(X)

    # rng = check_random_state(42)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y_eventtia, train_size=0.8, random_state=rng)
    # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, train_size=0.5, random_state=rng)

    # print(X_train.shape, X_valid.shape, X_test.shape)
    # print(y_train.shape, y_valid.shape, y_test.shape)
    
    
    ## Split data into training set and test set "split" as defined by function
    X = np.squeeze(X)
    X = np.float32(X)

    train_idx = splitx["idx"][splitx['type'] == "train"].to_numpy() -1 
    valid_idx = splitx["idx"][splitx['type'] == "val"].to_numpy() - 1 
    test_idx = splitx["idx"][splitx['type'] == "test"].to_numpy() - 1 

    X_train = X[train_idx]
    # y_train = Y_eventtia[train_idx]
    y_train = Y_new[train_idx]
    X_valid = X[valid_idx]
    # y_valid = Y_eventtia[valid_idx]
    y_valid = Y_new[valid_idx]
    X_test = X[test_idx]
    # y_test = Y_eventtia[test_idx]
    y_test = Y_new[test_idx]

    if check_print:
        print("End shapes X (train, val, test): ", X_train.shape, X_valid.shape, X_test.shape)
        print("End shapes y (train, val, test): ", y_train.shape, y_valid.shape, y_test.shape)
        
        
    ## safe data in table
    results = pd.DataFrame(
        {"p_idx": test_idx+1,
         "p_id": p_id[test_idx],
         "mrs": Y_mrs[test_idx],
         "unfavorable": y_test
        }
    )
    
    return (X_train, X_valid, X_test), (y_train, y_valid, y_test), results

# For 10 Fold data and a given fold: split data into training, validation and test set
def split_data(id_tab, X, fold, X_tab = None):
    # id_tab: table with patient ids and folds
    # X: image data
    # fold: which fold to use (0-9)
    # X_tab: tabular data, if needed
    #
    # Returns a dictionary with the following keys:
    # X, y, (and X_tab) each containing a dictionary with keys train, valid, test

    # make sure id_tab is sorted
    id_tab = id_tab.sort_values("p_id").reset_index(drop=True)
    
    # define indices of train, val, test
    train_idx_tab = id_tab[id_tab["fold" + str(fold)] == "train"]
    valid_idx_tab = id_tab[id_tab["fold" + str(fold)] == "val"]
    test_idx_tab = id_tab[id_tab["fold" + str(fold)] == "test"]
    
    # for X and y it is not the same, because X is defined for all valid patients,
    # but id_tab is only defined for patients with a stroke (no tia) in V3.
    # In V0, V1 and V2 X and id_tab are the same.
    
    # define data
    X_train = X[train_idx_tab.p_idx.to_numpy() - 1]
    y_train = id_tab["unfavorable"].to_numpy()[train_idx_tab.index.to_numpy()]
    X_valid = X[valid_idx_tab.p_idx.to_numpy() - 1]
    y_valid = id_tab["unfavorable"].to_numpy()[valid_idx_tab.index.to_numpy()]
    X_test = X[test_idx_tab.p_idx.to_numpy() - 1]
    y_test = id_tab["unfavorable"].to_numpy()[test_idx_tab.index.to_numpy()]

    data_dict = {"X" : {"train": X_train, "valid": X_valid, "test": X_test}, 
                 "y" : {"train": y_train, "valid": y_valid, "test": y_test}}

    if X_tab is not None:
        X_tab = X_tab.sort_values("p_id").reset_index(drop=True)
        X_tab = X_tab.drop(columns = ["p_id"])

        X_train_tab = X_tab.iloc[train_idx_tab.p_idx.to_numpy() - 1,].to_numpy()
        X_valid_tab = X_tab.iloc[valid_idx_tab.p_idx.to_numpy() - 1,].to_numpy()
        X_test_tab = X_tab.iloc[test_idx_tab.p_idx.to_numpy() - 1,].to_numpy()

        data_dict["X_tab"] = {"train": X_train_tab, "valid": X_valid_tab, "test": X_test_tab}
        
    return data_dict

# Returns data for a given data and model version
# if version == "andrea": returns data for andrea split
def version_setup(DATA_DIR, version, model_version, compatibility_mode = False):
    # DATA_DIR: directory where data is stored
    # version: which data to use (e.g. 10Fold_sigmoid_V1)
    # model_version: which model version to use

    # Returns: 
    #   X_in: 4D numpy array, 3d image data
    #   pat: 1D numpy array, patient ids
    #   id_tab: pandas dataframe, patient ids and folds
    #   all_results_tab: pandas dataframe, results of all models
    #   pat_orig_tab: pandas dataframe, unnormalized tabular data of patients
    #   pat_norm_tab: pandas dataframe, normalized tabular data of patients (only when LSX)
    #   num_models: int, number of models
    #   compatibility_mode: bool, if True, uses old naming convention

    if (version.endswith("V0") or version.endswith("sigmoid") or 
        version.endswith("CIB") or version.endswith("CIBLSX")):
        id_tab = pd.read_csv(DATA_DIR + "10Fold_ids_V0.csv", sep=",")
        num_models = 5
    elif version.endswith("V1"):
        id_tab = pd.read_csv(DATA_DIR + "10Fold_ids_V1.csv", sep=",")
        num_models = 10
    elif version.endswith("V2") or version.endswith("V2f"):
        id_tab = pd.read_csv(DATA_DIR + "10Fold_ids_V2.csv", sep=",")
        num_models = 5
    elif version.endswith("V3"):
        id_tab = pd.read_csv(DATA_DIR + "10Fold_ids_V3.csv", sep=",")
        num_models = 5
    pat = id_tab["p_id"].to_numpy()
    X_in = np.load(DATA_DIR + "prepocessed_dicom_3d.npy")

    if not compatibility_mode:
        # load results
        path_results = DATA_DIR + "all_tab_results_" + version + "_M" + str(model_version) + ".csv" # 10 Fold
    elif compatibility_mode:
        path_results = DATA_DIR + "all_tab_results_10Fold_" + version + "_M" + str(model_version) + ".csv"

    if os.path.exists(path_results):    
        all_results_tab = pd.read_csv(path_results, sep=",")
        all_results_tab = all_results_tab.sort_values("p_idx").reset_index(drop=True)
    else: 
        all_results_tab = None
    
    pat_orig_tab = pd.read_csv(DATA_DIR + "/baseline_data_zurich_prepared0.csv", sep=";", decimal=",")
    pat_orig_tab = pat_orig_tab.sort_values("p_id").reset_index(drop=True)
    pat_orig_tab = pat_orig_tab[pat_orig_tab["p_id"].isin(pat)]

    if "LSX" in version:
        pat_norm_tab = pd.read_csv(DATA_DIR + "/baseline_data_zurich_prepared.csv", sep=",")
        pat_norm_tab = pat_norm_tab.sort_values("p_id").reset_index(drop=True)
        pat_norm_tab = pat_norm_tab[pat_norm_tab["p_id"].isin(pat)]
        pat_norm_tab = pat_norm_tab[["p_id", "age", "sexm", "nihss_baseline", "mrs_before",
                                     "stroke_beforey", "tia_beforey", "ich_beforey", 
                                     "rf_hypertoniay", "rf_diabetesy", "rf_hypercholesterolemiay", 
                                     "rf_smokery", "rf_atrial_fibrillationy", "rf_chdy"]]
    else:
        pat_norm_tab = None
        
    return X_in, pat, id_tab, all_results_tab, pat_orig_tab, pat_norm_tab, num_models

# Returns directories for a given data and model version
def dir_setup(INPUTDIR, OUTPUTDIR, 
              version, model_version, 
              weight_mode = "avg", 
              hm_type = "gc", 
              pred_hm = True, 
              hm_norm = False,
              compatibility_mode = False):
    
    # INPUTDIR: root directory for inputs (data, weights)
    # OUTPUTDIR: root directory for outputs (working directory)
    # version: which data to use (e.g. 10Fold_sigmoid_V1)
    # model_version: which model version to use
    # hm_type: which heatmap type to use (gc (gradcam), oc (occlusion))
    # pred_hm: predicted-class (if True) or both classes
    # hm_norm: heatmap normalization (= True) and False for un-normalized heatmaps
    # compatibility_mode: if True: old_name convention, if False: new name convention
    
    DATA_DIR = INPUTDIR + "data/"
    WEIGHT_DIR = INPUTDIR + "weights/10Fold_" + version + "/"
    DATA_OUTPUT_DIR = OUTPUTDIR  + "pictures/10Fold_" + version + "/"
    PIC_OUTPUT_DIR = OUTPUTDIR  + "pictures/10Fold_" + version + "/"

    if not compatibility_mode:
        save_name = version + "_M" + str(model_version)    
    elif compatibility_mode:
        save_name = "10Fold_" + version + "_M" + str(model_version)

    if weight_mode is not None:
        save_name = save_name + "_" + weight_mode

    if hm_type is not None:
        save_name = save_name + "_" + hm_type   

    if pred_hm:
        save_name = save_name + "_" + "predcl"
    elif not pred_hm:
        save_name = save_name + "_" + "bothcl" 

    if not hm_norm:
        save_name = save_name + "_" + "unnormalized"    
    elif not compatibility_mode and hm_norm:
        save_name = save_name + "_" + "normalized"
       
    return DATA_DIR, WEIGHT_DIR, DATA_OUTPUT_DIR, PIC_OUTPUT_DIR, save_name


# def normalize(volume):
#     """Normalize the volume"""
#     min = np.min(volume)
#     max = np.max(volume) 
#     volume = (volume - min) / (max - min)
#     volume = volume.astype("float32")
#     return volume

# X_in = np.array([normalize(img) for img in X_in])
# print(X_in.shape, X_in.min(), X_in.max(), X_in.mean(), X_in.std())


#newly created by Maurice
def split_data_tabular(id_tab, X, fold):    
    
    with h5py.File('/tf/notebooks/hezo/stroke_perfusion/data/dicom_2d_192x192x3_clean_interpolated_18_02_2021_preprocessed2.h5', "r") as h5:
        pat = h5["pat"][:]

    # already normalized
    dat = pd.read_csv("/tf/notebooks/hezo/stroke_perfusion/data/baseline_data_zurich_prepared.csv", sep = ",")    

    ## extract X and Y and split into train, val, test
    n = []
    for p in pat:
        if p in dat.p_id.values:
            n.append(p)
    n = len(n)
    X_tab = np.zeros((n, 13))

    i = 0
    for j, p in enumerate(pat):
        if p in dat.p_id.values:
            k = np.where(dat.p_id.values == p)[0]
            X_tab[i,:] = dat.loc[k,["age", "sexm", "nihss_baseline", "mrs_before",
                                   "stroke_beforey", "tia_beforey", "ich_beforey", 
                                   "rf_hypertoniay", "rf_diabetesy", "rf_hypercholesterolemiay", 
                                   "rf_smokery", "rf_atrial_fibrillationy", "rf_chdy"]]

            i += 1    
        
    # id_tab: table with patient ids and folds
    # X: image data
    # fold: which fold to use (0-9)
    
    # define indices of train, val, test
    train_idx_tab = id_tab[id_tab["fold" + str(fold)] == "train"]
    valid_idx_tab = id_tab[id_tab["fold" + str(fold)] == "val"]
    test_idx_tab = id_tab[id_tab["fold" + str(fold)] == "test"]
    
    # for X and y it is not the same, because X is defined for all valid patients,
    # but id_tab is only defined for patients with a stroke (no tia) in V3.
    # In V0, V1 and V2 X and id_tab are the same.
    
    # define data
    X_train = X[train_idx_tab.p_idx.to_numpy() - 1]
    y_train = id_tab["unfavorable"].to_numpy()[train_idx_tab.index.to_numpy()]
    
    X_valid = X[valid_idx_tab.p_idx.to_numpy() - 1]
    y_valid = id_tab["unfavorable"].to_numpy()[valid_idx_tab.index.to_numpy()]
    
    X_test = X[test_idx_tab.p_idx.to_numpy() - 1]
    y_test = id_tab["unfavorable"].to_numpy()[test_idx_tab.index.to_numpy()]
    
    X_train_tab = X_tab[train_idx_tab.p_idx.to_numpy() - 1]
    X_valid_tab = X_tab[valid_idx_tab.p_idx.to_numpy() - 1]
    X_test_tab = X_tab[test_idx_tab.p_idx.to_numpy() - 1] 
           
    return (X_train, X_valid, X_test),(X_train_tab, X_valid_tab, X_test_tab), (y_train, y_valid, y_test)

#### graveyard


def split_data_tabular_test():    
    with h5py.File('/tf/notebooks/hezo/stroke_perfusion/data/dicom_2d_192x192x3_clean_interpolated_18_02_2021_preprocessed2.h5', "r") as h5:
        pat = h5["pat"][:]

    # already normalized
    dat = pd.read_csv("/tf/notebooks/hezo/stroke_perfusion/data/baseline_data_zurich_prepared.csv", sep=",")    

    ## extract X and Y and split into train, val, test
    n = []
    for p in pat:
        if p in dat.p_id.values:
            n.append(p)
    n = len(n)
    X_tab = np.zeros((n, 14))  # Increased the number of columns to accommodate patient ID

    i = 0
    for j, p in enumerate(pat):
        if p in dat.p_id.values:
            k = np.where(dat.p_id.values == p)[0]
            X_tab[i, :-1] = dat.loc[k, ["age", "sexm", "nihss_baseline", "mrs_before",
                                        "stroke_beforey", "tia_beforey", "ich_beforey", 
                                        "rf_hypertoniay", "rf_diabetesy", "rf_hypercholesterolemiay", 
                                        "rf_smokery", "rf_atrial_fibrillationy", "rf_chdy"]].values
            X_tab[i, -1] = p  # Add patient ID to the last column
            i += 1
    
    # Convert NumPy array to DataFrame
    columns = ["age", "sexm", "nihss_baseline", "mrs_before",
               "stroke_beforey", "tia_beforey", "ich_beforey", 
               "rf_hypertoniay", "rf_diabetesy", "rf_hypercholesterolemiay", 
               "rf_smokery", "rf_atrial_fibrillationy", "rf_chdy", "patient_id"]
    df_result = pd.DataFrame(X_tab, columns=columns)
    
    return df_result


####

