import os
DIR = "/tf/notebooks/schnemau/xAI_stroke_3d/"
os.chdir(DIR)
import Utils_maurice as utils
import numpy as np
from sklearn import metrics
from scipy.optimize import minimize

#------------------------------------------------------------------------------------------------------------------------------------------------------

'''def get_ensemble(intercepts, shift, X_tab, weights=None, n_ens=5):
    if weights is None:
        weights = np.ones(n_ens) / n_ens
    weighted_intercepts = np.average(np.array(intercepts), axis=0, weights=weights)
    weighted_shifts = np.average(np.array(shift), axis=0, weights=weights)
    linprod = np.dot(X_tab, weighted_shifts)
    return (1-utils.sigmoid(weighted_intercepts - linprod.flatten()))    

def get_w(intercepts, shift, X_tab, y_true, nens = 5):
    start = 1 / nens
   
    def opt_fun(w):
        w = w / np.sum(w)
        weighted_ens = get_ensemble(intercepts, shift, X_tab, weights = w)
        ret = metrics.log_loss(y_true, weighted_ens)
        
        if not np.isfinite(ret):
            ret = 1e6  
        return ret
    result = minimize(opt_fun, x0=np.full(nens, start), bounds=[(0, 1)] * nens, method='L-BFGS-B')
    w_optimized = result.x / np.sum(result.x)

    return w_optimized

def get_ensemble_CIB(intercepts,weights=None, n_ens=5):
    if weights is None:
        weights = np.ones(n_ens) / n_ens
    weighted_intercepts = np.average(np.array(intercepts), axis=0, weights=weights)
    return (1-utils.sigmoid(weighted_intercepts))    

def get_w_CIB(intercepts, y_true, nens = 5):
    start = 1 / nens

    # Define the optimization function
    def opt_fun(w):
        w = w / np.sum(w)
        weighted_ens = get_ensemble_CIB(intercepts, weights = w)
        ret = metrics.log_loss(y_true, weighted_ens)
        
        if not np.isfinite(ret):
            ret = 1e6  # Large value for optimization stability
        return ret

    # Perform optimization
    result = minimize(opt_fun, x0=np.full(nens, start), bounds=[(0, 1)] * nens, method='L-BFGS-B')
    w_optimized = result.x / np.sum(result.x)

    return w_optimized'''

#------------------------------------------------------------------------------------------------------------------------------------------------------
   
def get_ensemble(intercepts, 
                 shift=None,
                 X_tab=None,
                 weights=None, 
                 n_ens=5):
    
    if weights is None:
        weights = np.ones(n_ens) / n_ens

    weighted_intercepts = np.average(np.array(intercepts), axis=0, weights=weights)

    if X_tab is not None and shift is not None:
        weighted_shifts = np.average(np.array(shift), axis=0, weights=weights)
        linprod = np.dot(X_tab, weighted_shifts)
        return 1 - utils.sigmoid(weighted_intercepts - linprod.flatten())
    
    else:
        return 1 - utils.sigmoid(weighted_intercepts)
    
#------------------------------------------------------------------------------------------------------------------------------------------------------

def get_w(intercepts, 
          y_true,
          shift = None, 
          X_tab = None,
          nens = 5):
    
    start = 1 / nens
    
    if X_tab is not None and shift is not None:
        def opt_fun(w):
            w = w / np.sum(w)
            weighted_ens = get_ensemble(intercepts, shift = shift, X_tab = X_tab, weights = w)
            ret = metrics.log_loss(y_true, weighted_ens)
            
            if not np.isfinite(ret):
                ret = 1e6
            return ret
        
    else:
        def opt_fun(w):
            w = w / np.sum(w)
            weighted_ens = get_ensemble(intercepts, weights = w)
            ret = metrics.log_loss(y_true, weighted_ens)
            
            if not np.isfinite(ret):
                ret = 1e6 
            return ret

    # Perform optimization
    result = minimize(opt_fun, x0=np.full(nens, start), bounds=[(0, 1)] * nens, method='L-BFGS-B')
    w_optimized = result.x / np.sum(result.x)

    return w_optimized


