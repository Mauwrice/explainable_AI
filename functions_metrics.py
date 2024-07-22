import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from matplotlib import pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy import special


# Computes the confidence interval of AUC using bootstrapping
# adapted from https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals
def compute_auc_ci(y_true, y_pred, n_bootstraps=1000, rng_seed=42, alpha=0.05):
    # y_true: true binary labels
    # y_pred: target scores, can either be probability estimates of the positive class, 
    #         confidence values, or non-thresholded measure of decisions
    # n_bootstraps: number of bootstrap samples to use
    # rng_seed: seed for the random number generator
    # alpha: significance level (type I error rate)    
    
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = metrics.roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    confidence_lower = sorted_scores[int(alpha/2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(1-alpha/2 * len(sorted_scores))]
    
    return (confidence_lower, confidence_upper)


# Computes a classification report for a binary model 
#  including AUC, NLL, sensitivity, specificity, and accuracy
def bin_class_report(X_test,y_test, model):
    # X_test: test set features
    # y_test: test set labels
    # model: trained model
    
    NLL, Acc, AUC0 = model.evaluate( x=X_test, y=y_test, verbose=0)
    y_pred = model.predict(X_test)
    
    if y_pred.shape[1] == 2:
        y_test = y_test[:,1]
        y_pred = y_pred[:,1]
    
    # cm , AUC
    cm = confusion_matrix(np.round(y_test), np.round(y_pred))
    AUC =  metrics.roc_auc_score(np.round(y_test), np.round(y_pred))
    # AUC CI
    AUC_CI = compute_auc_ci(np.round(y_test), np.round(y_pred))
    AUC0_CI = compute_auc_ci(y_test, y_pred)
    #acc
    nobs = sum(sum(cm))
    count = sum([cm[0,0], cm[1,1]])
    Acc = count/nobs
    acc_ci_low, acc_ci_upp = proportion_confint(count , nobs,  alpha=0.05, method='wilson')
    #sens 
    sens = cm[1,1]/(cm[1,1]+cm[1,0])
    nobs = sum([cm[1,0],cm[1,1]])
    count = sum([cm[1,1]])
    sens_ci_low, sens_ci_upp = proportion_confint(count , nobs,  alpha=0.05, method='wilson')
    #spec 
    spec = cm[0,0]/(cm[0,1]+cm[0,0])
    nobs = sum([cm[0,1],cm[0,0]])
    count = sum([cm[0,0]])
    spec_ci_low, spec_ci_upp = proportion_confint(count , nobs,  alpha=0.05, method='wilson')
    
    print("\nPerformance on Test Set : ")
    print("\nAccuracy    [95% Conf.] :", np.around(Acc,4),np.around([acc_ci_low, acc_ci_upp],4))
    print("Sensitivity [95% Conf.] :", np.around(sens,4), np.around([sens_ci_low, sens_ci_upp],4))
    print("Specificity [95% Conf.] :", np.around(spec,4), np.around([spec_ci_low, spec_ci_upp],4))
    print("\nArea under Curve (AUC) Binary [95% Conf.]:", np.around(AUC,4),np.around([AUC_CI[0], AUC_CI[1]],4))
    print("Area under Curve (AUC) Probability [95% Conf.]:", np.around(AUC0,4),np.around([AUC0_CI[0], AUC0_CI[1]],4))
    print("Negative Log-Likelihood :", np.around(NLL, 4))
    return (AUC, NLL, sens, spec)

def compute_bootstrap_ci(y_true, y_pred, n_bootstraps=10000, rng_seed=42, alpha=0.05, 
                         metric = metrics.roc_auc_score, **kwargs):
    # y_true: true binary labels
    # y_pred: target scores, can either be probability estimates of the positive class, 
    #         confidence values, or non-thresholded measure of decisions
    # n_bootstraps: number of bootstrap samples to use
    # rng_seed: seed for the random number generator
    # alpha: significance level (type I error rate)  
    # metric: metric to be used  
    
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = metric(y_true[indices], y_pred[indices], **kwargs)
        bootstrapped_scores.append(score)
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    confidence_lower = sorted_scores[int(alpha/2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(1-alpha/2 * len(sorted_scores))]
    
    return (confidence_lower, confidence_upper)

# Computes a classification report for a binary model 
#  including AUC, NLL, sensitivity, specificity, and accuracy
def bin_class_report_ontram(y_true, y_pred_binary, y_pred):
    # y_true: true values
    # y_pred_binary: binarized predictions
    # y_pred: transformed average predictions (probabilities)
      
    # cm , AUC
    cm = confusion_matrix(y_true, y_pred_binary)
    AUC =  metrics.roc_auc_score(y_true, y_pred_binary)
    AUC0 =  metrics.roc_auc_score(y_true, y_pred)
    # AUC CI
    AUC_CI = compute_bootstrap_ci(y_true, y_pred_binary, metric=metrics.roc_auc_score)
    AUC0_CI = compute_bootstrap_ci(y_true, y_pred, metric=metrics.roc_auc_score)
    #acc
    nobs = sum(sum(cm))
    count = sum([cm[0,0], cm[1,1]])
    Acc = count/nobs
    acc_ci_low, acc_ci_upp = proportion_confint(count , nobs,  alpha=0.05, method='wilson')
    #sens 
    sens = cm[1,1]/(cm[1,1]+cm[1,0])
    nobs = sum([cm[1,0],cm[1,1]])
    count = sum([cm[1,1]])
    sens_ci_low, sens_ci_upp = proportion_confint(count , nobs,  alpha=0.05, method='wilson')
    #spec 
    spec = cm[0,0]/(cm[0,1]+cm[0,0])
    nobs = sum([cm[0,1],cm[0,0]])
    count = sum([cm[0,0]])
    spec_ci_low, spec_ci_upp = proportion_confint(count , nobs,  alpha=0.05, method='wilson')
    # nll
    nll = metrics.log_loss(y_true, y_pred)
    nll_ci_low, nll_ci_upp = compute_bootstrap_ci(y_true, y_pred, metric=metrics.log_loss)
    # f1
    f1_value = metrics.f1_score(y_true, y_pred_binary)
    f1_ci_low, f1_ci_upp = compute_bootstrap_ci(y_true, y_pred_binary, metric=metrics.f1_score)

    
    print("Performance on Test Set : ")
    print("\nAccuracy    [95% Conf.] :", np.around(Acc,4), np.around([acc_ci_low, acc_ci_upp],4))
    print("Sensitivity [95% Conf.] :", np.around(sens,4), np.around([sens_ci_low, sens_ci_upp],4))
    print("Specificity [95% Conf.] :", np.around(spec,4), np.around([spec_ci_low, spec_ci_upp],4))
    print("F1 Score [95% Conf.] :", np.around(f1_value,4), np.around([f1_ci_low, f1_ci_upp],4))
    print("Area under Curve (AUC) Binary [95% Conf.]:", np.around(AUC,4),np.around([AUC_CI[0], AUC_CI[1]],4))
    print("Area under Curve (AUC) Probability [95% Conf.]:", np.around(AUC0,4),np.around([AUC0_CI[0], AUC0_CI[1]],4))
    print("Negative Log-Likelihood :", np.around(nll, 4), np.around([nll_ci_low, nll_ci_upp],4))

# Computes a classification report for a binary model for given predictions and labels
def calc_metrics(y, p):
    # y: true binary labels
    # p: predicted probabilities
    
#     NLL = np.mean(-special.xlogy(y, p) - special.xlogy(1-y, 1-p))
    NLL = tf.keras.losses.binary_crossentropy(y, p)
    AUC =  metrics.roc_auc_score(y, p)
    AUCB =  metrics.roc_auc_score(y, np.round(p))
    
    cm = confusion_matrix(np.round(y), np.round(p))
    nobs = sum(sum(cm))
    count = sum([cm[0,0], cm[1,1]])
    ACC = count/nobs
    
    print("\nPerformance on Test Set : ")
    print("Accuracy:", np.around(ACC,4))
    print("Area under Curve (AUC) Binary :", np.around(AUCB,4))
    print("Area under Curve (AUC) Probability :", np.around(AUC,4))
    print("Negative Log-Likelihood :", np.around(NLL, 4))
    print("Confusion Matrix : \n", cm)

# Calculates the data for a calibration plot
# 4 bins are calculated based on the quantiles of the predicted probabilities
# the confidence intervals are calculated using the Wilson score interval
def cal_plot_data_prep(y_pred, y_test):
    # y_pred: predicted probabilities
    # y_test: true binary labels
    
    # create cuts
    cuts = np.quantile(y_pred, q = [0.25, 0.5, 0.75])
    cuts = np.insert(cuts, obj = 0, values = 0)
    cuts = np.append(cuts, 1)
    
    # get indices of cuts
    cut_idx = []
    bin_cuts = []
    for i in range(len(cuts) - 1):
        cut_idx.append(np.where((y_pred >= cuts[i]) & (y_pred < cuts[i+1]))[0]) 
        bin_cuts.append(str(np.round(cuts[i:i+2], 4)))
        
    pred_prob = []
    obs_prop = []
    obs_prop_up = []
    obs_prop_lo = []
    obs_cases = []

    # calc confint
    for idx in cut_idx:
        pred_prob.append(np.mean(y_pred[idx]))
        obs_prop.append(np.mean(y_test[idx]))
        obs_cases.append(np.sum(y_test[idx]))
        confint = proportion_confint(count = np.sum(y_test[idx]), 
                                 nobs = len(y_test[idx]), 
                                 alpha = 0.05, 
                                 method = "wilson")
        obs_prop_lo.append(confint[0])
        obs_prop_up.append(confint[1])
        
    # save as df
    cal_plot_data = pd.DataFrame(
        {"predicted_probability_mean": pred_prob,
         "predicted_probability_middle": np.convolve(cuts,[.5,.5],"valid"),
         "observed_proportion": obs_prop,
         "observed_proportion_lower": obs_prop_lo,
         "observed_proportion_upper": obs_prop_up,
         "observed_cases": obs_cases, # number of cases with bad outcome
         "bin_cuts": bin_cuts
        }
    )
    
    return cal_plot_data

# Plots a calibration plot for data created by cal_plot_data_prep
def cal_plot(dat, x_vals, y_vals, lwr_vals, upr_vals, alpha = 1, show = True, col = "C0", label = None):
    # dat: data created by cal_plot_data_prep
    # x_vals: column name of the x values
    # y_vals: column name of the y values
    # lwr_vals: column name of the lower confidence interval values
    # upr_vals: column name of the upper confidence interval values
    # alpha: alpha value for the plot
    # show: whether to show the plot
    # col: color of the plot
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    sns.lineplot(
        data=dat, x=x_vals, y=y_vals, 
        marker="o",
        alpha = alpha,
        color = col,
        label = label
    )

    plt.plot([0, 1], [0, 1], c = "grey", linewidth = 2)
    for i in range(dat.shape[0]):
        plt.axvline(dat[x_vals].tolist()[i], 
                    dat[lwr_vals].tolist()[i], 
                    dat[upr_vals].tolist()[i],
                    alpha = alpha,
                    color = col)
    if show:
        plt.show()
        
# inverse sigmoid function
# for trafo averaging
def inverse_sigmoid(y):
    return np.log(y/(1-y))

# sigmoid function 
# for trafo averaging
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# weighted variance with axis argument
# adjusted from https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
def wght_variance(values, weights=None, axis=None):
    """
    Return the weighted standard deviation.
    axis -- the axis for std calculation
    values, weights -- Numpy ndarrays with the same shape on the according axis.
    """
    average = np.expand_dims(np.average(values, weights=weights, axis=axis), axis=axis)
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return variance

def normalize_heatmap(hm, both_directions = False, hm_min_max = None):
    # hm: heatmap to normalize
    # both_directions: whether to normalize to [-1,1] or [0,1]
    #
    # returns normalized heatmap

    if hm_min_max is None:
        hm_min_max = [np.min(hm), np.max(hm)]

    if not both_directions:
        hm = ((hm - hm_min_max[0])/(hm_min_max[1]-hm_min_max[0]))
    elif both_directions:
        hm_abs_max = np.max(np.abs(hm_min_max))
        hm = hm / hm_abs_max

    return hm

def min_max_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))