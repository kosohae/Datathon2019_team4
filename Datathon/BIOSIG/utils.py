import copy

import torch
import numpy as np
from scipy.stats import pearsonr  # correlation coefficient
from sklearn.metrics import roc_curve, roc_auc_score, r2_score   # roc curve. 


def get_pearsonr(y_pred, gt):
    """
    co-efficient between variables
    """
    return pearsonr(y_pred, gt)


def _label2binary(gt, thr):
    """
    make labels to binary values.
    roc scores :  binary labels
    """
    label = gt[:]
    label[label >= thr] = 1
    label[label < thr] = 0
    return label


def get_roc_curve(y_pred, gt):
    """
    return scores
    """
    label = _label2binary(gt, 0.7)
    return roc_curve(label, y_pred)


def remove_and_replace_outlier(signal, m_val=4, ci_cut=False, verbose=False):
    """
    code for removing outlier
    """
    def outliers_index(data, m=m_val):
        if ci_cut:
            return [abs(data - np.mean(data)) > m * np.std(data)][0]
        else:
            return np.array((data > 250) | (data < 20))

    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    if type(signal) is torch.Tensor:

        try:
            signal = signal.numpy()
        except TypeError:
            signal = signal.cpu().numpy()

    if signal.ndim > 1:
        signal = np.squeeze(signal)

    if verbose:
        print('Signal')
        print(type(signal))
        print(signal.shape)
        print(signal)

    outlier_index = outliers_index(signal)
    if verbose:
        print('outlier_index')
        print(outlier_index)

    if (~outlier_index).all():
        print("No outlier detected with m = {}".format(m_val))
        if verbose:
            print('Signal min: {}'.format(np.min(signal)))
            print('Signal max: {}'.format(np.max(signal)))
        return np.expand_dims(signal, axis=1)
    else:
        print("{} of outliers detected from {} points.".format(
            outlier_index.sum(), len(outlier_index)))
        outlier_points = list(np.where(outlier_index == True))[0]
        if verbose:
            print('outlier_points')
            print(outlier_points)

    for pnt in outlier_points:
        i = copy.deepcopy(pnt)
        if verbose:
            print('Current pnt value: {}'.format(pnt))
            print('Starting i: {}'.format(i))
            print('while loop test: {}'.format(outlier_index[i]))

        while outlier_index[i] == True:
            i -= 1
            if verbose:
                print('Updated i: {}'.format(i))
        pnt_left_idx = i
        if verbose:
            print('left_idx_found : {}'.format(pnt_left_idx))
        j = copy.deepcopy(pnt)
        if verbose:
            print('Current pnt value: {}'.format(pnt))
            print('Starting j: {}'.format(j))
            print('while loop test: {}'.format(outlier_index[j]))

        while outlier_index[j] == True:
            j += 1
            if verbose:
                print('Updated j: {}'.format(j))
        pnt_right_idx = j
        if verbose:
            print('right_idx_found : {}'.format(pnt_right_idx))

        interpolated = signal[pnt_left_idx] + signal[pnt_right_idx] / 2
        if verbose:
            print(
                'interpolated to: {} --> {}'.format(signal[pnt], interpolated))
        signal[pnt] = interpolated
    return np.expand_dims(signal, axis=1)
