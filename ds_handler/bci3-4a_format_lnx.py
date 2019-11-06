# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas

"""
5 subjects (aa, al, av, aw, ay)
2 classes (rh, foot)
118 channels
Fs 100Hz
280 trials (140 per class) - 1 session
Epoch distribution:
    aa : train=168 test=112  
    al : train=224 test=56
    av : train=84  test=196
    aw : train=56  test=224
    ay : train=28  test=252
"""

import numpy as np
import pickle as pck
import math
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold

def corrigeNaN(data):
    for ch in range(data.shape[0] - 1):
        this_chan = data[ch]
        data[ch] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
        mask = np.isnan(data[ch])
        meanChannel = np.nanmean(data[ch])
        data[ch, mask] = meanChannel
    return data

if __name__ == '__main__':
    
    Fs = 100
    Tmin, Tmax = 0, 5 # Start trial= 0; Start cue=0; Start MI= 0; End trial(break)= 5
    sample_min = int(math.floor(Tmin * Fs)) # initial sample (ex. 0)
    sample_max = int(math.floor(Tmax * Fs)) # final sample (ex. 2500)
    
    folder = '/mnt/dados/datasets/BCI3_4a/'
    
    subjects = ['aa','al','av','aw','ay'] 
    
    for suj in subjects:
        mat = loadmat(folder + suj + '.mat')
        cnt = 0.1 * mat['cnt'] # convert to uV
        pos = mat['mrk'][0][0][0][0]
        y = mat['mrk'][0][0][1][0]
        
        cnt = corrigeNaN(cnt.T).T
        
        true_mat = loadmat(folder + 'true_labels/trues_' + suj + '.mat')
        true_y = np.ravel(true_mat['true_y']) # RH=1 Foot=2
        # true_test_idx = np.ravel(true_mat['test_idx'])
        
        epochs = [ cnt[p+sample_min:p+sample_max] for p in pos ]
        
        epochs = np.asarray(epochs).transpose(0,2,1)
        
        X = [ epochs[np.where(true_y==i)] for i in [1,2] ]
            
        np.save('../eeg_epochs/BCI3_4a/' + suj, X)