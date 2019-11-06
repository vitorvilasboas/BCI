# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
"""
1 subject (CL)
3 classes (lh, rh, foot)
16 channels 
Fs 125Hz
lh-rh -> 100 trials (50 per class) 5*20 - 1 session
lh-ft -> 48 trials (24 per class) 3*16 - 1 session
"""
import numpy as np
import math

def corrigeNaN(data):
    for ch in range(data.shape[0] - 1):
        this_chan = data[ch]
        data[ch] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
        mask = np.isnan(data[ch])
        meanChannel = np.nanmean(data[ch])
        data[ch, mask] = meanChannel
    return data

if __name__ == "__main__":
    Fs = 125
    Tmin, Tmax = 0, 13 # Start trial= 0; Beep= 1; Wait= 2; Start cue= 3; Start MI= 3; End MI= 9; End trial(break)= 14
    sample_min = int(math.floor(Tmin * Fs)) # amostra inicial (ex. 0)
    sample_max = int(math.floor(Tmax * Fs)) # amostra final (ex. 2500)
    folder = '/mnt/dados/datasets/BCI_CAMTUC/'
    filename = ['CL_S1','CL_S2'] 
    
    for ds in filename:
        data = np.load(folder + ds + '_data.npy') #matriz de SAMPLES x CHANNELS 
        timestamps = np.load(folder + ds + '_events.npy') #matriz de SAMPLE_STAMP x LABEL 
        
        data = corrigeNaN(data.T).T
        
        idx = np.where(timestamps[:,1]==0)
        labels = timestamps[idx[0]+1,1] # lh=1, rh=2
        
        pos_begin = timestamps[idx[0],0] + sample_min
        pos_end = timestamps[idx[0],0] + sample_max
        
        n_epochs = len(labels)
        n_channels = 16
        n_samples = sample_max - sample_min
        
        epochs = np.zeros([n_epochs, n_channels, n_samples])
        
        data = data.T[range(n_channels)]
        
        # for i in range(n_epochs): epochs[i, :, :] = data[:, pos_begin[i]:pos_end[i]] # no verification incomplete epoch
        
        bad_epoch_list = []
        for i in range(n_epochs): # with verification incomplete epoch
            epoch = data[:, int(pos_begin[i]):int(pos_end[i])]
            if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch # Check if epoch is complete   
            else:
                print('Incomplete epoch detected...',i)
                bad_epoch_list.append(i)
        labels = np.delete(labels, bad_epoch_list)
        epochs = np.delete(epochs, bad_epoch_list, axis=0)
        
        X = [ epochs[np.where(labels==i)] for i in [1,2] ] # X = class epochs list
            
        np.save('../eeg_epochs/BCI_CAMTUC/' + ds + '.npy', X)
