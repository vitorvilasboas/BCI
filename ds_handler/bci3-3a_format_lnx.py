# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import pandas as pd
import numpy as np
import mne
import math

"""
3 subjects (K3, K6, L1)
4 classes 
60 channels
Fs 250Hz
K3->(360 trials (90 per class)) - 2 sessions
K6,L1->(240 trials (60 per class)) - 2 sessions
"""

def labeling(labels, trueLabels):
    labels[np.where(labels==4)] = trueLabels  # Start trial t=0
    return labels

def save_class_epoch(folder,filename,epochs,labels,ds):
    X = [epochs[np.where(labels==i)] for i in [1,2,3,4]]
    for i in [1,2,3,4]: np.save(folder + 'epochs/' + filename + '_' + ds + '_' + str(i), X[i]) 

def corrigeNaN(data):
    for ch in range(data.shape[0] - 1):
        this_chan = data[ch]
        data[ch] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
        mask = np.isnan(data[ch])
        meanChannel = np.nanmean(data[ch])
        data[ch, mask] = meanChannel
    return data

def nanCleaner(epoch):
    """Removes NaN from data by interpolation
    data_in : input data - np matrix channels x samples
    data_out : clean dataset with no NaN samples"""
    for i in range(epoch.shape[0]):
        bad_idx = np.isnan(epoch[i, :])
        epoch[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], epoch[i, ~bad_idx])
    return epoch

if __name__ == '__main__':
    Fs = 250.0
    Tmin, Tmax = 0, 10 # startTrial=0; beep/cross=2; startCue=3; startMI=4; endMI=7; endTrial(break)=10
    sample_min = int(math.floor(Tmin * Fs)) # initial sample (ex. 0)
    sample_max = int(math.floor(Tmax * Fs)) # final sample (ex. 2500)
    
    folder = '/mnt/dados/datasets/BCI3_3a/' # Path to gdf files
    filename = ['K3','K6','L1']
    mne.set_log_level('WARNING','DEBUG')
    
    # ds=0
    for ds in filename:
        
        labels = np.ravel(pd.read_csv(folder + 'true_labels/trues_' + ds + '.csv'))
        raw = mne.io.read_raw_gdf(folder + ds + '.gdf')
        raw.load_data()     
        
        data = raw.get_data()
        events = raw.find_edf_events()
        
        data = corrigeNaN(data) # Correction NaN raw data
        
        timestamps = np.delete(events[0],1,axis=1)
        cond = False
        for i in [1,2,3]: cond += (timestamps[:,1] == i)
        idx = np.where(cond)[0]
        timestamps = np.delete(timestamps,idx,axis=0)
         
        # timestamps[:,1] = labeling(timestamps[:,1], labels)
        timestamps[np.where(timestamps[:,1]==4),1] = labels  # Start trial t=0 
        
        cond = False
        for i in [1,2,3,4]: cond += (timestamps[:,1] == i)
        idx = np.where(cond)[0]
        
        pos_begin = timestamps[idx,0] + sample_min
        pos_end = timestamps[idx,0] + sample_max
        n_epochs = len(timestamps[idx])
        n_channels = len(data)
        n_samples = sample_max - sample_min
        epochs = np.zeros([n_epochs, n_channels, n_samples])
        data = data[range(n_channels)]
        
        # for i in range(n_epochs): epochs[i, :, :] = data[:, pos_begin[i]:pos_end[i]] # without complete epoch verification
        bad_epoch_list = []
        for i in range(n_epochs):
            epoch = data[:, pos_begin[i]:pos_end[i]]
            if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch # Check if epoch is complete   
            else:
                print('Incomplete epoch detected...',i)
                bad_epoch_list.append(i)
        labels = np.delete(labels, bad_epoch_list)
        epochs = np.delete(epochs, bad_epoch_list, axis=0)
        
        cond = False
        for i in [5,6,7,8,9]: cond += (timestamps[:,1] == i)
        classes = timestamps[np.where(cond)]
        
        idx = np.where(classes[:,1] == 9)[0]
        epochs_test = epochs[idx]
        labels_test = labels[idx]
        XE = [ epochs_test[np.where(labels_test==i)] for i in [1,2,3,4] ] # XE = class epochs evaluate list
        np.save('../eeg_epochs/BCI3_3a/' + ds + '_E.npy', XE)
        
        idx = np.where(classes[:,1] != 9)[0]
        epochs_train = epochs[idx]
        labels_train = labels[idx]
        XT = [ epochs_train[np.where(labels_train==i)] for i in [1,2,3,4] ] # X = class epochs training list
        np.save('../eeg_epochs/BCI3_3a/' + ds + '_T.npy', XT)
        
        #save_class_epoch('../eeg_epochs/BCI3_3a/',filename[ds],epochs_test,labels_test,'E') # if 1 file per class E
        #save_class_epoch('../eeg_epochs/BCI3_3a/',filename[ds],epochs_train,labels_train,'T') # if 1 file per class T