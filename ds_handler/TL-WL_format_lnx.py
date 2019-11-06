# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
"""
2 subjects (TL, WL)
2 classes (lh, rh)
Fs 250Hz
40 trials (20 per class) - TL: 2 sessions; WL:3 sessions
8 channels (1=Cz 2=Cpz 3=C1 4=C3 5=CP3 6=C2 7=C4 8=CP4)
Scalp map:
C3  C1  Cz  C2  CP4 	4  3  1  6  7
   CP3   CPz  CP4 		  5   2   8
"""
import numpy as np
import mne
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
    
    Fs = 250.0
    Tmin, Tmax = 0, 10 # Start trial= 0; Wait beep= 2; Start cue=3; Start MI= 4.25; End MI= 8; End trial(break)= 10-12
    sample_min = int(math.floor(Tmin * Fs)) # initial sample (ex. 0)
    sample_max = int(math.floor(Tmax * Fs)) # final sample (ex. 2500)
    mne.set_log_level('WARNING','DEBUG')
    folder = '/mnt/dados/datasets/BCI_CAMTUC/'
    filename = ['TL_S1','TL_S2','WL_S1','WL_S2','WL_S3']   
    
    for ds in filename: # only first run
        
        raw = mne.io.read_raw_gdf(folder + ds + '.gdf')
        raw.load_data()
             
        data = raw.get_data() # 11 x 117920
        events = raw.find_edf_events()
        
        data = corrigeNaN(data)
        
        timestamps = np.delete(events[0],1,axis=1) # elimina coluna de zeros
        timestamps = np.delete(timestamps,np.where(timestamps[:,1]==1),axis=0) # elimina marcações inuteis (cross on screen)
        timestamps = np.delete(timestamps,np.where(timestamps[:,1]==2),axis=0) # elimina marcações inuteis (feedback continuous)
        timestamps = np.delete(timestamps,np.where(timestamps[:,1]==4),axis=0) # elimina marcações inuteis (unknown)
        timestamps[:,1] = np.where(timestamps[:,1]==3, 0, timestamps[:,1]) # altera label start trial de 3 para 0
        timestamps[:,1] = np.where(timestamps[:,1]==5, 1, timestamps[:,1]) # altera label lh de 5 para 1
        timestamps[:,1] = np.where(timestamps[:,1]==6, 2, timestamps[:,1]) # altera label rh de 6 para 2
        
        idx = np.ravel(np.where(timestamps[:,1]==0)).T # capturando as amostras que iniciam cada trial
        labels = timestamps[idx+1,1] # construindo vetor de labels (target)
        
        pos_begin = timestamps[idx,0] + sample_min
        pos_end = timestamps[idx,0] + sample_max
        
        n_epochs = len(labels)
        n_channels = 8
        n_samples = sample_max - sample_min
        
        epochs = np.zeros([n_epochs, n_channels, n_samples])
        
        data = data[range(n_channels)] # 8 x 117920
        
        # for i in range(n_epochs): epochs[i, :, :] = data[:, pos_begin[i]:pos_end[i]] # no verification incomplete epoch
        
        bad_epoch_list = []
        for i in range(n_epochs): # with verification incomplete epoch
            epoch = data[:, pos_begin[i]:pos_end[i]]
            if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch # Check if epoch is complete   
            else:
                print('Incomplete epoch detected...',i)
                bad_epoch_list.append(i)
        labels = np.delete(labels, bad_epoch_list)
        epochs = np.delete(epochs, bad_epoch_list, axis=0)
        
        X = [ epochs[np.where(labels==i)] for i in [1,2] ] # X = class epochs list
        
        np.save('../eeg_epochs/BCI_CAMTUC/' + ds + '.npy', X)