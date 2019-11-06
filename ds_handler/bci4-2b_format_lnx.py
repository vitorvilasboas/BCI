# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import math
import mne
import numpy as np
from scipy.io import loadmat

"""
9 subjects
2 classes (lh, rh)
6 channels (first 3 is EEG: C3, C4, Cz; last 3 is EOG)
Fs 250Hz
120 trials (60 per class) - 5 sessions
2 sessions without feedback
3 sessions with feedback (smiley)
"""
# =============================================================================
# Dataset Description (by vboas): more info in http://bbci.de/competition/iv/desc_2b.pdf
#   01T e 02T (without feedback)
# 		10 trial * 2 classes * 6 runs * 2 sessions = 240 trials (120 per class)
# 		Cross t=0 (per 3s)
# 		beep t=2s
# 		cue t=3s (per 1.25s)
# 		MI t=4s (per 3s)
# 		Pause t=7s (per 1.5-2.5s)
# 		EndTrial t=8.5-9.5
# 
# 	03T, 04E e 05E (with feedback)
# 		10 trial * 2 classes * 4 runs * 3 sessions = 240 trials (120 per class)
# 		Smiley(grey) t=0 (per 3.5s)
# 		beep t=2s
# 		cue t=3s (per 4.5s)
# 		MI (Feedback períod) t=3.5s (per 4s)
# 		Pause t=7.5s (per 1-2s)
# 		EndTrial t=8.5-9.5
# 
# 	Meta-info 01T e 02T:
# 		1=1023 (rejected trial)
# 		2=768 (start trial)
# 		3=769 (Class 1 - LH - cue onset)
# 		4=770 (Class 2 - RH - cue onset)
# 		5=277 (Eye closed)
# 		6=276 (Eye open)
# 		7=1081 (Eye blinks)
# 		8=1078 (Eye rotation)
# 		9=1077 (Horizontal eye movement)
# 		10=32766 (Start a new run) *(to B0102T == 5)
# 		11=1078 (Vertical eye movement)
# 				
# 	Meta-info 03T:
# 		1=781 (BCI feedback - continuous)
# 		2=1023 (rejected trial)
# 		3=768 (start trial)
# 		4=769 (Class 1 - LH - cue onset)
# 		5=770 (Class 2 - RH - cue onset)
# 		6=277 (Eye closed)
# 		7=276 (Eye open)
# 		8=1081 (Eye blinks)
# 		9=1078 (Eye rotation)
# 		10=1077 (Horizontal eye movement)
# 		11=32766 (Start a new run)
# 		12=1078 (Vertical eye movement)
# 
# 
# 	Meta-info 04E e 05E:
# 		1=781 (BCI feedback - continuous)
# 		2=1023 (rejected trial)
# 		3=768 (start trial)
# 		4=783 (Cue unknown/undefined)
# 		5=277 (Eye closed)
# 		6=276 (Eye open)
# 		7=1081 (Eye blinks)
# 		8=1078 (Eye rotation)
# 		9=1077 (Horizontal eye movement)
# 		10=32766 (Start a new run)
# 		11=1078 (Vertical eye movement)
# =============================================================================


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

                       
if __name__ == '__main__': # load GDF and create NPY files
    folder = '/mnt/dados/datasets/BCI4_2b/'
    dataset = ['01T','02T','03T','04E','05E']
    classes = [1, 2]
    sujeitos = range(1,10)
    Fs = 250.0
    Tmin, Tmax = 0, 8 # startTrial=0; cue=3; startMI=4; endMI=7; endTrial=8.5-9.5
    sample_min = int(math.floor(Tmin * Fs)) # initial sample (ex. 0)
    sample_max = int(math.floor(Tmax * Fs)) # final sample (ex. 1875)
    for ds in dataset:
        for suj in sujeitos:
            ### Loading dataset with MNE package
            mne.set_log_level('WARNING','DEBUG')
            raw = mne.io.read_raw_gdf(folder + 'B0' + str(suj) + ds + '.gdf')
            raw.load_data()
            
            ### Extracting Matrix of raw data
            data = raw.get_data() # [p x q] [25 x 672528]
            data = corrigeNaN(data) # Correção de NaN nos dados brutos
            
            ### Extracting Events Info
            events = raw.find_edf_events()
            timestamps = np.delete(events[0],1,axis=1) # elimina coluna de zeros
            
            ### Loading true labels to use in evaluate files (E)
            truelabels = np.ravel(loadmat(folder + 'true_labels/B0' + str(suj) + ds + '.mat')['classlabel'])
            
            ### Labeling correctly the events like competition description
            # Remove marcações inúteis e normaliza rotulos de eventos conforme descrição oficial do dataset
            if ds in ['01T','02T']:
                for rm in range(5,12): timestamps = np.delete(timestamps,np.where(timestamps[:,1]==rm),axis=0) # detele various eye movements marks
                timestamps = np.delete(timestamps,np.where(timestamps[:,1]==1),axis=0) # delete rejected trials
                timestamps[:,1] = np.where(timestamps[:,1]==2, 0, timestamps[:,1]) # altera label start trial de 2 para 0
                timestamps[:,1] = np.where(timestamps[:,1]==3, 1, timestamps[:,1]) # altera label cue LH de 3 para 1
                timestamps[:,1] = np.where(timestamps[:,1]==4, 2, timestamps[:,1]) # altera label cue RH de 4 para 2
                for i in range(len(timestamps[:,1])):
                    if timestamps[i,1]==0: timestamps[i,1] = timestamps[i+1,1] + 768
            elif ds=='03T': 
                for rm in range(6,13): timestamps = np.delete(timestamps,np.where(timestamps[:,1]==rm),axis=0) # detele various eye movements marks
                timestamps = np.delete(timestamps,np.where(timestamps[:,1]==2),axis=0) # delete rejected trials
                timestamps = np.delete(timestamps,np.where(timestamps[:,1]==1),axis=0) # delete feedback continuous
                timestamps[:,1] = np.where(timestamps[:,1]==3, 0, timestamps[:,1]) # altera label start trial de 3 para 0
                timestamps[:,1] = np.where(timestamps[:,1]==4, 1, timestamps[:,1]) # altera label cue LH de 4 para 1
                timestamps[:,1] = np.where(timestamps[:,1]==5, 2, timestamps[:,1]) # altera label cue RH de 5 para 2
                for i in range(len(timestamps[:,1])):
                    if timestamps[i,1]==0: timestamps[i,1] = timestamps[i+1,1] + 768
            else:
                for rm in range(5,12): timestamps = np.delete(timestamps,np.where(timestamps[:,1]==rm),axis=0) # detele various eye movements marks
                timestamps = np.delete(timestamps,np.where(timestamps[:,1]==2),axis=0) # delete rejected trials
                timestamps = np.delete(timestamps,np.where(timestamps[:,1]==1),axis=0) # delete feedback continuous
                timestamps[:,1] = np.where(timestamps[:,1]==3, 0, timestamps[:,1]) # altera label start trial de 3 para 0
                timestamps[np.where(timestamps[:,1]==4),1] = truelabels #rotula momento da dica conforme truelabels
                for i in range(len(timestamps[:,1])):
                    if timestamps[i,1]==0: timestamps[i,1] = timestamps[i+1,1] + 768
            
            ### Extracting all epochs 
            data = data[range(3)] # get only 3 EEG channels

            cond = False # cond index=True if timestamps[:,1] in [1,2,3,4]
            for classe in classes: cond += (timestamps[:,1] == classe)
            
            idx = np.where(cond)[0] # storing 288 class labels indexis 
            
            t0_stamps = timestamps[idx, 0] # get sample_stamp(posições) relacionadas ao inicio das 288 tentativas de epocas das classes
            
            sBegin = t0_stamps + sample_min # vetor que marca as amostras que iniciam cada época
            sEnd = t0_stamps + sample_max # vetor que contém as amostras que finalizam cada epoca
            
            n_epochs = len(sBegin)
            n_channels = data.shape[0]
            n_samples = sample_max - sample_min
            epochs = np.zeros([n_epochs, n_channels, n_samples])
            
            labels = timestamps[idx,1] # vetor que contém os indices de todas as épocas das 2 classes
        
            bad_epoch_list = []
            for i in range(n_epochs): # Check if epoch is complete
                epoch = data[:, sBegin[i]:sEnd[i]]
                if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch 
                else:
                    print('Incomplete epoch detected...')
                    bad_epoch_list.append(i)
        
            labels = np.delete(labels, bad_epoch_list)
            epochs = np.delete(epochs, bad_epoch_list, axis=0)
            
            # epochs = nanCleaner(epochs) # Correção de NaN nas épocas
            
            ### Extrair épocas específicas de cada classe para cada sujeito e seçao (T ou E)
            X = [ epochs[np.where(labels==i)] for i in range(1, 3)]
            np.save('../eeg_epochs/BCI4_2b/B0' + str(suj) + ds, X)