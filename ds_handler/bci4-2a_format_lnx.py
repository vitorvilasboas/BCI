# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import math
import mne
import numpy as np
from scipy.io import loadmat
"""
9 subjects (A01 to A09)
4 classes
22 channels
Fs 250Hz
576 trials (144 per classe) - 2 sessions
"""
# =============================================================================
# Dataset description MNE (Linux) (by vboas): more info in http://bbci.de/competition/iv/desc_2a.pdf
# Meta-info (Training data _T):
# 	1=1023 (rejected trial)
# 	2=768 (start trial)
# 	3=1072 (Unknown/ Eye Moviments)
# 	4=769 (Class 1 - LH - cue onset)
# 	5=770 (Class 2 - RH - cue onset)
# 	6=771 (Class 3 - Foot - cue onset)
# 	7=772 (Class 4 - Tongue - cue onset)
# 	8=277 (Eye closed)
# 	9=276 (Eye open)
# 	10=32766 (Start a new run)
# Meta-info (Test data _E):
# 	1=1023 (rejected trial)
# 	2=768 (start trial)
# 	3=1072 (Unknown/ Eye Moviments)
# 	4=783 (Cue unknown)
# 	5=277 (Eye closed)
# 	6=276 (Eye open)
# 	7=32766 (Start a new run) 
# =============================================================================

def labeling(labels, ds, trues): # normaliza rotulos de eventos conforme descrição oficial do dataset
    labels = np.where(labels==1, 1023, labels) # Rejected trial
    labels = np.where(labels==2, 768, labels) # Start trial t=0
    labels = np.where(labels==3, 1072, labels) # Eye movements / Unknown
    
    if ds=='T': # if Training dataset (A0sT.gdf)
        
        labels = np.where(labels==8, 277, labels) # Idling EEG (eyes closed) 
        labels = np.where(labels==9, 276, labels) # Idling EEG (eyes open) 
        labels = np.where(labels==10, 32766, labels) # Start of a new run/segment (after a break) 
        labels = np.where(labels==4, 769, labels) # LH (classe 1) 
        labels = np.where(labels==5, 770, labels) # RH (classe 2) 
        labels = np.where(labels==6, 771, labels) # Foot (classe 3)
        labels = np.where(labels==7, 772, labels) # Tongue (classe 4)
        
        for i in range(0, len(labels)): 
            if labels[i]==768: # rotula [1 a 4] o inicio da trial...
                if labels[i+1] == 1023: labels[i] = labels[i+2] - labels[i] # (769,770,771 ou 772) - 768 = 1,2,3 ou 4
                else: labels[i] = labels[i+1] - labels[i] # a partir da proxima tarefa (769,770,771 ou 772) - 768 = 1,2,3 ou 4
        
    else: # if Evaluate dataset (A0sE.gdf) 
        labels = np.where(labels==5, 277, labels) # Idling EEG (eyes closed) 
        labels = np.where(labels==6, 276, labels) # Idling EEG (eyes open) 
        labels = np.where(labels==7, 32766, labels) # Start of a new run/segment (after a break)
        
        # muda padrão dos rotulos desconhecidos de 4 para 783 conforme descrição oficial do dataset
        idx4 = np.where(labels==4)
        labels[idx4] = trues + 768
        
        # rotula inicios de trials no dset de validação conforme rótulos verdadeiros fornecidos (truelabels)
        idx768 = np.where(labels==768)
        labels[idx768] = trues
        
    return labels

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
    folder = '/mnt/dados/datasets/BCI4_2a/'
    dataset = ['T','E'] #,'E'
    classes = [1, 2, 3, 4]
    sujeitos = range(9,10)
    Fs = 250.0
    Tmin, Tmax = 0, 7.5 # startTrial=0; cue=2; startMI=3.25; endMI=6; endTrial=7.5-8.5
    sample_min = int(math.floor(Tmin * Fs)) # initial sample (ex. 0)
    sample_max = int(math.floor(Tmax * Fs)) # final sample (ex. 1875)
    for ds in dataset:
        for suj in sujeitos:
            ### Loading dataset with MNE package
            mne.set_log_level('WARNING','DEBUG')
            raw = mne.io.read_raw_gdf(folder + 'A0' + str(suj) + ds + '.gdf')
            raw.load_data()
            
            ### Extracting Matrix of raw data
            data = raw.get_data() # [p x q] [25 x 672528]
            data = corrigeNaN(data) # Correção de NaN nos dados brutos
            
            ### Extracting Events Info
            events = raw.find_edf_events()
            timestamps = np.delete(events[0],1,axis=1) # elimina coluna de zeros
            
            ### Loading true labels to use in evaluate files (E)
            truelabels = np.ravel(loadmat(folder + 'true_labels/A0' + str(suj) + 'E.mat')['classlabel'])
            
            ### Labeling correctly the events like competition description
            timestamps[:,1] = labeling(timestamps[:,1], ds, truelabels)
            
            ### Extracting all 288 epochs (72 per class)
            data = data[range(22)] # get only 22 EEG channels

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
            
            labels = timestamps[idx,1] # vetor que contém os indices das 288 épocas das 4 classes
        
            bad_epoch_list = []
            for i in range(n_epochs): # Check if epoch is complete
                epoch = data[:, sBegin[i]:sEnd[i]]
                if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch 
                else:
                    print('Incomplete epoch detected...')
                    bad_epoch_list.append(i)
        
            labels = np.delete(labels, bad_epoch_list) # [288]
            epochs = np.delete(epochs, bad_epoch_list, axis=0) # [288, 22, 1000]
            
            # epochs = nanCleaner(epochs) # Correção de NaN nas épocas
            
            ### Extrair épocas específicas de cada classe para cada sujeito e seçao (T ou E)
            X = [ epochs[np.where(labels==i)] for i in range(1, 5)]
            np.save('../eeg_epochs/BCI4_2a/A0' + str(suj) + ds, X)
            
            # for i in range(1, 5): np.save('../eeg_epochs/BCI4_2a/npy/A0' + str(suj) + ds + '_' + str(i), X[i-1])