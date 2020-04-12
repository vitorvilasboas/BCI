#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 08:24:12 2020
@author: vboas
"""
import warnings
import mne
import math
import itertools
import numpy as np
from scipy.linalg import eigh
from scipy.fftpack import fft
# from bci_utils import extractEpochs
from scipy.io import loadmat
from scipy.signal import lfilter, butter, iirfilter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

warnings.filterwarnings("ignore", category=DeprecationWarning)

def extractEpochs(data, events, smin, smax, class_ids):
    events_list = events[:, 1] # get class labels column
    cond = False
    for i in range(len(class_ids)): cond += (events_list == class_ids[i]) #get only class_ids pos in events_list
    idx = np.where(cond)[0]
    s0 = events[idx, 0] # get initial timestamps of each class epochs
    sBegin = s0 + smin
    sEnd = s0 + smax
    n_epochs = len(sBegin)
    n_channels = data.shape[0]
    n_samples = smax - smin
    epochs = np.zeros([n_epochs, n_channels, n_samples])
    labels = events_list[idx]
    bad_epoch_list = []
    for i in range(n_epochs):
        epoch = data[:, sBegin[i]:sEnd[i]]
        if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch # Check if epoch is complete
        else:
            print('Incomplete epoch detected...')
            bad_epoch_list.append(i)
    labels = np.delete(labels, bad_epoch_list)
    epochs = np.delete(epochs, bad_epoch_list, axis=0)
    return epochs, labels

def labeling(path, suj):   
    #mne.set_log_level(50, 50)
    raw = mne.io.read_raw_gdf(path + '/A0'+str(suj)+'T.gdf').load_data()
    dt = raw.get_data()[:22] # [channels x samples]
    et_raw = mne.events_from_annotations(raw) # raw.find_edf_events()
    et = np.delete(et_raw[0], 1, axis=1) # remove MNE zero columns
    et = np.delete(et,np.where(et[:,1]==1), axis=0) # remove rejected trial
    et = np.delete(et,np.where(et[:,1]==3), axis=0) # remove eye movements/unknown
    et = np.delete(et,np.where(et[:,1]==8), axis=0) # remove eyes closed
    et = np.delete(et,np.where(et[:,1]==9), axis=0) # remove eyes open 
    et = np.delete(et,np.where(et[:,1]==10), axis=0) # remove start of a new run/segment
    et[:,1] = np.where(et[:,1]==2, 0, et[:,1]) # start trial t=0
    et[:,1] = np.where(et[:,1]==4, 1, et[:,1]) # LH 
    et[:,1] = np.where(et[:,1]==5, 2, et[:,1]) # RH 
    et[:,1] = np.where(et[:,1]==6, 3, et[:,1]) # Foot
    et[:,1] = np.where(et[:,1]==7, 4, et[:,1]) # Tongue
    for i in range(0, len(et)):
        if et[i,1]==0: et[i,1] = (et[i+1,1]+10) # labeling start trial [11 a 14] according cue [1,2,3,4]
             
    raw = mne.io.read_raw_gdf(path + '/A0'+str(suj)+'E.gdf').load_data()
    trues = np.ravel(loadmat(path + '/true_labels/A0'+str(suj)+'E.mat' )['classlabel'])
    dv = raw.get_data()[:22] # [channels x samples]
    ev_raw = mne.events_from_annotations(raw) #raw.find_edf_events()
    ev = np.delete(ev_raw[0], 1, axis=1) # remove MNE zero columns
    ev = np.delete(ev,np.where(ev[:,1]==1), axis=0) # remove rejected trial
    ev = np.delete(ev,np.where(ev[:,1]==3), axis=0) # remove eye movements/unknown
    ev = np.delete(ev,np.where(ev[:,1]==5), axis=0) # remove eyes closed
    ev = np.delete(ev,np.where(ev[:,1]==6), axis=0) # remove eyes open
    ev = np.delete(ev,np.where(ev[:,1]==7), axis=0) # remove start of a new run/segment
    ev[:,1] = np.where(ev[:,1]==2, 0, ev[:,1]) # start trial t=0
    ev[np.where(ev[:,1]==4),1] = trues # change unknown value labels(4) to value in [1,2,3,4]
    for i in range(0, len(ev)):
        if ev[i,1]==0: ev[i,1] = (ev[i+1,1]+10) # labeling start trial [11 a 14] according cue [1,2,3,4]
    
    return dt, et, dv, ev


path = '/mnt/dados/eeg_data/IV2a/gdf' ## >>> SET HERE THE DATA SET PATH
subject = 1
Fs = 250
class_ids = [1, 2]

dt_train, ev_train, dt_test, ev_test = labeling(path, subject)

# data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01T.npy', allow_pickle=True) 
smin, smax = math.floor(0.5 * Fs), math.floor(2.5 * Fs)
epochs, labels = extractEpochs(dt_train, ev_train, smin, smax, class_ids)
ZT = [epochs[np.where(labels==i)] for i in class_ids]
ZT = np.r_[ZT[0],ZT[1]]
tT = np.r_[class_ids[0]*np.ones(int(len(ZT)/2)), class_ids[1]*np.ones(int(len(ZT)/2))]

# data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01E.npy', allow_pickle=True) 
smin, smax = math.floor(0.5 * Fs), math.floor(2.5 * Fs)
epochs, labels = extractEpochs(dt_test, ev_test, smin, smax, class_ids)
ZV = [epochs[np.where(labels==i)] for i in class_ids]
ZV = np.r_[ZV[0],ZV[1]]
tV = np.r_[class_ids[0]*np.ones(int(len(ZV)/2)), class_ids[1]*np.ones(int(len(ZV)/2))]

f_low, f_high = 8, 30
DFT = 0

#%% Filtering
if DFT:
    buffer_len = smax - smin
    dft_res_freq = Fs/buffer_len # resolução em frequência fft
    dft_size_band = round(2/dft_res_freq) # 2 representa sen e cos que foram separados do componente complexo da fft intercalados
    
    data_out = fft(ZT)
    REAL = np.transpose(np.real(data_out), (2, 0, 1))
    IMAG = np.transpose(np.imag(data_out), (2, 0, 1))
    data_out = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    XT_FFT = np.transpose(data_out, (1, 2, 0))
    
    data_out = fft(ZV)
    REAL = np.transpose(np.real(data_out), (2, 0, 1))
    IMAG = np.transpose(np.imag(data_out), (2, 0, 1))
    data_out = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    XV_FFT = np.transpose(data_out, (1, 2, 0))
    
    bmin = f_low * dft_size_band
    bmax = f_high * dft_size_band
    # print(bmin, bmax)
    XT = XT_FFT[:, :, bmin:bmax]
    XV = XV_FFT[:, :, bmin:bmax]

else: # IIR Filtering
    nyq = 0.5 * Fs
    low = f_low / nyq
    high = f_high / nyq
    if high >= 1: high = 0.99
    b, a = butter(5, [low, high], btype='bandpass')
    # b, a = iirfilter(5, [low,high], btype='band')
    # XT = lfilter(b, a, ZT) 
    # XV = lfilter(b, a, ZV)
    XT = filtfilt(b, a, ZT)
    XV = filtfilt(b, a, ZV)


#%% CSP
ncomp = 8
e, c, s = XT.shape
classes = np.unique(tT)   
Xa = XT[classes[0] == tT,:,:]
Xb = XT[classes[1] == tT,:,:]

Sa = np.zeros((c, c)) 
Sb = np.zeros((c, c))
for i in range(int(e/2)):
    # Sa += np.dot(Xa[i,:,:], Xa[i,:,:].T)
    # Sb += np.dot(Xb[i,:,:], Xb[i,:,:].T)
    Sa += np.dot(Xa[i,:,:], Xa[i,:,:].T) / Xa[i].shape[-1] # sum((Xa * Xa.T)/q)
    Sb += np.dot(Xb[i,:,:], Xb[i,:,:].T) / Xb[i].shape[-1] # sum((Xb * Xb.T)/q)
Sa /= len(Xa)
Sb /= len(Xb)

[D, W] = eigh(Sa, Sa + Sb)
ind = np.empty(c, dtype=int)
ind[0::2] = np.arange(c - 1, c // 2 - 1, -1) 
ind[1::2] = np.arange(0, c // 2)
W = W[:, ind]
Wf = W.T[:ncomp]
       
YT = np.asarray([np.dot(Wf, ep) for ep in XT])
YV = np.asarray([np.dot(Wf, ep) for ep in XV])


#%% Feature extraction
XT_CSP = np.log(np.mean(YT ** 2, axis=2))
XV_CSP = np.log(np.mean(YV ** 2, axis=2))
# XV_CSPi = np.log(np.mean(YV[0] ** 2, axis=1))


#%% LDA Classifier
clf = LDA()
clf.fit(XT_CSP, tT)
scores_labels = clf.predict(XV_CSP)
acc = np.mean(scores_labels == tV)
print('\nAccuracy:', round(acc,4))
