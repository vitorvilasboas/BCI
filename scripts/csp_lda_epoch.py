#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 08:24:12 2020
@author: vboas
"""
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
    raw = mne.io.read_raw_gdf(path + '/A0'+str(suj)+'T.gdf').load_data()
    dt = raw.get_data()[:22] # [channels x samples]
    et_raw = raw.find_edf_events()
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
    ev_raw = raw.find_edf_events()
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

f_low, f_high = 8, 30
DFT = 1

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

    bmin = f_low * dft_size_band
    bmax = f_high * dft_size_band
    # print(bmin, bmax)
    XT = XT_FFT[:, :, bmin:bmax]

else: # IIR Filtering
    nyq = 0.5 * Fs
    low = f_low / nyq
    high = f_high / nyq
    if high >= 1: high = 0.99
    b, a = butter(5, [low, high], btype='bandpass')
    # b, a = iirfilter(5, [low,high], btype='band')
    # XT = lfilter(b, a, ZT) 
    XT = filtfilt(b, a, ZT)

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

#%% Feature extraction
XT_CSP = np.log(np.mean(YT ** 2, axis=2))

#%% LDA Classifier
clf = LDA()
clf.fit(XT_CSP, tT)

############################################################
#%% ################# EVALUATE UNIQUE EPOCH
############################################################

# data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01E.npy', allow_pickle=True) 
smin, smax = math.floor(0.5 * Fs), math.floor(2.5 * Fs)
epochs, labels = extractEpochs(dt_test, ev_test, smin, smax, class_ids)

for ep in range(1):
    idx = int(np.random.choice(epochs.shape[0], 1)) # escolhe uma época de teste
    Z, t = epochs[idx], labels[idx]
    
    #%%  Filtering
    if DFT:
        data_out = fft(Z)
        REAL = np.real(data_out).T
        IMAG = np.imag(data_out).T
        XFFT = np.transpose(list(itertools.chain.from_iterable(zip(IMAG, REAL))))
        bmin = f_low * dft_size_band
        bmax = f_high * dft_size_band
        # print(bmin, bmax)
        X = XFFT[:, bmin:bmax]
    else:
        nyq = 0.5 * Fs
        low = f_low / nyq
        high = f_high / nyq
        if high >= 1: high = 0.99
        b, a = butter(5, [low, high], btype='bandpass')
        # b, a = iirfilter(5, [low,high], btype='band')
        # X = lfilter(b, a, ZT) 
        X = filtfilt(b, a, ZT)
    
    #%% CSP
    Y = np.dot(Wf, X)
    features = np.log(np.mean(Y**2, axis=1))
    
    #%% LDA Classifier
    y_label = clf.predict(features.reshape(1, -1))
    y_prob = clf.predict_proba(features.reshape(1, -1))
    print('\n', idx, t, y_label==t, y_prob, '\n')