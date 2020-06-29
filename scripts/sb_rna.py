#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:25:04 2020
@author: vboas
"""
import mne
import warnings
import itertools
import numpy as np
from sklearn.svm import SVC
from scipy.io import loadmat
from scipy.stats import norm
from scipy.fftpack import fft
from scipy.linalg import eigh
from bci_utils import extractEpochs, Filter
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.signal import lfilter, butter, filtfilt, firwin, iirfilter, decimate, welch
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)


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
            print('Incomplete epoch detected...', n_samples, '!=', epoch.shape[1])
            bad_epoch_list.append(i)
    labels = np.delete(labels, bad_epoch_list)
    epochs = np.delete(epochs, bad_epoch_list, axis=0)
    epochs = nanCleaner(epochs)
    return epochs, labels


def nanCleaner(epoch):
    """Removes NaN from data by interpolation
    data_in : input data - np matrix channels x samples
    data_out : clean dataset with no NaN samples"""
    for i in range(epoch.shape[0]):
        bad_idx = np.isnan(epoch[i, :])
        epoch[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], epoch[i, ~bad_idx])
    return epoch


class Filter:
    def __init__(self, fl, fh, srate, filt_info, forder=5, band_type='bandpass'):
        self.ftype = filt_info['design']
        if self.ftype != 'DFT':
            nyq = 0.5 * srate
            low = fl / nyq
            high = fh / nyq        
            if low == 0: low = 0.001
            if high >= 1: high = 0.99
            if self.ftype == 'IIR':
                forder = filt_info['iir_order']
                self.b, self.a = butter(forder, [low, high], btype=band_type) # iirfilter(btype='band')
            elif self.ftype == 'FIR':
                forder = filt_info['iir_order']
                self.b = firwin(forder, [low, high], window='hamming', pass_zero=False)
                self.a = [1]

    def apply_filter(self, X, is_epoch=False):
        if self.ftype != 'DFT': XF = lfilter(self.b, self.a, X) # lfilter, filtfilt
        else:
            XF = fft(X)
            if is_epoch:
                real, imag = np.real(XF).T, np.imag(XF).T
                XF = np.transpose(list(itertools.chain.from_iterable(zip(imag, real))))
            else:
                real = np.transpose(np.real(XF), (2, 0, 1))
                imag = np.transpose(np.imag(XF), (2, 0, 1))
                XF = np.transpose(list(itertools.chain.from_iterable(zip(imag, real))), (1, 2, 0)) 
        return XF
    
    
class CSP():
    def __init__(self, n_components):
        self.n_components = n_components
        self.filters_ = None
    def fit(self, X, y):
        ch = X.shape[1]
        classes = np.unique(y)
        X1 = X[classes[0] == y,:,:]
        X2 = X[classes[1] == y,:,:]
        S1, S2 = np.zeros((ch, ch)), np.zeros((ch, ch))  
        for i in range(len(X1)): S1 += np.dot(X1[i, :, :], X1[i, :, :].T) / X1[i].shape[-1]  # sum((X0 * X0.T)/q)
        for i in range(len(X2)): S2 += np.dot(X2[i, :, :], X2[i, :, :].T) / X2[i].shape[-1] 
        S1 /= len(X1); S2 /= len(X2)
        [D, W] = eigh(S1, S1 + S2) # + 1e-10 * np.eye(22))
        ind = np.empty(ch, dtype=int)
        ind[0::2] = np.arange(ch - 1, ch // 2 - 1, -1) 
        ind[1::2] = np.arange(0, ch // 2)
        # W += 1e-1 * np.eye(22)
        W = W[:, ind]
        self.filters_ = W.T[:self.n_components]
    def transform(self, X):        
        Y = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        # FEAT = np.log(np.mean(Y**2, axis=2))
        FEAT = np.var(Y, axis=2)
        return FEAT


if __name__ == "__main__":
    suj = 1
    class_ids = [1,2]
    path_gdf = '/mnt/dados/eeg_data/IV2a/A0'
    path_trues = '/mnt/dados/eeg_data/IV2a/true_labels/A0'
    tmin, tmax, fl, fh, ncsp, nbands = 0.5, 2.5, 4, 40, 8, 9
    filtering = {'design':'DFT'} # {'design':'IIR', 'iir_order':5}

    #######
    
    Fs = 250
    smin, smax = int(tmin*Fs), int(tmax*Fs)
    
    ### to gdf file
    eeg = mne.io.read_raw_gdf(path_gdf + str(suj) + 'T.gdf').load_data()
    data1 = eeg.get_data()[:22] # [channels x samples]
    events1 = mne.events_from_annotations(eeg) # raw.find_edf_events()
    
    eeg = mne.io.read_raw_gdf(path_gdf + str(suj) + 'E.gdf').load_data()
    data2 = eeg.get_data()[:22] # [channels x samples]
    events2 = mne.events_from_annotations(eeg) # raw.find_edf_events()
    
    # for k,v in events1[1].items(): print(f'{k}:: {v}')
    # for k,v in events2[1].items(): print(f'{k}:: {v}')
    lb_utils = [4,5,6,7]
    
    events1 = np.delete(events1[0], 1, axis=1)
    events2 = np.delete(events2[0], 1, axis=1)
    
    ZT, tt = extractEpochs(data1, events1, smin, smax, lb_utils)
    for i,k in zip(lb_utils, range(1, 5)): tt = np.where(tt == i, k, tt)
    ZT = np.vstack([ ZT[np.where(tt == k)] for k in class_ids ])
    tt = np.hstack([ np.ones(len(ZT)//2)*k for k in class_ids ]).astype(int) 
    
    ZV, _ = extractEpochs(data2, events2, smin, smax, [4])
    tv = np.ravel(loadmat(path_trues + str(suj) + 'E.mat' )['classlabel'])
    ZV = np.vstack([ ZV[np.where(tv == k)] for k in class_ids ])
    tv = np.hstack([ np.ones(len(ZV)//2)*k for k in class_ids ]).astype(int) 

    ### to npy file
    # data1, events1, _  = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + 'T.npy', allow_pickle=True) # sessão 1
    # data2, events2, _  = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + 'E.npy', allow_pickle=True) # sessão 2
    # ZT, tt = extractEpochs(data1, events1, smin, smax, class_ids)
    # ZV, tv = extractEpochs(data2, events2, smin, smax, class_ids)
    
    step = (fh-fl) / (nbands+1) # n_bins/setup['nbands']+1
    size = step / 0.5 # step/overlap
    sub_bands = []
    for i in range(nbands):
        fl_sb = i * step + fl
        fh_sb = i * step + size + fl
        sub_bands.append([fl_sb, fh_sb])
        
    XT, XV = [], []
    if filtering['design'] == 'DFT':
        filt = Filter(fl, fh, Fs, filtering)
        XTF = filt.apply_filter(ZT)
        XVF = filt.apply_filter(ZV)
        for i in range(nbands):
            bsize = 2 / ( Fs / ZT.shape[-1] ) # 2 representa sen e cos
            XT.append(XTF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])
            bsize = 2 / ( Fs / ZV.shape[-1] )
            XV.append(XVF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])         
    elif filtering['design'] == 'IIR':
        for i in range(nbands):
            filt = Filter(sub_bands[i][0], sub_bands[i][1], Fs, filtering)
            XT.append(filt.apply_filter(ZT))
            XV.append(filt.apply_filter(ZV))
            
    csp = [ CSP(n_components=ncsp) for i in range(nbands) ] # mne.decoding.CSP
    for i in range(nbands): csp[i].fit(XT[i], tt)
    FT = [ csp[i].transform(XT[i]) for i in range(nbands) ]
    FV = [ csp[i].transform(XV[i]) for i in range(nbands) ]
    
    lda = [ LDA() for i in range(nbands) ]
    for i in range(nbands): lda[i].fit(FT[i], tt)
    ST = np.asarray([ np.ravel(lda[i].transform(FT[i])) for i in range(nbands)]).T # Score LDA
    SV = np.asarray([ np.ravel(lda[i].transform(FV[i])) for i in range(nbands)]).T 
        
    p0 = norm(np.mean(ST[tt == class_ids[0], :], axis=0), np.std(ST[tt == class_ids[0], :], axis=0))
    p1 = norm(np.mean(ST[tt == class_ids[1], :], axis=0), np.std(ST[tt == class_ids[1], :], axis=0))
    META_ST = np.log(p0.pdf(ST) / p1.pdf(ST))
    META_SV = np.log(p0.pdf(SV) / p1.pdf(SV))
    
    svm = SVC(kernel='linear', C=1e-4, probability=True)
    svm.fit(META_ST, tt)
    y, yp = svm.predict(META_SV), svm.predict_proba(META_SV); 
    acc_svm = np.mean(y == tv)
    print('SVM acc:', round(svm.score(META_SV, tv)*100,2))
    
    lda = LDA()
    lda.fit(META_ST, tt)
    y, yp = lda.predict(META_SV), lda.predict_proba(META_SV)
    acc_lda = np.mean(y == tv)
    print('LDA acc:', round(lda.score(META_SV, tv)*100,2))
    
    TT, TV = np.zeros((len(ZT), 2)), np.zeros((len(ZV), 2))
    for i in range(2): TT[:,i] = np.where(tt == i+1, 1, TT[:,i])
    for i in range(2): TV[:,i] = np.where(tv == i+1, 1, TV[:,i])
    
    mlp = MLPClassifier(hidden_layer_sizes=(100,2), max_iter=10000, activation='tanh', tol=1e-4, verbose=False) #, random_state=42)
    mlp.out_activation = 'softmax' # 'logistic', 'softmax'
    # mlp.outputs = 3
    mlp.fit(META_ST, TT)
    y, yp = mlp.predict(META_SV), mlp.predict_proba(META_SV)
    print('MLP acc:', round(mlp.score(META_SV, TV)*100,2), '\n')
    
    
    # =============================================================================
    #     #FULL TRIAL 
    # =============================================================================
    smin, smax = int(-2*Fs), int(4*Fs)
    
    ### to gdf file
    trialsT, labelsT = extractEpochs(data1, events1, smin, smax, lb_utils)
    for i,k in zip(lb_utils, range(1, 5)): labelsT = np.where(labelsT == i, k, labelsT)
    trialsT = np.vstack([ trialsT[np.where(labelsT == k)] for k in class_ids ])
    labelsT = np.hstack([ np.ones(len(trialsT)//2)*k for k in class_ids ]).astype(int) 
    
    trialsV, _ = extractEpochs(data2, events2, smin, smax, [4])
    labelsV = np.ravel(loadmat(path_trues + str(suj) + 'E.mat' )['classlabel'])
    trialsV = np.vstack([ trialsV[np.where(labelsV == k)] for k in class_ids ])
    labelsV = np.hstack([ np.ones(len(trialsV)//2)*k for k in class_ids ]).astype(int) 
    
    ### to npy file
    # data1, events1, _  = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + 'T.npy', allow_pickle=True) # sessão 1
    # data2, events2, _  = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + 'E.npy', allow_pickle=True) # sessão 2
    # trialsT, labelsT = extractEpochs(data1, events1, smin, smax, class_ids)
    # trialsV, labelsV = extractEpochs(data2, events2, smin, smax, class_ids)
    
    delta_t = 0.2; delta_s = int(delta_t * Fs)
    q = int((tmax*Fs)-(tmin*Fs))  # q = largura em amostras
    ZT, tt, ZV, tv = [], [], [], []
    for k in range(len(trialsT)): # len(trials_cal)
        n = q   # n=localização em amostras (fim)
        m = q/2 # localização em amostra do ponto médio da época
        inc = (0.0 * q)  # para ser considerado comando, no min 70% da amostra (m+10%) deve estar contida no periodo de MI
        while n <= trialsT.shape[-1]:
            ZT.append(trialsT[k, :, n-q:n])
            ZV.append(trialsV[k, :, n-q:n])
            if (m <= (500+inc)) or (m >= (1500-inc)): tt.append(0); tv.append(0); 
            else: tt.append(labelsT[k]); tv.append(labelsV[k])
            m += delta_s
            n += delta_s
    ZT, ZV = np.asarray(ZT), np.asarray(ZV)
    tt, tv = np.asarray(tt), np.asarray(tv)
    
    step = (fh-fl) / (nbands+1) # n_bins/setup['nbands']+1
    size = step / 0.5 # step/overlap
    sub_bands = []
    for i in range(nbands):
        fl_sb = i * step + fl
        fh_sb = i * step + size + fl
        sub_bands.append([fl_sb, fh_sb])
        
    XT, XV = [], []
    if filtering['design'] == 'DFT':
        filt = Filter(fl, fh, Fs, filtering)
        XTF = filt.apply_filter(ZT)
        XVF = filt.apply_filter(ZV)
        for i in range(nbands):
            bsize = 2 / ( Fs / ZT.shape[-1] ) # 2 representa sen e cos
            XT.append(XTF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])
            bsize = 2 / ( Fs / ZV.shape[-1] )
            XV.append(XVF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])         
    elif filtering['design'] == 'IIR':
        for i in range(nbands):
            filt = Filter(sub_bands[i][0], sub_bands[i][1], Fs, filtering)
            XT.append(filt.apply_filter(ZT))
            XV.append(filt.apply_filter(ZV))
            
    csp = [ CSP(n_components=ncsp) for i in range(nbands) ] # mne.decoding.CSP
    for i in range(nbands): csp[i].fit(XT[i], tt)
    FT = [ csp[i].transform(XT[i]) for i in range(nbands) ]
    FV = [ csp[i].transform(XV[i]) for i in range(nbands) ]
    
    FT = np.vstack(np.transpose(FT, (0,2,1))).T    
    FV = np.vstack(np.transpose(FV, (0,2,1))).T 
    # FT = normalize(FT, norm='l2')
    # FV = normalize(FV, norm='l2')
    
    lda.fit(FT, tt)
    y, yp = lda.predict(FV), lda.predict_proba(FV)
    print('On LDA acc:', round(lda.score(FV, tv)*100,2))
     
    TT, TV = np.zeros((len(ZT), 3)), np.zeros((len(ZV), 3))
    for i in range(3): TT[:,i] = np.where(tt == i, 1, TT[:,i])
    for i in range(3): TV[:,i] = np.where(tv == i, 1, TV[:,i])
    
    # mlp = MLPClassifier(hidden_layer_sizes=(10,2), max_iter=10000, activation='tanh', tol=1e-4, verbose=False) #, random_state=42)
    # mlp.out_activation = 'softmax' # 'logistic', 'softmax'
    mlp.fit(FT, TT)
    y, yp = mlp.predict(FV), mlp.predict_proba(FV)
    print('On MLP acc:', round(mlp.score(FV, TV)*100,2), '\n') 
    # print(confusion_matrix(tv, y))