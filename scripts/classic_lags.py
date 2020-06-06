#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:30:07 2020
@author: vboas
"""
import math
import itertools
import numpy as np
from scipy.linalg import eigh
from scipy.fftpack import fft
from sklearn.metrics import cohen_kappa_score
from bci_utils import labeling, extractEpochs
from scipy.signal import lfilter, butter, iirfilter, filtfilt, decimate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
           
#%% DATASET AND SCENARIO INFO
''' III3a: subjects={'K3','K6','L1'}; prefix=''; class_ids={1,2,3,4}; sessions={None}; channels=[:60]     
    III4a: subjects={'aa','al','av','aw','ay'}; prefix=''; class_ids={1,3}; sessions={None}; channels=[:118]
    IV2a: subjects={1,2,...,9}; prefix='A0'; class_ids={1,2,3,4} ; sessions={'T','E'} ; channels=[:22]
    IV2b: subjects={1,2,...,9}; prefix='B0'; class_ids={1,2}; sessions={'01T','02T','03T','04E','05E'}; channels=[:3]        
    Lee19: subjects={1,2,...,54}; prefix='S'; class_ids={1,2}; sessions={1,2}; channels=[:62]; suffix='sess'; ch_cortex=[7,32,8,9,33,10,34,12,35,13,36,14,37,17,38,18,39,19,40,20] 
'''

ds = 'IV2a' # 'IV2a','IV2b','III3a','III4a','Lee19'      
path = '/mnt/dados/eeg_data/' + ds + '/' # PATH TO DATASET
suj = 5
class_ids = [1, 2]
sessions = ['T', 'E']
channels = None
prefix, suffix = 'A0', ''

#%% Load data
d_train, e_train, i_train = labeling(path=path, ds=ds, session=sessions[0], subj=suj, channels=channels, save=False)
# d_train, e_train, i_train = np.load(path + 'npy/' + prefix + str(suj) + '' + sessions[0] + '.npy', allow_pickle=True)

if not ds in ['III3a','III4a']: 
    d_test, e_test, i_test = labeling(path=path, ds=ds, session=sessions[1], subj=suj, channels=channels, save=False)
    # d_test, e_test, i_test = np.load(path + 'npy/' + prefix + str(suj) + '' + sessions[1] + '.npy', allow_pickle=True)

#%% Segmentation
# Fs = 250 if dataset in ['IV2a', 'IV2b', 'III3a', 'Lee19'] else 100
Fs = i_train['fs']

# # Downsampling
# factor = 2
# d_train = decimate(d_train, factor)
# d_test = decimate(d_test, factor)
# Fs = Fs/factor
# # et = np.copy(e_train)
# # ev = np.copy(e_test)
# e_train[:, 0] = [round(e_train[i, 0]/factor) for i in range(e_train.shape[0])]
# e_test[:, 0] = [round(e_test[i, 0]/factor) for i in range(e_test.shape[0])]

smin, smax = math.floor(0.5 * Fs), math.floor(2.5 * Fs)

L = 2
ZTL, ZVL = [], []
for l in range(0, L + 1):
    # print(smin+l, smax+l)
    
    
    ################
    epochsT, labelsT = extractEpochs(d_train, e_train, smin+l, smax+l, class_ids)
    
    if not ds in ['III3a','III4a']: 
        epochsV, labelsV = extractEpochs(d_test, e_test, smin+l, smax+l, class_ids)
    else: 
        epochs, labels = np.copy(epochsT), np.copy(labelsT)
        test_size = int(len(epochs) * 0.5)
        train_size = int(len(epochs) - test_size)
        train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
        epochsT, labelsT = epochs[:train_size], labels[:train_size] 
        epochsV, labelsV = epochs[train_size:], labels[train_size:]
    
    ZT = [epochsT[np.where(labelsT==i)] for i in class_ids]
    ZT = np.r_[ZT[0],ZT[1]]
    tT = np.r_[class_ids[0]*np.ones(int(len(ZT)/2)), class_ids[1]*np.ones(int(len(ZT)/2))]
    
    ZV = [epochsV[np.where(labelsV==i)] for i in class_ids]
    ZV = np.r_[ZV[0],ZV[1]]
    tV = np.r_[class_ids[0]*np.ones(int(len(ZV)/2)), class_ids[1]*np.ones(int(len(ZV)/2))]
    #################
    
    
    ZTL.append(np.transpose(ZT, (0,2,1)))
    ZVL.append(np.transpose(ZV, (0,2,1)))

ZT, ZV = ZTL[0], ZVL[0]
for i in range(1, len(ZTL)):
    ZT = np.c_[ZT,ZTL[i]]
    ZV = np.c_[ZV,ZVL[i]]

ZT = np.transpose(ZT, (0,2,1))
ZV = np.transpose(ZV, (0,2,1))

print(ZT.shape, ZV.shape)


#%% Filtering
f_low, f_high = 8, 30
DFT = 0 # 0=IIR, 1=DFT

if DFT:
    print(DFT)
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
    if low == 0: low = 0.001
    if high >= 1: high = 0.99
    b, a = butter(5, [low, high], btype='bandpass')
    # b, a = iirfilter(5, [low,high], btype='band')
    # XT = lfilter(b, a, ZT) 
    # XV = lfilter(b, a, ZV)
    XT = filtfilt(b, a, ZT)
    XV = filtfilt(b, a, ZV)


#%% CSP
ncomp = 6
e, c, s = XT.shape
classes = np.unique(tT)   
Xa = XT[classes[0] == tT,:,:]
Xb = XT[classes[1] == tT,:,:]

Sa = np.zeros((c, c)) 
Sb = np.zeros((c, c))
for i in range(int(e/2)):
    # Sa += np.dot(Xa[i,:,:], Xa[i,:,:].T)
    # Sb += np.dot(Xb[i,:,:], Xb[i,:,:].T)
    Sa += np.dot(Xa[i,:,:], Xa[i,:,:].T) #/ Xa[i].shape[-1] # sum((Xa * Xa.T)/q)
    Sb += np.dot(Xb[i,:,:], Xb[i,:,:].T) #/ Xb[i].shape[-1] # sum((Xb * Xb.T)/q)
Sa /= (len(Xa) * s)
Sb /= (len(Xb) * s)

[D, W] = eigh(Sa, Sa + Sb)
ind = np.empty(c, dtype=int)
ind[0::2] = np.arange(c - 1, c // 2 - 1, -1) 
ind[1::2] = np.arange(0, c // 2)
W = W[:, ind]
print(W.shape)
Wf = W.T[:ncomp]
       
YT = np.asarray([np.dot(Wf, ep) for ep in XT])
YV = np.asarray([np.dot(Wf, ep) for ep in XV])


#%% Feature extraction
XT_CSP = np.log(np.mean(YT ** 2, axis=2))
XV_CSP = np.log(np.mean(YV ** 2, axis=2))
# XT_CSP = np.log(np.var(YT, axis=2))
# XV_CSP = np.log(np.var(YV, axis=2))
# XV_CSPi = np.log(np.mean(YV[0] ** 2, axis=1))

#%% LDA Classifier
clf = LDA()
clf.fit(XT_CSP, tT)
scores_labels = clf.predict(XV_CSP)
acc = np.mean(scores_labels == tV)
kappa = cohen_kappa_score(scores_labels, tV)
print('Accuracy:', round(acc,4))
print('kappa:', round(kappa,4))

    #%%


# def load_data(sujeito, classes):
#     ds = ['T_', 'E_']
#     X = []  # vetor de dimensão 4 onde as duas primeiras contém os dados de treinamento das duas classes e as duas últimas os dados de validação das duas classes      
#     for i in range(2):
#         for j in range(2):
#             path = '/home/vboas/devto/datasets/BCICIV_2a/npy/A0' + str(sujeito) + ds[i] + str(classes[j]) + '.npy'
#             dados = load(open(path, 'rb'))
#             X.append(dados)
#     return X

# def windowing(X, fs, t_0, t_start, t_end, atraso):
#     W_Start = int((t_start - t_0) * fs)
#     W_End = int((t_end - t_0) * fs)
#     XJ = []
#     atraso = atraso
#     for i in range(4):
#         # XJ.append(XF[i][:, :, W_Start:W_End])
#         janela = transpose(X[i][:,:,W_Start:W_End], (1, 0, 2))
#         for cont in range(1, atraso + 1):
#             jAvancada = transpose(X[i][:,:,W_Start+cont:W_End+cont], (1, 0, 2)) # avanço (janela - cont)
#             jAtrasada = transpose(X[i][:,:,W_Start+cont:W_End+cont], (1, 0, 2)) # atraso (janela + cont)
#             janela = concatenate([janela, jAtrasada])
#         #print(janela[0][0][0],' é igual a ',janela[22][0][1])
#         XJ.append(transpose(janela, (1, 0, 2)))
#     #print(XJ[0][0][0][498],' é igual a ',XJ[0][0][22][499])
#     return XJ

# def filtragemFFT2(XJ, fs, fl, fh):
#     nf = fs/2.
#     bin0 = int(fl * (fs/nf))  # para fl = 8 bin0 = 15
#     binN = int(fh * (fs/nf)) # para fl = 8 bin0 = 15
#     # print bin0, binN
#     XF = []
#     for i in range(4): # filtra os dados de treinamento e validação das duas classes 
#         filtrado = fft(XJ[i])
#         REAL = transpose(real(filtrado)[:,:,bin0:binN], (2, 0, 1)) #transpoe para intercalar
#         IMAG = transpose(imag(filtrado)[:,:,bin0:binN], (2, 0, 1)) #transpoe para intercalar
#         filtrado = list(itertools.chain.from_iterable(zip(IMAG, REAL))) #intercalando
#         filtrado = transpose(filtrado, (1, 2, 0)) # retorna ao formato original      
#         XF.append(filtrado)
#     return XF

# def filtragemIIR(XJ, fs, fl, fh, ordem=5):
#     nf = fs/2.
#     b, a = butter(ordem, [fl/nf, fh/nf], btype='bandpass')
#     XF = []
#     for i in range(4): # filtra os dados de treinamento e validação das duas classes 
#         filtrado = lfilter(b, a, XJ[i])
#         XF.append(filtrado)
#     return XF

# class CSP():
#     def __init__(self, n_components):
#         self.n_components = n_components

#     def fit(self, X, y):
#         e, c, t = X.shape
#         classes = unique(y)
        
#         X0 = X[classes[0] == y,:,:]
#         X1 = X[classes[1] == y,:,:]

#         # Sum up covariance matrix
#         S0 = zeros((c, c))
#         S1 = zeros((c, c))
#         for i in range(int(e/2)): # add conversão int() ?
#             S0 += dot(X0[i,:,:], X0[i,:,:].T)
#             S1 += dot(X1[i,:,:], X1[i,:,:].T)

#         [D, W] = eigh(S0, S0 + S1)

#         ind = empty(c, dtype=int)
#         ind[0::2] = arange(c - 1, c // 2 - 1, -1)
#         ind[1::2] = arange(0, c // 2)
        
#         W = W[:, ind]
        
#         self.filters_ = W.T[:self.n_components]

#     def transform(self, X):
#         XT = asarray([dot(self.filters_, epoch) for epoch in X])
#         XVAR = log(mean(XT ** 2, axis=2))
        
#         return XVAR
    
# def sbcsp(sujeito, classes, args):
#     atraso, t_filtro, n_componentes, fs, fl, fh, ordem = args  
    
#     X = load_data(sujeito, classes)
    
#     t0 = time()
    
#     XJ = windowing(X, fs, 0, 2.5, 4.5, atraso)
    
#     if t_filtro == 'IIR': XF = filtragemIIR(XJ, fs, fl, fh)
#     elif t_filtro == 'FFT': XF = filtragemFFT2(XJ, fs, fl, fh)

#     XT = concatenate([XF[0],XF[1]]) # treinamento A e B
#     XV = concatenate([XF[2],XF[3]]) # validação A e B
    
#     y = concatenate([zeros(72), ones(72)]) #vetor gabarito
    
#     # CSP
#     csp = CSP(n_components=n_componentes)
#     csp.fit(XT, y)
#     XT_CSP = csp.transform(XT)
#     XV_CSP = csp.transform(XV) 
    
#     # LDA
#     clf = LinearDiscriminantAnalysis()
#     clf.fit(XT_CSP, y)
#     saida_lda = clf.predict(XV_CSP)
    
#     # for aa in saida_lda: print(aa)
    
#     acuracia = mean(saida_lda == y)
    
#     tempo = time() - t0
    
#     print(sujeito, classes, round(tempo, 2), round(acuracia * 100, 2))
#     #print(round(acuracia * 100, 2))
#     # print asarray(XJ).shape 
    
#     return acuracia, tempo

# if __name__ == "__main__":
#     # sujeito = arange(1, 10)
#     classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
#     atrasos = range(0,6) 
#     t_filtro = ['IIR','FFT']  # opções: 'IIR' 'FFT' 'PROJ'
#     n_componentes = 6
#     fs = 250
#     fl = 8 
#     fh = 30
#     ordem = 5
    
#     print('\n>> Qtd sujeitos: 9')
#     print('>> Classes: LH x RH')
#     print('>> Abordagem: CSP + LDA')
#     print('>> Parâmetros: [8-30Hz] butter_order=5 ')
    
#     for tf in t_filtro:
#         print('\n========================',tf,'=========================')
#         print('Atrasos\t','| Acc Média (%)','| DP (%)', '| Custo (s)   ', '| Custo_i (s)')
#         print('======================================================')
#         for atraso in atrasos:
#             args = atraso, tf, n_componentes, fs, fl, fh, ordem
#             result = [sbcsp(suj, classes[i], args) for suj in range(1, 2) for i in range(0, 1) ]
#             print ('  '+ str(atraso) +'\t |  ' + 
#                    str(round(mean(asarray(result)[:,0])*100, 2)) + '%\t |  ' +
#                    str(round(std(asarray(result)[:,0])*100, 2)) + '% |  ' +
#                    str(round(sum(asarray(result)[:,1]), 2)) + 's\t| ' +
#                    str(round(mean(asarray(result)[:,1]), 2)) + 's')
            
#             #print ('Acc média:\t' + str(round(mean(asarray(result)[:,0])*100, 2)) + '%')
#             #print ('Custo total:\t' + str(round(sum(asarray(result)[:,1]), 2)) + 's')
#             #print ('Custo médio:\t' + str(round(mean(asarray(result)[:,1]), 2)) + 's\n')