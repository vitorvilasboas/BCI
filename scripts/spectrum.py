# -*- coding: utf-8 -*-
import os
import math
import time
import pickle
import pyOpenBCI
import collections
import numpy as np
import pandas as pd
import pywt # pacote Py Wavelets
from scipy.io import loadmat
from scipy.linalg import eigh
from datetime import datetime
from scipy.signal import welch, butter, lfilter, filtfilt, iirfilter, stft, morlet, cwt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scripts.bci_utils import extractEpochs, nanCleaner, Filter

from scipy.fftpack import fft

# data, events, info = labeling(path='/mnt/dados/eeg_data/IV2a/', ds='IV2a', session='T', subj=1, channels=None, save=False)
data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A05T.npy', allow_pickle=True)
Fs = 250


######### EEG tempo real --------------------------------------------------------
# ## Filtro A
# # b, a = butter(5, [8/125, 30/125], btype='bandpass')
# b, a = iirfilter(5, [8/125, 30/125], btype='band')
# # X_ = lfilter(b, a, X) #filtfilt

# plt.ion()
# fig = plt.figure(figsize=(15, 4), facecolor='mintcream')

# gridspec.GridSpec(2,2)
# axes = fig.add_subplot(111)
# T, tx = 4, 0.1
# # t0, tn = 1/Fs, (1+T)/Fs
# t0, tn = tx, T
# for i in range(0, 15000, int(Fs*tx)):
#     # plt.figure(figsize=(15, 4), facecolor='mintcream')
#     # axes.ylim((-120, 120))
#     plt.subplot2grid((2,2), (0,0), colspan=2)
#     # plt.subplot(2, 1, 1)
#     y = data[13, i:i+(T*Fs)]*1e6 # data[13, i:i+1000]*1e6
#     # print(i)
#     x = np.linspace(t0, tn, T*Fs)
#     plt.gca().cla() # optionally clear axes # plt.clf(), plt.cla(), plt.close()
#     plt.plot(x,y)
#     plt.title('Sinal bruto')
#     plt.ylim((-60, 60))
#     plt.ylabel(r'$\mu$V')
#     plt.xlabel('Tempo (s)')
#     plt.draw()
#     # i += 0.1
    
#     # plt.subplot(2, 1, 2)
#     plt.subplot2grid((2,2), (1,0))
#     y_ = lfilter(b, a, y) #filtfilt
#     freq, plh = welch(y_, fs=Fs, nfft=(y_.shape[-1]-1))
#     plt.gca().cla() # optionally clear axes # plt.clf(), plt.cla(), plt.close()
#     plt.plot(freq, plh*1e13)
#     plt.subplots_adjust(hspace = .6)
#     plt.title('Welch')
#     plt.xlim((6, 32))
#     plt.ylim((0, 20*1e13))
#     plt.ylabel(r'$\mu$V')
#     plt.xlabel('Frequência (Hz)')
#     plt.draw()
    
#     t0 += (tx)
#     tn += (tx)
#     plt.pause(tx)

# fig.tight_layout()
# plt.show(block=True)



# # plt.ion()
# # fig = plt.figure(figsize=(10, 4), facecolor='mintcream')
# # axes = fig.add_subplot(111)
# # T, tx = 4, 0.5
# # # t0, tn = 1/Fs, (1+T)/Fs
# # t0, tn = tx, T
# # for i in range(0, 15000, int(Fs*tx)):
# #     y = data[13, i:i+(T*Fs)]*1e6 # data[13, i:i+1000]*1e6
# #     print(i)
# #     # x = np.linspace(t0, tn, T*Fs)
# #     y_ = lfilter(b, a, y) #filtfilt
# #     freq, plh = welch(y_, fs=Fs, nfft=(y_.shape[-1]-1))
# #     plt.gca().cla() # optionally clear axes # plt.clf(), plt.cla(), plt.close()
# #     plt.plot(freq, plh*1e13)
# #     plt.xlim((5, 35))
# #     plt.draw()
# #     # i += 0.1
# #     t0 += (tx)
# #     tn += (tx)
# #     plt.pause(tx)
# # plt.show(block=True)
#### -----------------------------------------------------------------------------


class_ids = [1,2]
smin = math.floor(0.5 * info['fs'])
smax = math.floor(2.5 * info['fs'])
buffer_len = smax - smin
epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
epochs = nanCleaner(epochs)
Fs = 250

X = [ epochs[np.where(labels == i)] for i in class_ids ]
X = np.vstack(X)


# Filtro A
b, a = butter(5, [8/125, 30/125], btype='bandpass')
# b, a = iirfilter(5, [8/125, 30/125], btype='band')
X_ = lfilter(b, a, X) #filtfilt



lh = X_[0]
rh = X_[100]

# Filtro B
# D = np.eye(22,22) - np.ones((22,22))/22
# lh = D @ lh
# rh = D @ rh


ch = 13 # 7,13 = hemisf esquerdo (atenua RH) 
# ch = 17 # 11,17 = hemisf direito (atenua LH)
lado = 'hemif. esquerdo' if ch in [7,13] else 'hemif. direito'

###########################################################################

# ## Welch
# # sinais de duas trials de classes diferentes com o mesmo canal localizado em um dos lados do cérebro
# freq, plh = welch(lh[ch], fs=info['fs'], nfft=(lh.shape[-1]-1))
# _   , prh = welch(rh[ch], fs=info['fs'], nfft=(rh.shape[-1]-1)) 

# plt.plot(freq, plh*1e13, label='LH') 
# plt.plot(freq, prh*1e13, label='RH')
# plt.xlim((0,40))
# # plt.ylim((-35, -25))
# plt.title(f'Welch C3 ({lado})')
# plt.ylabel(r'$\mu$V')
# plt.xlabel('Frequência (Hz)')
# plt.legend()

# ## FFT
# T = 1/info['fs']
# # freq = np.linspace(0.0, 1.0/(2.0*T), lh.shape[-1]//2)
# freq = np.fft.fftfreq(lh.shape[-1], T)
# mask = freq>0
# freq = freq[mask]

# flh = np.abs(np.fft.fft(lh[ch]))[mask]
# frh = np.abs(np.fft.fft(rh[ch]))[mask]

# # flh = (2.0 * np.abs( fft(lh[ch]) / lh.shape[-1]))[mask]
# # frh = (2.0 * np.abs( fft(rh[ch]) / rh.shape[-1]))[mask]

# plt.figure()
# plt.plot(freq, flh*1e5, label='LH')
# plt.plot(freq, frh*1e5, label='RH')
# plt.xlim((0,40))
# plt.title(f'FFT C3 ({lado})')
# plt.ylabel(r'$\mu$V')
# plt.xlabel('Frequência (Hz)')
# plt.legend()

# # freq, tempo, Zxx = stft(lh[ch], info['fs'], nperseg=lh.shape[-1])
# # plt.figure()
# # plt.pcolormesh(tempo, freq, np.abs(Zxx)) # , vmin=0, vmax=amp
# # plt.title('STFT Magnitude')
# # plt.ylabel('Frequency [Hz]')
# # plt.xlabel('Time [sec]')
# # plt.legend()

# # (cA, cD) = pywt.dwt(lh, 'db1')


# ###########################################################

X = [ epochs[np.where(labels == i)] for i in class_ids ]
Xa = X[0] # all epochs LH
Xb = X[1] # all epochs RH

plt.subplot(2, 1, 1)
plt.plot(Xa[0, ch])

D = np.eye(22,22) - np.ones((22,22))/22
Ra = np.asarray([D @ Xa[i] for i in range(len(Xa))])
Rb = np.asarray([D @ Xb[i] for i in range(len(Xb))])

# Ra = lfilter(b, a, Xa)
# Rb = lfilter(b, a, Xb)

xa = Ra[:,ch] # all epochs, 1 channel, all samples LH
xb = Rb[:,ch] # all epochs, 1 channel, all samples RH
# xa[np.isnan(xa)] = np.zeros(xa[np.isnan(xa)].shape)
# xb[np.isnan(xb)] = np.zeros(xb[np.isnan(xb)].shape)

# plt.psd(xa.T, 512, 1 / 0.04)
# plt.psd(xb.T, 512, 1 / 0.04)

## Welch
freq, pa = welch(xa, fs=250, nfft=(xa.shape[-1]-1)) # nfft=499 para q=500
_   , pb = welch(xb, fs=250, nfft=(xb.shape[-1]-1)) 
pa, pb = np.real(pa), np.real(pb)
ma, mb = np.mean(pa,0), np.mean(pb,0)

# plt.figure(figsize=(10, 5), facecolor='mintcream')
plt.subplots(figsize=(10, 12), facecolor='mintcream')
plt.subplot(2, 1, 2)
# plt.semilogy(freq, np.mean(p1,0), label='LH')
# plt.semilogy(freq, np.mean(p2,0), label='RH')
plt.plot(freq, ma*1e5, label='mão esquerda')  # np.log10(ma)
plt.plot(freq, mb*1e5, label='mão direita')
plt.xlim((0,40))
# plt.ylim((-14, -11.5))
plt.title(f'Welch C3 ({lado})')
plt.ylabel(r'$\mu$V')
plt.xlabel('Frequência (Hz)')
plt.legend()


# ## FFT
# T = 1/info['fs']
# # freq = np.linspace(0.0, 1.0/(2.0*T), xa.shape[-1]//2)
# freq = np.fft.fftfreq(xa.shape[-1], T)
# mask = freq>0
# freq = freq[mask]

# fa = np.abs(np.fft.fft(xa))[:, mask]
# fb = np.abs(np.fft.fft(xb))[:, mask]
# # fa = (2.0 * np.abs( fft(xa) / xa.shape[-1]))[:, mask]
# # fb = (2.0 * np.abs( fft(xb) / xb.shape[-1]))[:, mask]

# ma, mb = np.mean(fa,0), np.mean(fb,0)

# plt.subplot(2, 1, 2)
# plt.plot(freq, ma*1e5, label='LH')
# plt.plot(freq, mb*1e5, label='RH')
# plt.xlim((0,40))
# plt.title(f'FFT C3 ({lado})')
# plt.ylabel(r'$\mu$V')
# plt.xlabel('Frequência (Hz)')
# plt.legend()
# plt.savefig('/home/vboas/Desktop/psd_welch_fft_c3.png', format='png', dpi=300, transparent=True, bbox_inches='tight')



# =============================================================================
# ### Plot Raw x Filtered signals
# =============================================================================
# X = [ epochs[np.where(labels == i)] for i in class_ids ]
# X = np.vstack(X)
# b, a = butter(5, [8/125, 30/125], btype='bandpass')
# X_ = lfilter(b, a, X) #filtfilt

# gridspec.GridSpec(2,1)
# plt.subplots(figsize=(10,5))
# plt.subplot2grid((2,1), (0,0))
# # plt.subplot(2, 1, 1)
# plt.plot(np.linspace(3, 4, Fs), X[0, 13, :]*1e6, linewidth=.8, color='dodgerblue', label='sinal bruto')
# plt.ylabel(r'Amplitude ($\mu$V)', fontsize=12, labelpad=1)
# plt.xlabel(r'tempo (seg)', fontsize=12, labelpad=1)
# plt.grid(axis='both', **dict(ls='--', alpha=0.6))
# plt.ylim((-26,26))
# plt.xlim((3,4))
# plt.yticks((-20,0,20))
# plt.xticks(np.linspace(3,4,11))
# plt.legend(loc='upper right')

# plt.subplot2grid((2,1), (1,0))
# # plt.subplot(2, 1, 2)
# plt.plot(np.linspace(3, 4, Fs), X_[0, 13, :]*1e6, linewidth=.8, color='crimson', marker='.', markersize=3, label='sinal filtrado')
# plt.ylabel(r'Amplitude ($\mu$V)', fontsize=12, labelpad=1)
# plt.xlabel(r'tempo (seg)', fontsize=12, labelpad=1)
# plt.yticks((-20,0,20))
# plt.ylim((-26,26))
# plt.grid(axis='both', **dict(ls='--', alpha=0.6))
# plt.xlim((3,4))
# plt.xticks(np.linspace(3,4,11))
# plt.subplots_adjust(hspace=0.5, wspace=0)
# plt.legend(loc='upper right')
# plt.savefig('/home/vboas/Desktop/raw_x_filtered_signals.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


################## BKPs

# plt.ion()
# fig = plt.figure(figsize=(15, 4), facecolor='mintcream')
# # axes = fig.add_subplot(111)
# T, tx = 1, 0.1
# # t0, tn = 1/Fs, (1+T)/Fs
# t0, tn = tx, T
# for i in range(0, 15000, int(Fs*tx)):
#     # plt.figure(figsize=(15, 4), facecolor='mintcream')
#     plt.ylim((-120, 120))
#     y = data[13, i:i+(T*Fs)]*1e6 # data[13, i:i+1000]*1e6
#     print(i)
#     x = np.linspace(t0, tn, T*Fs)
#     plt.gca().cla() # optionally clear axes # plt.clf(), plt.cla(), plt.close()
#     plt.plot(x,y)
#     plt.draw()
#     # i += 0.1
#     t0 += tx
#     tn += tx
#     plt.pause(tx)
# plt.show(block=True)


# =============================================================================
# FIGURA RAW x FFT
# =============================================================================
# data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A05T.npy', allow_pickle=True)
# class_ids = [1,2]
# smin = math.floor(0.5 * info['fs'])
# smax = math.floor(2.5 * info['fs'])
# buffer_len = smax - smin
# epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
# epochs = nanCleaner(epochs)
# Fs = info['fs']

# X = [ epochs[np.where(labels == i)] for i in class_ids ]
# X = np.vstack(X)
# b, a = butter(5, [8/125, 30/125], btype='bandpass')
# X_ = lfilter(b, a, X) #filtfilt

# x = X[0, 11, :]
# x_= X_[0, 11, :]

# gridspec.GridSpec(2,1)
# plt.subplots(figsize=(11,6))
# plt.subplot2grid((3,1), (0,0))
# # plt.subplot(2, 1, 1)
# plt.plot(np.linspace(2.5, 4.5, 2*Fs), x*1e6, linewidth=1.5, color='dodgerblue', label=r'Sinal de EEG bruto ($\mathbf{Z}_p$)')
# plt.ylabel(r'Amplitude ($\mu$V)', fontsize=12, labelpad=1)
# plt.xlabel(r'tempo (seg)', fontsize=12, labelpad=1)
# plt.grid(axis='both', **dict(ls='--', alpha=0.5))
# plt.ylim((-26,26))
# plt.xlim((2.5,4.5))
# plt.yticks((-20,0,20))
# plt.xticks(np.linspace(2.5,4.5,21))
# plt.legend(loc='upper right')

# f = np.abs(np.fft.fft(x))*1e4

# plt.subplot2grid((3,1), (1,0))
# plt.plot(f, linewidth=1.2, color='crimson', label=r'Espectro do sinal bruto $\mathbf{Z}_p$')
# plt.ylabel(r'Amplitude', fontsize=12, labelpad=1)
# plt.xlabel(r'Índice de coeficientes DFT', fontsize=12, labelpad=1)
# plt.grid(axis='both', **dict(ls='--', alpha=0.5))
# plt.xlim((0,len(f)))
# plt.xticks(np.linspace(0,500,21))
# plt.subplots_adjust(hspace=0.4, wspace=0)
# plt.legend(loc='upper right')

# T = 1/info['fs']
# freq = np.fft.fftfreq(x.shape[-1], T)
# # ff = np.abs(np.fft.fft(x_))*1e4
# mask = freq>0
# freq = freq[mask]
# ff = np.abs(np.fft.fft(x_)[mask])*1e4

# plt.subplot2grid((3,1), (2,0))
# plt.plot(freq, ff, linewidth=1.2, color='g', label=r'Espectro do sinal filtrado $\mathbf{\tilde X}_p$')
# plt.ylabel(r'Amplitude', fontsize=12, labelpad=1)
# plt.xlabel('Frequência (Hz)', fontsize=12, labelpad=1)
# plt.grid(axis='both', **dict(ls='--', alpha=0.5))
# plt.xlim((0,Fs/2))
# plt.xticks(np.linspace(0,Fs/2-5,13))
# plt.subplots_adjust(hspace=0.4, wspace=0)
# plt.legend(loc='upper right')
# plt.savefig('/home/vboas/Desktop/raw_x_fft.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

# # =============================================================================
# # RAW x IIR x COEF FFT
# # =============================================================================
# data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A05T.npy', allow_pickle=True)
# class_ids = [1,2]
# smin = math.floor(0.5 * info['fs'])
# smax = math.floor(2.5 * info['fs'])
# buffer_len = smax - smin
# epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
# epochs = nanCleaner(epochs)
# Fs = info['fs']

# X = [ epochs[np.where(labels == i)] for i in class_ids ]
# X = np.vstack(X)
# b, a = butter(5, [8/125, 12/125], btype='bandpass')
# X_ = lfilter(b, a, X) #filtfilt

# x = X[0, 13, :]
# x_= X_[0, 13, :]

# gridspec.GridSpec(2,1)
# plt.subplots(figsize=(11,5))
# plt.subplot2grid((2,1), (0,0))
# # plt.subplot(2, 1, 1)
# # plt.plot(np.linspace(2.5, 4.5, 2*Fs), x*1e6, linewidth=2, color='dodgerblue', label='sinal bruto')
# # plt.ylabel(r'Amplitude ($\mu$V)', fontsize=12, labelpad=1)
# # plt.xlabel(r'tempo (seg)', fontsize=12, labelpad=1)
# # plt.grid(axis='both', **dict(ls='--', alpha=0.6))
# # plt.ylim((-26,26))
# # plt.xlim((2.5,4.5))
# # plt.yticks((-20,0,20))
# # plt.xticks(np.linspace(2.5,4.5,21))
# # plt.legend(loc='lower right')

# # plt.subplot2grid((3,1), (1,0))
# # plt.subplot(2, 1, 2)
# plt.plot(np.linspace(2.5, 4.5, 2*Fs), x_*1e6, linewidth=.8, color='g', marker='o', markerfacecolor='lavender', markersize=4, label=r'Sinal filtrado ($\mathbf{\widetilde X}_p$)')
# plt.ylabel(r'Amplitude', fontsize=12, labelpad=1)
# plt.xlabel(r'tempo (seg)', fontsize=12, labelpad=1)
# plt.yticks((-4,0,4))
# plt.ylim((-5,5))
# plt.grid(axis='both', **dict(ls='--', alpha=0.6))
# plt.xlim((2.5,4.5))
# plt.xticks(np.linspace(2.5,4.5,21))
# plt.subplots_adjust(hspace=0.5, wspace=0)
# plt.legend(loc='upper right', borderaxespad=0.2, prop={'size':9})

# T = 1/info['fs']
# # freq = np.linspace(0.0, 1.0/(2.0*T), xa.shape[-1]//2)
# freq = np.fft.fftfreq(x.shape[-1], T)
# f = np.fft.fft(x)

# mask = freq>0
# freq = freq[mask]
# freq = freq[15:32]
# f = np.fft.fft(x)[mask]
# f = f[15:32]

# real = np.real(f)*1e4
# imag = np.imag(f)*1e4

# # plt.figure(figsize=(10,6), facecolor='mintcream')
# # plt.subplot(2, 1, 2)
# plt.subplot2grid((2,1), (1,0))
# plt.scatter(freq, real, label=r'$\mathbf{\widehat X}_p$ (cos)', linewidth=1.2, marker='o', facecolor='lavender', alpha=.9, edgecolors='royalblue')
# plt.scatter(freq, imag, label=r'$\mathbf{\widehat X}_p$ (sen)', linewidth=1.2, marker='o', facecolor='lavender', alpha=.9, edgecolors='red')
# plt.plot(np.arange(0,35), np.zeros(35), color='k', linewidth=.9)
# for i in range(len(real)):
#     plt.plot(freq[i]*np.ones(10), np.linspace(0,real[i],10), color='royalblue', linewidth=.5)
#     plt.plot(freq[i]*np.ones(10), np.linspace(0,imag[i],10), color='r', linewidth=.5)
# plt.xlim((7.7,16.3))
# plt.ylim((-5,5))
# plt.grid(axis='both', **dict(ls='--', alpha=0.6))
# # plt.title(f'FFT C3 ({lado})')
# plt.ylabel('Amplitude', fontsize=12, labelpad=1)
# plt.xlabel('Frequência (Hz)', fontsize=12, labelpad=1)
# plt.legend(loc='upper right', labelspacing=0, borderaxespad=0.2, prop={'size':9})
# plt.xticks(np.linspace(8,16,9))
# plt.yticks((-4,0,4))
# plt.subplots_adjust(hspace=0.5, wspace=0)
# # plt.savefig('/home/vboas/Desktop/iir_x_fft.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

# # freq = np.fft.fftfreq(x.shape[-1], T)
# # # ff = np.abs(np.fft.fft(x_))*1e4
# # mask = freq>0
# # freq = freq[mask]
# # freq = freq[0:100]
# # ff = np.abs(np.fft.fft(x_)[mask])*1e4
# # ff = ff[0:100]

# # f = np.abs(np.fft.fft(x)[mask])*1e4
# # f = f[0:100]

# # plt.subplot2grid((3,1), (2,0))
# # plt.plot(freq, ff, linewidth=1.2, color='g', label=r'Espectro do sinal filtrado $\mathbf{\tilde X}_p$')
# # plt.plot(freq, f, linewidth=1.2, color='k', label=r'Espectro do sinal filtrado $\mathbf{\tilde X}_p$')
# # plt.ylabel(r'Amplitude', fontsize=12, labelpad=1)
# # plt.xlabel('Frequência (Hz)', fontsize=12, labelpad=1)
# # plt.grid(axis='both', **dict(ls='--', alpha=0.5))
# # plt.xlim((-1,51))
# # plt.xticks(np.linspace(0,50,11))
# # plt.subplots_adjust(hspace=0.4, wspace=0)
# # plt.legend(loc='upper right')
# # # plt.savefig('/home/vboas/Desktop/raw_x_fft.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


