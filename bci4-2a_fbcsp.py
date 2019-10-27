# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import lfilter, butter
from scipy.stats import norm
from sklearn.svm import SVC

class CSP():
    def __init__(self, n_components):
        self.n_components = n_components
        # self.filters_ = None
    def fit(self, X, y):
        e, c, s = X.shape
        classes = np.unique(y)   
        Xa = X[classes[0] == y,:,:]
        Xb = X[classes[1] == y,:,:]
        S0 = np.zeros((c, c)) 
        S1 = np.zeros((c, c))
        for epoca in range(int(e/2)):
            S0 += np.dot(Xa[epoca,:,:], Xa[epoca,:,:].T) #covA Xa[epoca]
            S1 += np.dot(Xb[epoca,:,:], Xb[epoca,:,:].T) #covB Xb[epoca]
        [D, W] = eigh(S0, S0 + S1)
        ind = np.empty(c, dtype=int)
        ind[0::2] = np.arange(c - 1, c // 2 - 1, -1) 
        ind[1::2] = np.arange(0, c // 2)
        W = W[:, ind]
        self.filters_ = W.T[:self.n_components]
        return self # instruction add because cross-validation pipeline
    def transform(self, X):        
        XT = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        XVAR = np.log(np.mean(XT ** 2, axis=2)) # Xcsp
        return XVAR
    
if __name__ == "__main__":
    
    sujeitos = range(1,10)
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    n_comp = 6 
    fs = 250
    Clog = -4
    
    fl = 8
    fh = 30
    ordem = 5
    nf = fs/2.
    n_bandas = 12
    
    t_start,t_end = 2.5,4.5
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    
    #ACC = np.zeros([len(sujeitos),len(classes)]) 
    ACC = []
    for cls,cl in zip(classes,range(len(classes))):
        #print("\n")
        for suj in sujeitos:
            XT = np.load('./eeg_epochs/BCI4_2a/A0' + str(suj) + 'T.npy')
            XV = np.load('./eeg_epochs/BCI4_2a/A0' + str(suj) + 'E.npy')
            
            XT = ([XT[cls[0]-1], XT[cls[1]-1]])
            XV = ([XV[cls[0]-1], XV[cls[1]-1]])
            
            # Windowing
            XTJ = [ XT[i][:,:,W_Start:W_End] for i in range(len(XT)) ]
            XVJ = [ XV[i][:,:,W_Start:W_End] for i in range(len(XV)) ]
            
            # Concatenating training and evaluating data
            XTC = np.concatenate([XTJ[0],XTJ[1]]) # Dados de treinamento das classes A, B [ ne*2 x nc x n_bins ]
            XVC = np.concatenate([XVJ[0],XVJ[1]]) # Dados de validação das classes A, B 
            y = np.concatenate([np.zeros(int(len(XTC)/2)), np.ones(int(len(XTC)/2))]) # Criando vetor gabarito
            
            # Divide sub-bands
            n_bins = fh-fl
            overlap = 2
            step = int(n_bins / n_bandas)
            size = int(step * overlap) # tamanho fixo p/ todas sub bandas. overlap em 50%
            
            XTF = []
            XVF = [] 
            for i in range(n_bandas):
                freq_low = i*step+fl
                freq_high = i*step+size+fl
                if freq_low == 0: freq_low = 0.001
                if freq_high > fh: freq_high = fh
        
                b, a = butter(ordem, [freq_low/nf, freq_high/nf], btype='bandpass') # to filt IIR
                XTF.append( lfilter(b, a, XTC) ) # Temporal/Spectral filtering
                XVF.append( lfilter(b, a, XVC) ) # o filtro é aplicado por padrão na última dimensão
            
            csp = [CSP(n_components=n_comp) for i in range(n_bandas)]
            for i in range(n_bandas): csp[i].fit(XTF[i], y)
            XT_CSP = [csp[i].transform(XTF[i]) for i in range(n_bandas)]
            XV_CSP = [csp[i].transform(XVF[i]) for i in range(n_bandas)]
        
            # LDA
            SCORE_T = np.zeros((144, n_bandas))
            SCORE_V = np.zeros((144, n_bandas))
            clf = [LinearDiscriminantAnalysis() for i in range(n_bandas)]
            for i in range(n_bandas):
                clf[i].fit(XT_CSP[i], y)
                SCORE_T[:, i] = np.ravel(clf[i].transform(XT_CSP[i])) # classificaçoes de cada época nas N sub bandas - auto validação
                SCORE_V[:, i] = np.ravel(clf[i].transform(XV_CSP[i])) # validação
            
            # Meta-classificador Bayesiano
            SCORE_T0 = SCORE_T[y == 0, :]
            m0 = np.mean(SCORE_T0, axis=0) #media classe A
            std0 = np.std(SCORE_T0, axis=0) #desvio padrão classe A
            
            SCORE_T1 = SCORE_T[y == 1, :]
            m1 = np.mean(SCORE_T1, axis=0)
            std1 = np.std(SCORE_T1, axis=0)
        
            p0 = norm(m0, std0) # p0 e p1 representam uma distribuição normal de médias m0 e m1, e desvio padrão std0 e std1
            p1 = norm(m1, std1)
            
            META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
            META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
            
            # SVM on top of the meta-classifier
            svc = SVC(kernel="poly", C=10**Clog)
            svc.fit(META_SCORE_T, y)
            saidas_svm = svc.predict(META_SCORE_V)
            
            acc = np.mean(saidas_svm == y) # Results
            print (suj, cls, str(round(acc*100, 1))+'%')
            
            #ACC[suj-1,cl] = acc
            ACC.append([suj,cls,round(acc * 100, 1)])
       
        FINAL = pd.DataFrame(ACC, columns=['Suj','Classes','Acc'])
    
    