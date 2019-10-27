# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import lfilter, butter

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

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
            S0 += np.dot(Xa[epoca,:,:], Xa[epoca,:,:].T) #covA Xa[epoca] # Sum up cov all epochs A
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
    # Classes = LH, RH
    # Fs = 250Hz
    # Nchannels = 8 (1=Cz 2=Cpz 3=C1 4=C3 5=CP3 6=C2 7=C4 8=CP4)
    #                               Scalp map
    #				C3  C1  Cz  C2  CP4 	    4  3  1  6  7 
    #				  CP3   CPz  CP4 		  5   2   8
    # Nsubjects = 2 (TL_,WL_)
    # Nsessions = 2 (S1, S2) -> 2*20 = 40 trials (20 per class) -> each session 
    # Timestamps Protocol: startTrial=0; waitBeep=2; startCue=3; startMI=4.25; endMI=8; endTtrial=10-12
    # Samplestamps Protocol: startTrial=0; cue=750; startMI=1063; endMI=2000; endTrial=2500

    dataset = ['S1','S2'] # S1[0]=LH S1[1]=RH ||| S2[0]=LH S2[1]=RH
    subjects = ['TL_','WL_']
    n_comp = 6
    fl = 8 
    fh = 30
    fs = 250
    nf = fs/2
    ordem = 5
    
    t_start,t_end = 3,8
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    
    path = './eeg_epochs/BCI_CAMTUC/'
    
    ACC = []
    for suj in subjects:
        XT = list(np.load(path + suj + dataset[0] + '.npy')) # XT[0]=LH, XV[1]=RH (2*20 epochs)
        XV = list(np.load(path + suj + dataset[1] + '.npy')) # XT[0]=LH, XV[1]=RH (2*20 epochs)
        
        XTJ = [ XT[i][:,:,W_Start:W_End] for i in range(len(XT)) ]
        XVJ = [ XV[i][:,:,W_Start:W_End] for i in range(len(XV)) ]
        
        b, a = butter(ordem, [fl/nf, fh/nf], btype='bandpass')
        XTF = [ lfilter(b, a, XTJ[i]) for i in range(len(XTJ)) ]
        XVF = [ lfilter(b, a, XVJ[i]) for i in range(len(XVJ)) ]
        
        XTF = np.concatenate([XTF[0],XTF[1]]) # Classes A and B - Training data
        XVF = np.concatenate([XVF[0],XVF[1]]) # Classes A and B - Evaluate data
        y = np.concatenate([np.zeros(int(len(XTF)/2)), np.ones(int(len(XTF)/2))]) # target vector
        
        # TRAIN
        csp = CSP(n_components=n_comp)
        csp.fit(XTF, y)
        XT_CSP = csp.transform(XTF)
        
        clf = LinearDiscriminantAnalysis()    
        #clf = SVC(kernel="poly", C=10**(-4))
        #clf = KNeighborsClassifier(n_neighbors=24, metric='minkowski', p=2) #minkowski e p=2 -> para usar distancia euclidiana padrão
        #clf = DecisionTreeClassifier(criterion='entropy', random_state=0) #max_depth = None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
        #clf = GaussianNB()
        #clf = MLPClassifier(verbose=False, max_iter=10000, tol=0.0001, activation='logistic', learning_rate_init=0.001, learning_rate='invscaling',  solver='adam') #hidden_layer_sizes=(100,),
        
        clf.fit(XT_CSP, y)
        
        # TEST
        XV_CSP = csp.transform(XVF)
        scores = clf.predict(XV_CSP)
        
        acc = np.mean(scores == y)
        print(round(acc * 100, 1))
        
        ACC.append([suj,round(acc * 100, 1)])
    
    FINAL = pd.DataFrame(ACC, columns=['Suj','Acc'])
