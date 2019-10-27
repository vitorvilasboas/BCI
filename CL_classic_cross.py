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

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit

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
    # Classes = LH, RH, FooT
    # Nchannels = 16
    # Nsubjects = 1 (CL_)
    # Nsessions = 2 (S1, S2)
    #      S1 - lh-rh -> 5*20 = 100 trials (50 per class) 
	#      S2 - lh-ft -> 3*16 = 48 trials (24 per class) 
    # Timestamps Protocol: startTrial=0; Beep=1; Wait=2; Cue=3; startMI=3; endMI=9; endTrial=14
    # Samplestamps Protocol: startTrial=0; cue/startMI=375; endMI=1125; endTrial=1625
    
    dataset = ['S1','S2'] # S1[0]=LH S1[1]=RH ||| S2[0]=LH S2[1]=FT
    n_comp = 8
    fl = 8 
    fh = 30
    fs = 125
    nf = int(fs/2)
    ordem = 5
    
    t_start,t_end = 3,7
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    
    folds = 6
    sizeTest = 0.2
    
    path = './eeg_epochs/BCI_CAMTUC/CL_'
    
    ACC = []
    for ds in dataset:
        X = list(np.load(path + ds + '.npy'))
        
        X = np.concatenate([X[0],X[1]])
        
        X = X[:,:,W_Start:W_End] 
        b, a = butter(ordem, [fl/nf, fh/nf], btype='bandpass')
        X = lfilter(b, a, X)
        
        y = np.concatenate([np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))]) # target vector
        
        clf = LinearDiscriminantAnalysis()
        #clf = SVC(kernel="poly", C=10**(-4))
        #clf = KNeighborsClassifier(n_neighbors=24, metric='minkowski', p=2) #minkowski e p=2 -> para usar distancia euclidiana padrão
        #clf = DecisionTreeClassifier(criterion='entropy', random_state=0) #max_depth = None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
        #clf = GaussianNB()
        #clf = MLPClassifier(verbose=False, max_iter=10000, tol=0.0001, activation='logistic', learning_rate_init=0.001, learning_rate='invscaling',  solver='adam') #hidden_layer_sizes=(100,),
    
        ## Cross-validation: 
        process = Pipeline([('CSP', CSP(n_comp)), ('classifier', clf)]) # executa uma sequencia de processamento com um classificador no final
        cv = StratifiedShuffleSplit(folds, test_size=sizeTest, random_state=42)
        scores = cross_val_score(process, X, y, cv=cv)
        
        acc = np.mean(scores)
        print(round(acc*100,1))
        ACC.append([ds,round(acc * 100, 1)])
    
    FINAL = pd.DataFrame(ACC, columns=['Session','Mean Acc'])
        
