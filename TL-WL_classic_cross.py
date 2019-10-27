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
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, train_test_split, StratifiedKFold

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
    #				  CP3   CPz  CP4 	      5   2   8
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
    
    t_start,t_end = 3.5,6.5
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    
    folds = 6
    sizeTest = 0.2
    
    path = './eeg_epochs/BCI_CAMTUC/'
    
    ACC = []
    for suj in subjects:
        XS1 = list(np.load(path + suj + dataset[0] + '.npy')) # XS1[0]=LH, XS1[1]=RH (2*20 epochs)
        XS2 = list(np.load(path + suj + dataset[1] + '.npy')) # XS2[0]=LH, XS2[1]=RH (2*20 epochs)
        
        Xa = np.concatenate([XS1[0],XS2[0]]) # Xa = LH
        Xb = np.concatenate([XS1[1],XS2[1]]) # Xb = RH
        X = np.concatenate([Xa,Xb]) # X[0]=LH, X[1]=RH (2*40 epochs) 
        
        X = X[:,:,W_Start:W_End]
        b, a = butter(ordem, [fl/nf, fh/nf], btype='bandpass')
        X = lfilter(b, a, X)
        
        y = np.concatenate([np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))]) # target vector
        
        clf = LinearDiscriminantAnalysis()
        # clf = SVC(kernel="poly", C=10**(-4))
        # clf = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) 
        # clf = DecisionTreeClassifier(criterion='entropy', random_state=0) 
        # clf = GaussianNB()
        # clf = MLPClassifier(verbose=False, max_iter=10000, tol=0.0001, activation='logistic', learning_rate_init=0.001, learning_rate='invscaling',  solver='adam')
    
        # Cross-validation: Option 1 (compact code)
        process = Pipeline([('CSP', CSP(n_comp)), ('classifier', clf)]) # run a process sequence csp + clf
        cv = StratifiedShuffleSplit(folds, test_size=sizeTest, random_state=1) #random_state keep fixed the train and test indexes
        scores = cross_val_score(process, X, y, cv=cv)
        acc1 = np.mean(scores)
        
        ## Cross-validation: Option 2
        acc2 = []
        kf = StratifiedKFold(folds, False, 0)
        for train_index, test_index in kf.split(X, y):
            XT = X[train_index]
            XV = X[test_index]
            # TRAIN
            csp = CSP(n_components=n_comp)
            csp.fit(XT, y[train_index])
            XT_CSP = csp.transform(XT)
            clf.fit(XT_CSP, y[train_index])
            # TEST
            XV_CSP = csp.transform(XV)
            scores = clf.predict(XV_CSP)
            acc_fold = np.mean(scores == y[test_index])
            acc2.append(acc_fold)        
        
        ACC.append([suj,round(acc1 * 100, 1),round(np.mean(acc2) * 100, 1)])
        
        print('CV 1:', suj, round(acc1*100,1))
        print('CV 2:', suj, round(np.mean(acc2)*100,1))
    
    FINAL = pd.DataFrame(ACC, columns=['Suj','Acc cv1','Acc cv2'])
    
    
