# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import math
import pickle
import numpy as np
from time import time
from scipy.stats import norm
from sklearn.svm import SVC
from scipy.io import loadmat
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from bci_utils import extractEpochs, nanCleaner, Filter, CSP
from sklearn.metrics import cohen_kappa_score
from bci_utils import BCI


ds = 'III4a' # III3a, III4a, IV2a, IV2b, Lee19, CL, TW

## >>> ENTER THE PATH TO THE DATASET HERE
path = '/mnt/dados/eeg_data/' + ds + '/npy/' + ('S' if ds=='Lee19' else 'A0' if ds=='IV2a' else 'B0' if ds=='IV2b' else '') 
fs = 250 if ds=='Lee19' else 100 if ds=='III4a' else 125 if ds=='CL' else 250
subjects = range(1,55) if ds=='Lee19' else ['aa','al','av','aw','ay'] if ds=='III4a' else ['K3','K6','L1'] if ds=='III4a' else range(1,10)
class_ids = [1, 3] if ds=='III4a' else [1, 2]

if ds=='Lee19':
    cortex_only = True
    one_session = True
    lee_session = 1
    lee_option = ('_s' + str(lee_session) + '_cortex') if one_session and cortex_only else '_cortex' if cortex_only else ''

crossval = False
overlap = True
nfolds = 10
test_perc = 0.1 if crossval else 0.5

suj = 'aa'

f_low, f_high, ncomp, tmin, tmax = 0, 40, 4, 0.5, 2.5 
            
# clf = {'model':'Bayes'}
# clf = {'model':'LDA', 'lda_solver':'eigen'} # 'lda_solver': 'svd','lsqr','eigen'
# clf = {'model':'KNN', 'metric':'minkowski', 'neig':58} # 'metric': 'euclidean','manhattan','minkowski','chebyshev'
clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4} # 'kernel': 'linear', 'poly', 'sigmoid', 'rbf'
# clf = {'model':'MLP', 'eta':-3, 'activ':{'af':'relu'}, 'n_neurons':200, 'n_hidden':1, 'mlp_solver':'adam'} # 'alpha':-7,  'mlp_solver':'adam', 'lbfgs', 'sgd' # 'af':'identity', 'logistic', 'tanh', 'relu'
# clf = {'model':'DTree', 'crit':'entropy'} # 'crit': 'entropy' or 'gini'

# approach = {'option':'classic'}
approach = {'option':'sbcsp', 'nbands':9}

filtering = {'design':'DFT'}
# filtering = {'design':'IIR', 'iir_order':5}
# filtering = {'design':'FIR', 'fir_order':5}

################################################### 


if ds=='Lee19' and one_session: path_to_data = path + str(suj) + '_sess' + str(lee_session) + '.npy' # '.omi' 
else: path_to_data = path + str(suj) + '.npy'

data, events, info = np.load(path_to_data, allow_pickle=True) # pickle.load(open(path_to_data, 'rb'))

if ds=='Lee19' and cortex_only:
    cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
    data = data[cortex]   
    info['eeg_channels'] = len(cortex)
    info['ch_labels'] = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
# print(data.shape, events.shape)


#%% USE bci_utils PROCESSING
# bci = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, crossval=crossval, nfolds=nfolds, test_perc=test_perc,
#           f_low=f_low, f_high=f_high, tmin=tmin, tmax=tmax, ncomp=ncomp, ap=approach, filt_info=filtering, clf=clf)
# st = time()
# bci.evaluate()
# cost = time() - st
# print(suj, class_ids, str(round(bci.acc*100,2))+'%', str(round(bci.kappa,3)), str(round(cost, 2))+'s')
# if crossval: print(bci.cross_scores)


#%% USE LOCAL PROCESSING
smin = math.floor(tmin * fs)
smax = math.floor(tmax * fs)
buffer_len = smax - smin

dft_rf = fs/buffer_len # resolução em frequência fft
dft_size_band = 2/dft_rf # 2 representa sen e cos que foram separados do componente complexo da fft intercalados     

epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
epochs = nanCleaner(epochs)
# self.epochs = np.asarray([ nanCleaner(ep) for ep in self.epochs ])

# if ds=='Lee19' and one_session:
    # epochs, labels = epochs[:int(len(epochs)/2)], labels[:int(len(labels)/2)] # Lee19 somente sessão 1
    # epochs, labels = epochs[int(len(epochs)/2):], labels[int(len(labels)/2):] # Lee19 somente sessão 2

filt = Filter(f_low, f_high, fs, filtering)

csp = CSP(n_components=ncomp)

if clf['model'] == 'LDA': clf_final = LDA(solver=clf['lda_solver'], shrinkage=None)
if clf['model'] == 'Bayes': clf_final = GaussianNB()
if clf['model'] == 'SVM': clf_final = SVC(kernel=clf['kernel']['kf'], C=10 ** (clf['C']), gamma='scale', degree=3, probability=True)
if clf['model'] == 'KNN': clf_final = KNeighborsClassifier(n_neighbors=int(clf['neig']), metric=clf['metric'], p=3)           
if clf['model'] == 'DTree': clf_final = DecisionTreeClassifier(criterion=clf['crit'], random_state=0, max_depth=None, min_samples_split=2)  
if clf['model'] == 'MLP': clf_final = MLPClassifier(verbose=False, max_iter=10000, tol=1e-4, learning_rate_init=10**clf['eta'], # alpha=10**clf['alpha'], 
                                                    activation=clf['activ']['af'], hidden_layer_sizes=(int(clf['n_neurons']), int(clf['n_hidden'])),
                                                    learning_rate='constant', solver=clf['mlp_solver'])

if crossval:
    
    cross_scores = []
    cross_kappa = []
    
    kf = StratifiedShuffleSplit(nfolds, test_size=test_perc, random_state=42)
    # kf = StratifiedKFold(nfolds, False)
    
    st = time()
    
    if approach['option'] == 'classic':
        
        XF = filt.apply_filter(epochs)
        
        if filtering['design'] == 'DFT': # extrai somente os bins referentes à banda de interesse
            bmin = f_low * dft_size_band
            bmax = f_high * dft_size_band
            XF = XF[:, :, bmin:bmax]
            # print(bmin, bmax)
        
        # chain = Pipeline([('CSP', csp), ('SVC', clf_final)])
        # cross_scores = cross_val_score(chain, XF, labels, cv=kf)
        
        for idx_treino, idx_teste in kf.split(XF, labels):
            XT, XV, yT, yV = XF[idx_treino], XF[idx_teste], labels[idx_treino], labels[idx_teste]
            
            # Option 1
            csp.fit(XT, yT)
            XT_CSP = csp.transform(XT)
            XV_CSP = csp.transform(XV) 
            clf_final.fit(XT_CSP, yT)
            scores = clf_final.predict(XV_CSP)
            csp_filters = csp.filters_
            
            # Option 2
            # chain = Pipeline([('CSP', csp), ('SVC', clf_final)])
            # chain.fit(XT, yT)
            # scores = chain.predict(XV)
            # csp_filters = chain['CSP'].filters_
            
            acc_fold = np.mean(scores == yV) # or cross_scores.append(chain.score(XV, yV)) 
            kappa_fold = cohen_kappa_score(scores, yV)     
            cross_scores.append(acc_fold)
            cross_kappa.append(kappa_fold)
    
    
    
    elif approach['option'] == 'sbcsp':
        
        for idx_treino, idx_teste in kf.split(epochs, labels):
            
            XT, XV, yT, yV = epochs[idx_treino], epochs[idx_teste], labels[idx_treino], labels[idx_teste]
            
            
            nbands = int(approach['nbands'])
            # if nbands > (f_high-f_low): nbands = (f_high-f_low)
            
            n_bins = f_high - f_low
            overlap = 0.5 if overlap else 1
            step = n_bins / (nbands+1)
            size = step / overlap
            
            sub_bands = []
            for i in range(nbands):
                fl_sb = i * step + f_low
                fh_sb = i * step + size + f_low
                # if fh_sb <= self.f_high: sub_bands.append([fl_sb, fh_sb]) # extrapola limite superior 1: descarta última sub-banda 
                # if fh_sb > self.f_high: fh_sb = self.f_high # extrapola limite superior 2: ajusta f_high ao limite
                sub_bands.append([fl_sb, fh_sb])
            
            nbands = len(sub_bands)
            
            XTF, XVF = [], []
            if filtering['design'] == 'DFT':
                XT_FFT = filt.apply_filter(XT)
                XV_FFT = filt.apply_filter(XV)
                for i in range(nbands):
                    bmin = round(sub_bands[i][0] * dft_size_band)
                    bmax = round(sub_bands[i][1] * dft_size_band)
                    XTF.append(XT_FFT[:, :, bmin:bmax])
                    XVF.append(XV_FFT[:, :, bmin:bmax])
                    
            elif filtering['design'] in ['IIR' or 'FIR']:
                for i in range(nbands):
                    filt_sb = Filter(sub_bands[i][0], sub_bands[i][1], fs, filtering)
                    XTF.append(filt_sb.apply_filter(XT))
                    XVF.append(filt_sb.apply_filter(XV))
            
            chain = [ Pipeline([('CSP', CSP(n_components=ncomp)), ('LDA', LDA())]) for i in range(nbands) ]
            
            for i in range(nbands): chain[i]['CSP'].fit(XTF[i], yT)
                
            XT_CSP = [ chain[i]['CSP'].transform(XTF[i]) for i in range(nbands) ]
            XV_CSP = [ chain[i]['CSP'].transform(XVF[i]) for i in range(nbands) ]
            
            SCORE_T = np.zeros((len(XT), nbands))
            SCORE_V = np.zeros((len(XV), nbands))
            for i in range(len(sub_bands)): 
                chain[i]['LDA'].fit(XT_CSP[i], yT)
                SCORE_T[:, i] = np.ravel(chain[i]['LDA'].transform(XT_CSP[i]))  # classificações de cada época nas N sub bandas - auto validação
                SCORE_V[:, i] = np.ravel(chain[i]['LDA'].transform(XV_CSP[i]))
            
            csp_filters_sblist = [chain[i]['CSP'].filters_ for i in range(nbands)]
            lda_sblist = [chain[i]['LDA'] for i in range(nbands)] 
            
            SCORE_T0 = SCORE_T[yT == class_ids[0], :]
            SCORE_T1 = SCORE_T[yT == class_ids[1], :]
            p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
            p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
            META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
            META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
    
            clf_final.fit(META_SCORE_T, yT)
            scores = clf_final.predict(META_SCORE_V)
                                    
            acc_fold = np.mean(scores == yV) # calculo acurácia
            kappa_fold = cohen_kappa_score(scores, yV)
                   
            cross_scores.append(acc_fold)
            cross_kappa.append( kappa_fold )
    
    acc = np.mean(cross_scores)  # acurácia média (todas nfolds iterações)
    kappa = np.mean(cross_kappa) 
    
    cost = time() - st
    
else: # not crossval
    
    test_size = int(len(epochs) * test_perc)
    train_size = int(len(epochs) - test_size)
    train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
    epochsT, labelsT = epochs[:train_size], labels[:train_size] 
    epochsV, labelsV = epochs[train_size:], labels[train_size:]
    
    XT = [ epochsT[np.where(labelsT == i)] for i in class_ids ] # Extrair épocas de cada classe
    XV = [ epochsV[np.where(labelsV == i)] for i in class_ids ]
    
    XT = np.concatenate([XT[0],XT[1]]) # Train data classes A + B
    XV = np.concatenate([XV[0],XV[1]]) # Test data classes A + B        
    yT = np.concatenate([class_ids[0] * np.ones(int(len(XT)/2)), class_ids[1] * np.ones(int(len(XT)/2))])
    yV = np.concatenate([class_ids[0] * np.ones(int(len(XV)/2)), class_ids[1] * np.ones(int(len(XV)/2))])
    
    # print(XT.shape, XV.shape, yT.shape, yV.shape)
    
    st = time()
    
    if approach['option'] == 'classic':
        
        XTF = filt.apply_filter(XT)
        XVF = filt.apply_filter(XV)
        
        if filtering['design'] == 'DFT': # extrai somente os bins referentes à banda de interesse
            bmin = f_low * dft_size_band
            bmax = f_high * dft_size_band
            XTF = XTF[:, :, bmin:bmax]
            XVF = XVF[:, :, bmin:bmax]
            # print(bmin, bmax)
        
        # Option 1

        csp.fit(XTF, yT)
        XT_CSP = csp.transform(XTF)
        XV_CSP = csp.transform(XVF) 
        clf_final.fit(XT_CSP, yT)
        scores = clf_final.predict(XV_CSP)
        csp_filters = csp.filters_
        
        # Option 2
        # chain = Pipeline([('CSP', csp), ('SVC', clf_final)])
        # chain.fit(XTF, yT)
        # scores = chain.predict(XVF)
        # csp_filters = chain['CSP'].filters_
        
        acc = np.mean(scores == yV) # or chain.score(XVF, yV)     
        kappa = cohen_kappa_score(scores, yV)
        
        
    elif approach['option'] == 'sbcsp':
        
        nbands = int(approach['nbands'])
        # if nbands > (f_high-f_low): nbands = (f_high-f_low)
        
        n_bins = f_high - f_low
        overlap = 0.5 if overlap else 1
        step = n_bins / (nbands+1)
        size = step / overlap
        
        sub_bands = []
        for i in range(nbands):
            fl_sb = i * step + f_low
            fh_sb = i * step + size + f_low
            # if fh_sb <= self.f_high: sub_bands.append([fl_sb, fh_sb]) # extrapola limite superior 1: descarta última sub-banda 
            # if fh_sb > self.f_high: fh_sb = self.f_high # extrapola limite superior 2: ajusta f_high ao limite
            sub_bands.append([fl_sb, fh_sb])
        
        # print(sub_bands)
        nbands = len(sub_bands)
        
        XTF, XVF = [], []
        if filtering['design'] == 'DFT':
            XT_FFT = filt.apply_filter(XT)
            XV_FFT = filt.apply_filter(XV)
            for i in range(nbands):
                bmin = round(sub_bands[i][0] * dft_size_band)
                bmax = round(sub_bands[i][1] * dft_size_band)
                XTF.append(XT_FFT[:, :, bmin:bmax])
                XVF.append(XV_FFT[:, :, bmin:bmax])

        elif filtering['design'] in ['IIR' or 'FIR']:
            for i in range(nbands):
                filt_sb = Filter(sub_bands[i][0], sub_bands[i][1], fs, filtering)
                XTF.append(filt_sb.apply_filter(XT))
                XVF.append(filt_sb.apply_filter(XV))
        
        # print(np.mean(XTF), np.mean(XVF))
        
        chain = [ Pipeline([('CSP', CSP(n_components=ncomp)), ('LDA', LDA())]) for i in range(nbands) ]
        
        for i in range(nbands): chain[i]['CSP'].fit(XTF[i], yT)
            
        XT_CSP = [ chain[i]['CSP'].transform(XTF[i]) for i in range(nbands) ]
        XV_CSP = [ chain[i]['CSP'].transform(XVF[i]) for i in range(nbands) ]
        
        # print(np.mean(XT_CSP), np.mean(XV_CSP))
        
        SCORE_T = np.zeros((len(XT), nbands))
        SCORE_V = np.zeros((len(XV), nbands))
        for i in range(len(sub_bands)): 
            chain[i]['LDA'].fit(XT_CSP[i], yT)
            SCORE_T[:, i] = np.ravel(chain[i]['LDA'].transform(XT_CSP[i]))  # classificações de cada época nas N sub bandas - auto validação
            SCORE_V[:, i] = np.ravel(chain[i]['LDA'].transform(XV_CSP[i]))
        
        csp_filters_sblist = [chain[i]['CSP'].filters_ for i in range(nbands)]
        lda_sblist = [chain[i]['LDA'] for i in range(nbands)] 
        
        SCORE_T0 = SCORE_T[yT == class_ids[0], :]
        SCORE_T1 = SCORE_T[yT == class_ids[1], :]
        p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
        p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
        META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
        META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))

        clf_final.fit(META_SCORE_T, yT)
        scores = clf_final.predict(META_SCORE_V)
        
        acc = np.mean(scores == yV)
        kappa = cohen_kappa_score(scores, yV)

    cost = time() - st

print(suj, class_ids, str(round(acc*100,2))+'%', str(round(kappa,3)), str(round(cost, 2))+'s')
if crossval: print(cross_scores)