import math
import pickle
import itertools
import numpy as np
import scipy.signal as sp
import scipy.linalg as lg
import scipy.stats as sst
from sklearn.svm import SVC
from scipy.fftpack import fft
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from mne import Epochs, pick_types, find_events
# from mne.decoding import CSP #Import Common Spatial Patterns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report, make_scorer, accuracy_score, precision_recall_fscore_support, \
    confusion_matrix, cohen_kappa_score
# from processing.processor import Learner, Filter
from processing.utils import PATH_TO_SESSION, load_npy_data, readEvents, extractEpochs, nanCleaner, load_pickle_data

class Approach:
    def __init__(self, session=None):
        self.session = session

    def define_approach(self, sb_approach, sample_rate, f_low, f_high, csp_nei, class_ids, epoch_start, epoch_end,
                        filt_type, clf_sb, clf_final, order=None, nbands=None, overlap=True, crossval=True, nfolds=10,
                        test_perc=0.2):
        self.class_ids = class_ids
        self.sample_rate = sample_rate
        self.f_low = f_low
        self.f_high = f_high
        self.smin = int(math.floor(epoch_start * sample_rate))
        self.smax = int(math.floor(epoch_end * sample_rate))
        self.buffer_len = self.smax - self.smin
        self.crossval = crossval
        self.nfolds = nfolds
        self.test_perc = test_perc
        self.ncsp = csp_nei

        self.dft_rf = self.sample_rate / self.buffer_len  # resolução em frequência fft
        self.dft_size_band = round(2 / self.dft_rf)  # 2 representa sen e cos que foram separados do componente complexo da fft intercalados

        self.learner = Learner()
        self.learner.design_CLF(clf_final)
        self.learner.design_CSP(self.ncsp)

        self.sb_approach = sb_approach
        if self.sb_approach:
            if nbands > (self.f_high - self.f_low): nbands = (self.f_high - self.f_low)
            self.nbands = nbands
            self.overlap = overlap
            self.learner.design_CLF(clf_sb, sb_level=True)

        self.filt_type = filt_type
        if self.filt_type in ['IIR','FIR']: self.order = order

        self.filter = Filter(f_low, f_high, self.buffer_len, sample_rate, forder=order, filt_type=filt_type)
        # self.learner.assemble_learner()

    def set_channels(self, channels):
        self.channels = channels
        # if self.channels == list([-1]): self.channels = list(np.arange(0, self.session.dp.data_max_channels))

    def set_eeg_path(self, eeg_path):
        self.eeg_path = eeg_path
        self.data, self.events, self.info = self.load_eeg_data(self.eeg_path)

    def load_eeg_data(self, eeg_path, data_format='path'):
        if data_format == 'path':
            data, events, info = np.load(eeg_path, allow_pickle=True) # load_pickle_data(eeg_path)
            data = data[self.channels]
            events = events.astype(int)
            # if self.channels == [-1]: data = data[:self.session.dp.max_channels]
            # else: data = data[self.channels]
        elif data_format == 'npy':
            data, events, info = eeg_path
        return data, events, info

    def validate_model(self):
        # data, events = self.load_data(self.data_cal_path, self.events_cal_path)
        data, events, info = self.data, self.events, self.info # self.load_eeg_data(self.eeg_path)
        epochs, labels = extractEpochs(data, events, self.smin, self.smax, self.class_ids)
        epochs = nanCleaner(epochs)
        self.learner.evaluate(self, epochs, labels)
        score = self.learner.get_results()
        return score

    def classify_set(self, epochs, labels):
        self.learner.evaluate_set(epochs, labels)
        score = self.learner.get_results()
        return score

    def classify_epoch(self, epoca, out_param='label'):
        epoch_f = nanCleaner(epoca)
        # epoch_f = self.preprocess(epoch)
        if not epoca == []:
            guess = self.learner.evaluate_epoch(self, epoch_f, out_param=out_param)
        else: guess = None
        return guess

    def set_cal_path(self, dpath):
        self.data_path = dpath

    def load_epochs(self, data, events):
        epochs, labels = extractEpochs(data, events, self.smin, self.smax, self.class_ids)
        return epochs, labels

    def preprocess(self, data_in):
        data = nanCleaner(data_in)
        data_out = self.filter.apply_filter(data)
        return data_out

    def saveSetup(self, path):
        self.data = self.events = self.info = None
        pickle.dump(self.__dict__, open(path + '/setup_info.pkl', 'wb'))

    def loadSetup(self, path):
        self.__dict__.update(pickle.load(open(path + '/setup_info.pkl', 'rb')))

    # def set_cal_path_old(self, dpath, evpath):
    #     self.data_cal_path = dpath
    #     self.events_cal_path = evpath

    # def set_val_path(self, dpath, evpath):
    #     self.data_val_path = dpath
    #     self.events_val_path = evpath
    #     self.data, self.events = self.load_data(self.data_val_path, self.events_val_path) ## add
    #     self.epochs, self.labels = extractEpochs(self.data, self.events, self.smin, self.smax, self.class_ids)
    #     self.epochs = nanCleaner(self.epochs)

    # def load_data(self, dpath, evpath, data_format='path'):
    #     if data_format == 'path':
    #         # if self.channels == [-1]:
    #         #     data = load_npy_data(dpath)
    #         #     max_channels = data.shape[0]
    #         #     data = data[:self.session.acq.ds_max_channels]
    #         # else:
    #         #     data = load_npy_data(dpath)[self.channels]
    #         data = load_npy_data(dpath)[self.channels]
    #         events = readEvents(evpath)
    #     elif data_format == 'npy':
    #         data, events = dpath, evpath
    #     return data, events


class CSP():
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None

    def get_params(self, deep=True):
        return {"n_components": self.n_components}

    def fit(self, X, y):
        e, c, t = X.shape
        classes = np.unique(y)
        X0 = X[classes[0] == y, :, :]
        X1 = X[classes[1] == y, :, :]
        S0 = np.zeros((c, c))  # Sum up covariance matrix
        S1 = np.zeros((c, c))
        for i in range(int(e / 2)):
            S0 += np.dot(X0[i, :, :], X0[i, :, :].T)  # covA X0[epoca]
            S1 += np.dot(X1[i, :, :], X1[i, :, :].T)  # covB X1[epoca]
        [D, W] = lg.eigh(S0, S0 + S1)
        ind = np.empty(c, dtype=int)
        ind[0::2] = np.arange(c - 1, c // 2 - 1, -1)
        ind[1::2] = np.arange(0, c // 2)
        W = W[:, ind]
        self.filters_ = W.T[:self.n_components]
        return self  # used on cross-validation pipeline

    def transform(self, X):
        XT = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        XVAR = np.log(np.mean(XT ** 2, axis=2))
        return XVAR


class Filter:
    def __init__(self, fl, fh, buffer_len, srate, forder=None, filt_type='IIR', band_type='bandpass'):
        self.ftype = filt_type
        self.nyq = 0.5 * srate
        self.res_freq = (srate / buffer_len)
        # if fl == 0: fl = 0.001
        low = fl / self.nyq
        high = fh / self.nyq
        if low == 0: low = 0.001
        if high >= 1: high = 0.99

        if self.ftype == 'IIR':
            # self.b, self.a = sp.iirfilter(forder, [low, high], btype='band')
            self.b, self.a = sp.butter(forder, [low, high], btype=band_type)
        elif self.ftype == 'FIR':
            self.b = sp.firwin(forder, [low, high], window='hamming', pass_zero=False)
            self.a = [1]
        elif self.ftype == 'DFT':
            self.bmin = int(fl / self.res_freq)  # int(fl * (srate/self.nyq)) # int(low * srate)
            self.bmax = int(fh / self.res_freq)  # int(fh * (srate/self.nyq)) # int(high * srate)

    def apply_filter(self, data_in, is_epoch=False):
        if self.ftype != 'DFT':
            # data_out = sp.filtfilt(self.b, self.a, data_in)
            data_out = sp.lfilter(self.b, self.a, data_in)
        else:
            if is_epoch:
                data_out = fft(data_in)
                REAL = np.real(data_out).T #[:, self.bmin:self.bmax].T
                IMAG = np.imag(data_out).T #[:, self.bmin:self.bmax].T
                data_out = np.transpose(list(itertools.chain.from_iterable(zip(IMAG, REAL))))
            else:
                data_out = fft(data_in)
                REAL = np.transpose(np.real(data_out), (2, 0, 1))
                IMAG = np.transpose(np.imag(data_out), (2, 0, 1))
                # old: np.transpose(np.real(data_out)[:, :, self.bmin:self.bmax], (2, 0, 1))
                # old: np.transpose(np.imag(data_out)[:, :, self.bmin:self.bmax], (2, 0, 1))
                data_out = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
                data_out = np.transpose(data_out, (1, 2, 0))
        return data_out


class Learner:
    def __init__(self, model=None):
        # Loads a previous model if existent
        self.clf = model
        self.report = np.zeros([1, 4])
        self.TFNP_rate = np.array([0, 0, 0, 0])
        self.cv_counter = 0
        self.p0 = None
        self.p1 = None

    def design_CLF(self, clf_dict, sb_level=False):
        if clf_dict['model'] == 'LDA':
            # print('LDA SHRINKAGE:', clf_dict['shrinkage'])
            # if clf_dict['lda_solver'] == 'svd': lda_shrinkage = None
            # else:
            #     lda_shrinkage = clf_dict['shrinkage'] if clf_dict['shrinkage'] in [None,'auto'] else clf_dict['shrinkage']['shrinkage_float']
            # svc = LDA(solver=clf_dict['lda_solver'], shrinkage=lda_shrinkage)
            svc = LDA(solver=clf_dict['lda_solver'], shrinkage=None)
        if clf_dict['model'] == 'Bayes': svc = GaussianNB()
        if clf_dict['model'] == 'SVM':
            # print(clf_dict)
            # degree = clf_dict['kernel']['degree'] if clf_dict['kernel']['kf'] == 'poly' else 3
            # gamma = clf_dict['gamma'] if clf_dict['gamma'] in ['scale', 'auto'] else 10 ** (clf_dict['gamma']['gamma_float'])
            svc = SVC(kernel=clf_dict['kernel']['kf'],
                      C=10 ** (clf_dict['C']),
                      # C=clf_dict['C'],
                      gamma='scale',  # gamma=gamma,
                      degree=3,
                      probability=True
                      )
        if clf_dict['model'] == 'KNN':
            svc = KNeighborsClassifier(n_neighbors=int(clf_dict['neig']),
                                       metric=clf_dict['metric'],  # metric=clf_dict['metric']['mf'],
                                       p=3  # p=clf_dict['p']
                                       )  # minkowski,p=2 -> distancia euclidiana padrão
        if clf_dict['model'] == 'DTree':
            # print(clf_dict['min_split'])
            # if clf_dict['min_split'] == 1.0: clf_dict['min_split'] += 1
            # max_depth = clf_dict['max_depth'] if clf_dict['max_depth'] is None else int(clf_dict['max_depth']['max_depth_int'])
            svc = DecisionTreeClassifier(criterion=clf_dict['crit'], random_state=0,
                                         max_depth=None,  # max_depth=max_depth,
                                         min_samples_split=2
                                         # min_samples_split=clf_dict['min_split'], #math.ceil(clf_dict['min_split']),
                                         )
        if clf_dict['model'] == 'MLP':
            svc = MLPClassifier(verbose=False, max_iter=10000, tol=0.0001,
                                learning_rate_init=10 ** clf_dict['eta'],
                                # alpha=10 ** clf_dict['alpha'],
                                activation=clf_dict['activ']['af'],
                                hidden_layer_sizes=(int(clf_dict['n_neurons']), int(clf_dict['n_hidden'])),
                                learning_rate='constant',  # learning_rate=clf_dict['eta_type'],
                                solver=clf_dict['mlp_solver'],
                                )
        if sb_level: self.svc_sb = svc
        else: self.svc_final = svc

    def design_CSP(self, n_comp):
        # self.csp = CSP_v2(n_components=n_comp, reg=None, log=True, cov_est='epoch')
        self.csp = CSP(n_components=n_comp)
        self.lda = LDA()

    def evaluate(self, ap, epochs, labels):
        if ap.crossval:
            self.cross_scores = []
            self.cross_kappa = []
            kf = StratifiedShuffleSplit(ap.nfolds, test_size=ap.test_perc, random_state=42)
            # kf = StratifiedKFold(ap.nfolds, False)

            if ap.sb_approach:
                ## print('sbcsp with cross')
                for idx_treino, idx_teste in kf.split(epochs, labels):
                    acc_fold, kappa_fold = self.sbcsp_approach(ap, epochs[idx_treino], epochs[idx_teste], labels[idx_treino], labels[idx_teste])
                    self.cross_scores.append(acc_fold)
                    self.cross_kappa.append(kappa_fold)

            else:
                ## print('classic with cross')
                XF = ap.filter.apply_filter(epochs)

                if ap.filt_type == 'DFT':  # extrai somente os bins referentes à banda de interesse
                    bmin = round(ap.f_low * ap.dft_size_band)
                    bmax = round(ap.f_high * ap.dft_size_band)
                    XF = XF[:, :, bmin:bmax]
                    # print(bmin, bmax)

                # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
                # self.cross_scores = cross_val_score(self.chain, XF, self.labels, cv=kf)

                for idx_treino, idx_teste in kf.split(XF, labels):
                    XT, XV, yT, yV = XF[idx_treino], XF[idx_teste], labels[idx_treino], labels[idx_teste]

                    # Option 1
                    self.csp.fit(XT, yT)
                    XT_CSP = self.csp.transform(XT)
                    XV_CSP = self.csp.transform(XV)
                    self.svc_final.fit(XT_CSP, yT)
                    self.scores = self.svc_final.predict(XV_CSP)
                    self.csp_filters = self.csp.filters_

                    # Option 2
                    # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.svc_final)])
                    # self.chain.fit(XT, yT)
                    # self.scores = self.chain.predict(XV)
                    # self.csp_filters = self.chain['CSP'].filters_

                    acc_fold = np.mean(self.scores == yV)  # or self.cross_scores.append(self.chain.score(XV, yV))
                    kappa_fold = cohen_kappa_score(self.scores, yV)
                    self.cross_scores.append(acc_fold)
                    self.cross_kappa.append(kappa_fold)

            self.score = np.mean(self.cross_scores)
            self.kappa = np.mean(self.cross_kappa)

        else: # not cross-val
            test_size = int(len(epochs) * ap.test_perc)
            train_size = int(len(epochs) - test_size)
            train_size = train_size if (train_size % 2 == 0) else train_size - 1  # garantir balanço entre as classes (amostragem estratificada)
            epochsT, labelsT = epochs[:train_size], labels[:train_size]
            epochsV, labelsV = epochs[train_size:], labels[train_size:]
            XT = [epochsT[np.where(labelsT == i)] for i in ap.class_ids]  # Extrair épocas de cada classe
            XV = [epochsV[np.where(labelsV == i)] for i in ap.class_ids]
            XT = np.concatenate([XT[0], XT[1]])  # Train data classes A + B
            XV = np.concatenate([XV[0], XV[1]])  # Test data classes A + B
            yT = np.concatenate([ap.class_ids[0] * np.ones(int(len(XT) / 2)), ap.class_ids[1] * np.ones(int(len(XT) / 2))])
            yV = np.concatenate([ap.class_ids[0] * np.ones(int(len(XV) / 2)), ap.class_ids[1] * np.ones(int(len(XV) / 2))])
            # print(XT.shape, XV.shape)

            if ap.sb_approach:
                ## print('sbcsp without cross')
                self.score, self.kappa = self.sbcsp_approach(ap, XT, XV, yT, yV)

            else:
                ## print('classic without cross')
                XTF = ap.filter.apply_filter(XT)
                XVF = ap.filter.apply_filter(XV)

                if ap.filt_type == 'DFT':  # extrai somente os bins referentes à banda de interesse
                    bmin = round(ap.f_low * ap.dft_size_band)
                    bmax = round(ap.f_high * ap.dft_size_band)
                    # print(bmin, bmax, ap.dft_size_band)
                    XTF = XTF[:, :, bmin:bmax]
                    XVF = XVF[:, :, bmin:bmax]
                    # print(bmin, bmax)

                # Option 1
                self.csp.fit(XTF, yT)
                XT_CSP = self.csp.transform(XTF)
                XV_CSP = self.csp.transform(XVF)
                self.svc_final.fit(XT_CSP, yT)
                self.scores = self.svc_final.predict(XV_CSP)
                self.csp_filters = self.csp.filters_

                # Option 2
                # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.svc_final)])
                # self.chain.fit(XTF, yT)
                # self.scores = self.chain.predict(XVF)
                # self.csp_filters = self.chain['CSP'].filters_

                self.score = np.mean(self.scores == yV)  # or chain.score(XVF, yV)
                self.kappa = cohen_kappa_score(self.scores, yV)

        ### OLD
        # if ap.sb_approach:
        #     self.clf = Pipeline([('CSP', self.csp), ('LDA', self.svc_sb), ('SVC', self.svc_final)])
        #     if ap.crossval:
        #         ## print('sbcsp with cross')
        #         kf = StratifiedShuffleSplit(ap.n_folds, test_size=ap.test_perc, random_state=42)
        #         kf = StratifiedKFold(ap.n_folds, False)
        #         self.score = np.mean(
        #             [self.apply_sbcsp(ap, epochs[idx_treino], epochs[idx_teste], labels[idx_treino], labels[idx_teste])
        #              for idx_treino, idx_teste in kf.split(epochs, labels)])
        #     else:
        #         ## print('sbcsp without cross')
        #         train_size = int(len(epochs) - int(len(epochs) * ap.test_perc))
        #         train_size = train_size if (train_size % 2 == 0) else train_size - 1  # garantir balanço entre as classes (amostragem estratificada)
        #         epochsT, labelsT = epochs[:train_size], labels[:train_size]
        #         epochsV, labelsV = epochs[train_size:], labels[train_size:]
        #         cl = np.unique(labels)
        #         XT = [epochsT[np.where(labelsT == i)] for i in cl]  # Extrair épocas de cada classe
        #         XV = [epochsV[np.where(labelsV == i)] for i in cl]  # Extrair épocas de cada classe
        #         XT = np.concatenate([XT[0], XT[1]])  # Dados de treinamento das classes A, B
        #         XV = np.concatenate([XV[0], XV[1]])  # Dados de teste das classes A, B
        #         yT = np.concatenate([cl[0] * np.ones(int(len(XT) / 2)), cl[1] * np.ones(int(len(XT) / 2))])
        #         yV = np.concatenate([cl[0] * np.ones(int(len(XV) / 2)), cl[1] * np.ones(int(len(XV) / 2))])
        #         self.score = self.apply_sbcsp(ap, XT, XV, yT, yV)
        #
        # else:  # classic
        #     self.clf = Pipeline([('CSP', self.csp), ('SVC', self.svc_final)])
        #     epochs = ap.filter.apply_filter(epochs)
        #     if ap.crossval:
        #         ## print('classic with cross')
        #         cv = StratifiedShuffleSplit(ap.n_folds, test_size=ap.test_perc, random_state=42)
        #         # scores = cross_val_score(self.clf, epochs, labels, cv=cv, scoring=make_scorer(self.my_score))
        #         scores = []
        #         for idx_treino, idx_teste in cv.split(epochs, labels):
        #             XT, XV, yT, yV = epochs[idx_treino], epochs[idx_teste], labels[idx_treino], labels[idx_teste]
        #             self.clf.fit(XT, yT)
        #             self.csp_filters = self.clf['CSP'].filters_
        #             scores.append(self.clf.score(XV, yV))
        #         self.score = np.mean(scores)  # .mean()
        #     else:
        #         ## print('classic without cross')
        #         train_size = int(len(epochs) - int(len(epochs) * ap.test_perc))
        #         train_size = train_size if (
        #                     train_size % 2 == 0) else train_size - 1  # garantir balanço entre as classes (amostragem estratificada)
        #         epochsT, labelsT = epochs[:train_size], labels[:train_size]
        #         epochsV, labelsV = epochs[train_size:], labels[train_size:]
        #         cl = np.unique(labels)
        #         XT = [epochsT[np.where(labelsT == i)] for i in cl]  # Extrair épocas de cada classe
        #         XV = [epochsV[np.where(labelsV == i)] for i in cl]  # Extrair épocas de cada classe
        #         XT = np.concatenate([XT[0], XT[1]])  # Dados de treinamento das classes A, B
        #         XV = np.concatenate([XV[0], XV[1]])  # Dados de teste das classes A, B
        #         yT = np.concatenate([cl[0] * np.ones(int(len(XT) / 2)), cl[1] * np.ones(int(len(XT) / 2))])
        #         yV = np.concatenate([cl[0] * np.ones(int(len(XV) / 2)), cl[1] * np.ones(int(len(XV) / 2))])
        #         self.clf.fit(XT, yT)
        #         self.csp_filters = self.clf['CSP'].filters_
        #         self.score = self.clf.score(XV, yV)

    def sbcsp_approach(self, ap, XT, XV, yT, yV):

        n_bins = ap.f_high - ap.f_low
        overlap = 0.5 if ap.overlap else 1
        step = n_bins / ap.nbands
        size = step / overlap

        sub_bands = []
        for i in range(ap.nbands):
            fl_sb = round(i * step + ap.f_low)
            fh_sb = round(i * step + size + ap.f_low)
            if fh_sb <= ap.f_high: sub_bands.append([fl_sb, fh_sb])
            # se ultrapassar o limite superior da banda total, desconsidera a última sub-banda
            # ... para casos em que a razão entre a banda total e n_bands não é exata

        nbands = len(sub_bands)

        XTF, XVF = [], []

        if ap.filt_type == 'DFT':

            XT_FFT = ap.filter.apply_filter(XT)
            XV_FFT = ap.filter.apply_filter(XV)
            for i in range(nbands):
                bmin = sub_bands[i][0] * ap.dft_size_band
                bmax = sub_bands[i][1] * ap.dft_size_band
                XTF.append(XT_FFT[:, :, bmin:bmax])
                XVF.append(XV_FFT[:, :, bmin:bmax])
                # print(bmin, bmax)

        elif ap.filt_type in ['IIR' or 'FIR']:

            for i in range(nbands):
                filter_sb = Filter(sub_bands[i][0], sub_bands[i][1], len(XT[0, 0, :]), ap.sample_rate, forder=ap.order,
                                   filt_type=ap.filt_type)
                XTF.append(filter_sb.apply_filter(XT))
                XVF.append(filter_sb.apply_filter(XV))


        self.chain = [Pipeline([('CSP', CSP(n_components=ap.ncsp)), ('LDA', LDA())]) for i in range(nbands)]
        # self.chain = [Pipeline([('CSP', self.csp), ('LDA', self.svc_sb)]) for i in range(nbands)]

        for i in range(nbands): self.chain[i]['CSP'].fit(XTF[i], yT)

        XT_CSP = [self.chain[i]['CSP'].transform(XTF[i]) for i in range(nbands)]
        XV_CSP = [self.chain[i]['CSP'].transform(XVF[i]) for i in range(nbands)]

        SCORE_T = np.zeros((len(XT), nbands))
        SCORE_V = np.zeros((len(XV), nbands))
        for i in range(len(sub_bands)):
            self.chain[i]['LDA'].fit(XT_CSP[i], yT)
            SCORE_T[:, i] = np.ravel(self.chain[i]['LDA'].transform(XT_CSP[i]))  # classificações de cada época nas N sub bandas - auto validação
            SCORE_V[:, i] = np.ravel(self.chain[i]['LDA'].transform(XV_CSP[i]))

        self.csp_filters_sblist = [self.chain[i]['CSP'].filters_ for i in range(nbands)]
        self.lda_sblist = [self.chain[i]['LDA'] for i in range(nbands)]

        SCORE_T0 = SCORE_T[yT == ap.class_ids[0], :]
        SCORE_T1 = SCORE_T[yT == ap.class_ids[1], :]
        self.p0 = sst.norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
        self.p1 = sst.norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
        META_SCORE_T = np.log(self.p0.pdf(SCORE_T) / self.p1.pdf(SCORE_T))
        META_SCORE_V = np.log(self.p0.pdf(SCORE_V) / self.p1.pdf(SCORE_V))

        self.svc_final.fit(META_SCORE_T, yT)
        self.scores = self.svc_final.predict(META_SCORE_V)

        score = np.mean(self.scores == yV)
        kappa = cohen_kappa_score(self.scores, yV)
        return score, kappa

    def evaluate_epoch(self, ap, epoch, out_param='label'):
        # print(ap.learner.get_results())
        if ap.sb_approach:
            n_bins = ap.f_high - ap.f_low
            overlap = 0.5 if ap.overlap else 1
            step = n_bins / ap.nbands
            size = step / overlap
            sub_bands = []
            for i in range(ap.nbands):
                fl_sb = round(i * step + ap.f_low)
                fh_sb = round(i * step + size + ap.f_low)
                if fh_sb <= ap.f_high: sub_bands.append([fl_sb, fh_sb])
                # se ultrapassar o limite superior da banda total, desconsidera a última sub-banda
                # ... para casos em que a razão entre a banda total e n_bands não é exata
            nbands = len(sub_bands)
            XF = []
            if ap.filt_type == 'DFT':
                # print(epoch.shape)
                XFFT = ap.filter.apply_filter(epoch, is_epoch=True)
                for i in range(nbands):
                    bmin = sub_bands[i][0] * ap.dft_size_band
                    bmax = sub_bands[i][1] * ap.dft_size_band
                    XF.append(XFFT[:, bmin:bmax])
                    # print(bmin, bmax)
            elif ap.filt_type in ['IIR' or 'FIR']:
                for i in range(nbands):
                    filter_sb = Filter(sub_bands[i][0], sub_bands[i][1], len(epoch[0, :]), ap.sample_rate, forder=ap.order, filt_type=ap.filt_type)
                    XF.append(filter_sb.apply_filter(epoch, is_epoch=True))
            XCSP0 = [np.dot(self.csp_filters_sblist[i], XF[i]) for i in range(nbands)]
            XCSP = [np.log(np.mean(XCSP0[i] ** 2, axis=1)) for i in range(nbands)]
            # XCSP = [self.sb_csp_filters[i].transform(XF[i]) for i in range(nbands)]
            SCORE = np.zeros(nbands)
            SCORE = np.asarray([ np.ravel(self.lda_sblist[i].transform(XCSP[i].reshape(1, -1))) for i in range(nbands) ]).T
            META_SCORE = np.log(self.p0.pdf(SCORE) / self.p1.pdf(SCORE))
            guess_prob = self.svc_final.predict_proba(META_SCORE.reshape(1, -1))
            guess_label = self.svc_final.predict(META_SCORE.reshape(1, -1))
        else:
            XF = ap.filter.apply_filter(epoch, is_epoch=True)
            if ap.filt_type == 'DFT':  # extrai somente os bins referentes à banda de interesse
                bmin = round(ap.f_low * ap.dft_size_band)
                bmax = round(ap.f_high * ap.dft_size_band)
                # print(bmin, bmax, ap.dft_size_band)
                XF = XF[:, bmin:bmax] # XF[:, :, bmin:bmax]
            XT = np.dot(self.csp_filters, XF)
            epoch = np.log(np.mean(XT ** 2, axis=1))
            # guess_prob = self.clf.predict_proba(epoch) #old
            # guess_label = self.clf.predict(epoch) #old
            guess_prob = self.svc_final.predict_proba(epoch.reshape(1, -1))
            guess_label = self.svc_final.predict(epoch.reshape(1, -1))

        print(guess_prob, guess_label)
        if out_param == 'prob': return guess_prob
        elif out_param == 'label': return guess_label

    def get_results(self):
        return self.score

    def Get_model(self):
        return self.clf

    def my_score(self, y_true, y_pred):
        global_accuracy = accuracy_score(y_true, y_pred)
        return global_accuracy


######### BKP code versions ################
#
# def apply_sbcsp(self, ap, XT, XV, yT, yV):
#     if ap.filt_type == 'DFT':
#         XT_FFT = ap.filter.apply_filter(XT)
#         XV_FFT = ap.filter.apply_filter(XV)
#         n_bins = len(XT_FFT[0, 0, :])  # ou (fh-fl) * 4 # Número total de bins de frequencia
#     else:
#         n_bins = ap.f_high - ap.f_low
#     overlap = 0.5 if ap.overlap else 1
#     step = int(n_bins / ap.n_sbands)
#     size = int(step / overlap)  # tamanho fixo p/ todas sub bandas. overlap em 50%
#     SCORE_T = np.zeros((len(XT), ap.n_sbands))
#     SCORE_V = np.zeros((len(XV), ap.n_sbands))
#     self.csp_filters_sblist = []
#     self.lda_sblist = []
#     for i in range(ap.n_sbands):
#         if ap.filt_type == 'DFT':
#             bin_ini = i * step
#             bin_fim = i * step + size
#             if bin_fim >= n_bins: bin_fim = n_bins - 1
#             XTF = XT_FFT[:, :, bin_ini:bin_fim]
#             XVF = XV_FFT[:, :, bin_ini:bin_fim]
#         else:
#             fl_sb = i * step + ap.f_low
#             fh_sb = i * step + size + ap.f_low
#             if fh_sb > ap.f_high: fh_sb = ap.f_high
#             filter_sb = Filter(fl_sb, fh_sb, len(XT[0, 0, :]), ap.sample_rate, forder=ap.order,
#                                filt_type=ap.filt_type)
#             XTF = filter_sb.apply_filter(XT)
#             XVF = filter_sb.apply_filter(XV)
#
#         self.clf['CSP'].fit(XTF, yT)
#         XT_CSP = self.clf['CSP'].transform(XTF)
#         XV_CSP = self.clf['CSP'].transform(XVF)
#         self.clf['LDA'].fit(XT_CSP, yT)
#         SCORE_T[:, i] = np.ravel(
#             self.clf['LDA'].transform(XT_CSP))  # classificações de cada época nas N sub bandas - auto validação
#         SCORE_V[:, i] = np.ravel(self.clf['LDA'].transform(XV_CSP))
#
#         # self.lda_coefs = self.clf['LDA'].coef_
#         # print(self.csp_filters, self.lda_coef)
#         self.csp_filters_sblist.append(self.clf['CSP'].filters_)  #
#         self.lda_sblist.append(self.clf['LDA'])  #
#
#     classes = np.unique(yT)
#     SCORE_T0 = SCORE_T[yT == classes[0], :]
#     SCORE_T1 = SCORE_T[yT == classes[1], :]
#     self.p0 = sst.norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
#     self.p1 = sst.norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
#     META_SCORE_T = np.log(self.p0.pdf(SCORE_T) / self.p1.pdf(SCORE_T))
#     META_SCORE_V = np.log(self.p0.pdf(SCORE_V) / self.p1.pdf(SCORE_V))
#
#     self.clf['SVC'].fit(META_SCORE_T, yT)
#     scores = self.clf['SVC'].predict(META_SCORE_V)
#     return np.mean(scores == yV)
#
#     #     ap.learner.csp.fit(XTF, yT)
#     #     XT_CSP = ap.learner.csp.transform(XTF)
#     #     XV_CSP = ap.learner.csp.transform(XVF)
#     #     ap.learner.svc_sb.fit(XT_CSP, yT)
#     #     SCORE_T[:, i] = np.ravel(ap.learner.svc_sb.transform(XT_CSP))  # classificações de cada época nas N sub bandas - auto validação
#     #     SCORE_V[:, i] = np.ravel(ap.learner.svc_sb.transform(XV_CSP))  # validação
#     # classes = np.unique(yT)
#     # SCORE_T0 = SCORE_T[yT == classes[0], :]
#     # SCORE_T1 = SCORE_T[yT == classes[1], :]
#     # ap.learner.p0 = sst.norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
#     # ap.learner.p1 = sst.norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
#     # META_SCORE_T = np.log(ap.learner.p0.pdf(SCORE_T) / ap.learner.p1.pdf(SCORE_T))
#     # META_SCORE_V = np.log(ap.learner.p0.pdf(SCORE_V) / ap.learner.p1.pdf(SCORE_V))
#     # ap.learner.svc_final.fit(META_SCORE_T, yT)
#     # scores = ap.learner.svc_final.predict(META_SCORE_V)
#     # return np.mean(scores == yV)


# class CSP_v2:
#     """M/EEG signal decomposition using the Common Spatial Patterns (CSP).
#     This object can be used as a supervised decomposition to estimate
#     spatial filters for feature extraction in a 2 class decoding problem.
#     CSP in the context of EEG was first described in [1]; a comprehensive
#     tutorial on CSP can be found in [2].
#     Parameters
#     ----------
#     n_components : int (default 4)
#         The number of components to decompose M/EEG signals.
#         This number should be set by cross-validation.
#     reg : float | str | None (default None)
#         if not None, allow regularization for covariance estimation
#         if float, shrinkage covariance is used (0 <= shrinkage <= 1).
#         if str, optimal shrinkage using Ledoit-Wolf Shrinkage ('ledoit_wolf')
#         or Oracle Approximating Shrinkage ('oas').
#     log : bool (default True)
#         If true, apply log to standardize the features.
#         If false, features are just z-scored.
#     cov_est : str (default 'concat')
#         If 'concat', covariance matrices are estimated on concatenated epochs
#         for each class.
#         If 'epoch', covariance matrices are estimated on each epoch separately
#         and then averaged over each class.
#     Attributes
#     ----------
#     filters_ : ndarray, shape (n_channels, n_channels)
#         If fit, the CSP components used to decompose the data, else None.
#     patterns_ : ndarray, shape (n_channels, n_channels)
#         If fit, the CSP patterns used to restore M/EEG signals, else None.
#     mean_ : ndarray, shape (n_channels,)
#         If fit, the mean squared power for each component.
#     std_ : ndarray, shape (n_channels,)
#         If fit, the std squared power for each component.
#     """
#
#     def __init__(self, n_components=4, reg=None, log=True, cov_est="concat"):
#         """Init of CSP."""
#         self.n_components = n_components
#         self.reg = reg
#         self.log = log
#         self.cov_est = cov_est
#         self.filters_ = None
#         self.patterns_ = None
#         self.mean_ = None
#         self.std_ = None
#
#     def get_params(self, deep=True):
#         """Return all parameters (mimics sklearn API).
#         Parameters
#         ----------
#         deep: boolean, optional
#             If True, will return the parameters for this estimator and
#             contained subobjects that are estimators.
#         """
#         params = {"n_components": self.n_components, "reg": self.reg, "log": self.log}
#         return params
#
#     def fit(self, epochs_data, y):
#         """Estimate the CSP decomposition on epochs.
#         Parameters:
#         epochs_data: ndarray, shape (n_epochs, n_channels, n_times)
#             The data to estimate the CSP on.
#         y: array, shape (n_epochs,)
#             The class for each epoch.
#         Returns:
#         self : instance of CSP
#             Returns the modified instance. """
#
#         if not isinstance(epochs_data, np.ndarray):
#             raise ValueError("epochs_data should be of type ndarray (got %s)." % type(epochs_data))
#         epochs_data = np.atleast_3d(epochs_data)
#         e, c, t = epochs_data.shape
#         # check number of epochs
#         if e != len(y): raise ValueError("n_epochs must be the same for epochs_data and y")
#         classes = np.unique(y)
#         if len(classes) != 2: raise ValueError("More than two different classes in the data.")
#         if not (self.cov_est == "concat" or self.cov_est == "epoch"): raise ValueError(
#             "unknown covariance estimation method")
#
#         if self.cov_est == "concat":  # concatenate epochs
#             class_1 = np.transpose(epochs_data[y == classes[0]], [1, 0, 2]).reshape(c, -1)
#             class_2 = np.transpose(epochs_data[y == classes[1]], [1, 0, 2]).reshape(c, -1)
#             cov_1 = _regularized_covariance(class_1, reg=self.reg)
#             cov_2 = _regularized_covariance(class_2, reg=self.reg)
#         elif self.cov_est == "epoch":
#             class_1 = epochs_data[y == classes[0]]
#             class_2 = epochs_data[y == classes[1]]
#             cov_1 = np.zeros((c, c))
#             for t in class_1: cov_1 += _regularized_covariance(t, reg=self.reg)
#             cov_1 /= class_1.shape[0]
#             cov_2 = np.zeros((c, c))
#             for t in class_2: cov_2 += _regularized_covariance(t, reg=self.reg)
#             cov_2 /= class_2.shape[0]
#
#         # normalize by trace
#         cov_1 /= np.trace(cov_1)
#         cov_2 /= np.trace(cov_2)
#
#         e, w = lg.eigh(cov_1, cov_1 + cov_2)
#         n_vals = len(e)
#         # Rearrange vectors
#         ind = np.empty(n_vals, dtype=int)
#         ind[::2] = np.arange(n_vals - 1, n_vals // 2 - 1, -1)
#         ind[1::2] = np.arange(0, n_vals // 2)
#         w = w[:, ind]  # first, last, second, second last, third, ...
#         self.filters_ = w.T
#         self.patterns_ = lg.pinv(w)
#
#         pick_filters = self.filters_[:self.n_components]
#         X = np.asarray([np.dot(pick_filters, epoch) for epoch in epochs_data])
#
#         # compute features (mean band power)
#         X = (X ** 2).mean(axis=-1)
#
#         # To standardize features
#         self.mean_ = X.mean(axis=0)
#         self.std_ = X.std(axis=0)
#
#         return self
#
#     def transform(self, epochs_data, y=None):
#         """Estimate epochs sources given the CSP filters.
#         Parameters:
#         epochs_data : array, shape (n_epochs, n_channels, n_times)
#             The data.
#         y : None
#             Not used.
#         Returns:
#         X : ndarray of shape (n_epochs, n_sources)
#             The CSP features averaged over time.
#         """
#         if not isinstance(epochs_data, np.ndarray): raise ValueError(
#             "epochs_data should be of type ndarray (got %s)." % type(epochs_data))
#         if self.filters_ is None: raise RuntimeError('No filters available. Please first fit CSP decomposition.')
#         if epochs_data.ndim == 2:
#             pick_filters = self.filters_[:self.n_components]
#             X = np.asarray([np.dot(pick_filters, epochs_data)])
#         else:
#             pick_filters = self.filters_[:self.n_components]
#             X = np.asarray([np.dot(pick_filters, epoch) for epoch in epochs_data])
#
#         # compute features (mean band power)
#         X = (X ** 2).mean(axis=-1)
#         if self.log:
#             X = np.log(X)
#         else:
#             X -= self.mean_
#             X /= self.std_
#         return X

# def _regularized_covariance(data, reg=None):
#     if reg is None: cov = np.cov(data)
#     return cov
