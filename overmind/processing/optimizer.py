import pickle
from hyperopt import base, fmin, tpe, hp
from processing.processor import Approach
from processing.utils import PATH_TO_SESSION

class Optimize:
    def __init__(self):
        pass

    def run_optimizer(self, session, load_last_setup):
        self.session = session
        self.ap = Approach(self.session)
        self.ap.set_channels(self.session.dp.channels)
        self.ap.set_eeg_path(self.session.dp.eeg_path)
        max_knn_neig = int((self.session.dp.eeg_info['trials_per_class'] * 2) * self.session.dp.test_perc)
        max_ncomp = len(self.ap.channels) # int(len(self.ap.channels) / 2)
        # print(max_ncomp, max_knn_neig)
        # if self.session.info.is_dataset:
        #     if (self.session.info.ds_name + '_' + self.session.info.ds_subject) in ['III 3a_K6', 'III 3a_L1', 'III 4a_aa']: max_ncomp = int(len(self.ap.channels)/2)
        self.space = (
            hp.uniformint('fl', 0, 20),
            hp.uniformint('fh', 30, 49),
            hp.quniform('tmin', 0, 2, 0.5),
            hp.quniform('tmax', 2, self.session.dp.eeg_info['trial_mi_time'], 0.5),
            hp.quniform('ncomp', 2, max_ncomp, 2),  # 21 #14 #116
            hp.choice('approach', [
                {'option':'classic', },
                {'option':'sbcsp', 'nbands': hp.uniformint('nbands', 2, 48)},
            ]),
            hp.choice('filt', [
                {'design':'DFT'},
                # {'design':'IIR', 'iir_order': hp.uniformint('iir_order', 1, 8)},
                # {'design':'FIR', 'fir_order': hp.uniformint('fir_order', 2, 7)},
            ]),
            hp.choice('clf', [
                {'model':'Bayes'},
                {'model':'LDA',
                 'lda_solver': hp.choice('lda_solver', ['svd','lsqr','eigen']), #
                 # 'shrinkage': hp.choice('shrinkage', [None, 'auto', {'shrinkage_float':  hp.uniform('shrinkage_float', 0, 1)}]) #np.logspace(-4, 0, 1)
                 },
                {'model':'KNN',
                 'neig': hp.uniformint('neig', 2, max_knn_neig), # 'neig': hp.quniform('neig', 2, max_knn_neig, 1),
                 'metric': hp.choice('metric', ['euclidean','manhattan','minkowski','chebyshev']), #{'mf':'cityblock'}, {'mf':'cosine'}, {'mf':'l1'}, {'mf':'l2'},
                 # 'p': hp.quniform('p', 2, 50, 1)
                 },
                {'model':'SVM',
                 'C': hp.quniform('C', -8, 4, 1), # np.logspace(-8, 4, 13), # hp.quniform('C', -8, 4, 1) hp.lognormal('C', 0, 1),
                 'kernel': hp.choice('kernel', [{'kf':'linear'}, {'kf':'poly'}, {'kf':'sigmoid'}, {'kf':'rbf'}]), #, 'width': hp.lognormal('width', 0, 1) # 'degree': hp.uniformint('degree', 2, 3)
                 # 'gamma': hp.choice('gamma', ['scale', 'auto', {'gamma_float': hp.quniform('gamma_float', -9, 4, 1)}]), # hp.loguniform('gamma_float', -9, 3)  np.logspace(-9, 3, 13)),
                 },
                {'model':'MLP',
                 'eta': hp.quniform('eta', -5, -2, 1),  # hp.quniform('eta', 0.0001, 0.1, 0.0001)    hp.choice('eta', [0.1,0.01,0.001,0.0001]),
                 'n_neurons': hp.quniform('n_neurons', 50, 500, 50), # hp.uniformint('n_neurons', 50, 500),
                 'n_hidden': hp.uniformint('n_hidden', 1, 2),
                 'activ': hp.choice('activ', [{'af':'identity'}, {'af':'logistic'}, {'af':'tanh'}, {'af':'relu'}]),
                 'mlp_solver': hp.choice('mlp_solver', ['adam', 'lbfgs', 'sgd']),
                 # 'alpha': hp.quniform('alpha', -8, 1, 1),  # hp.lognormal('alpha', 0, 1),
                 # 'eta_type': hp.choice('eta_type', ['constant', 'invscaling', 'adaptive']),
                 },
                {'model':'DTree',
                 'crit': hp.choice('crit', ['gini', 'entropy']),
                 #'max_depth': hp.choice('max_depth', [None, {'max_depth_int': hp.qlognormal('max_depth_int', 3, 1, 1)}]), # np.random.lognormal(3, 1, 1) ]),
                 #'min_split': hp.uniform('min_split', 0.0001, 1), #  np.random.lognormal(2, 1, 1) # hp.qlognormal('min_split', 2, 1, 1)
                }
            ])
        )

        if load_last_setup:
            try:
                print('Trying to pickle file')
                self.trials = pickle.load(open(PATH_TO_SESSION + self.session.info.nickname + '/trials_setup.pkl', 'rb'))
            except:
                print('No trial file at specified path, creating new one')
                self.trials = base.Trials()
            else: print('File found')
        else:
            print('No load last trial file, creating new one')
            self.trials = base.Trials()

        try:
            print('Size of object: ' + str(len(self.trials)))
            self.best = fmin(self.validate, space=self.space, algo=tpe.suggest, max_evals=len(self.trials) + self.session.dp.n_iter, trials=self.trials)
            pickle.dump(self.trials, open(PATH_TO_SESSION + self.session.info.nickname + '/trials_setup.pkl', 'wb'))
            # print(self.best)
        except:
            print('Exception raised')
            pickle.dump(self.trials, open(PATH_TO_SESSION + self.session.info.nickname + '/trials_setup.pkl', 'wb'))
            # print(self.trials.best_trial['misc']['vals'])
            raise

    def validate(self, args):
        print(args)
        fl, fh, tmin, tmax, ncomp, approach, filt, clf = args
        while(tmax-tmin) < 1: tmax += 0.5  # garante janela mínima de 1seg
        if approach['option'] == 'sbcsp':
            if int(approach['nbands']) >= (int(fh)-int(fl)): approach['nbands'] = (int(fh)-int(fl)) # -1
        self.ap.define_approach(
            True if approach['option'] == 'sbcsp' else False, self.session.acq.sample_rate, int(fl), int(fh),
            int(ncomp), self.session.dp.class_ids, tmin, tmax, filt['design'], {'model':'LDA', 'lda_solver':'svd'}, clf,
            None if filt['design'] == 'DFT' else filt['iir_order'] if filt['design'] == 'IIR' else filt['fir_order'],
            None if approach['option'] == 'classic' else int(approach['nbands']),
            self.session.dp.overlap, self.session.dp.cross_val, self.session.dp.n_folds, self.session.dp.test_perc
            # True, False, 10, 0.2
        )
        return self.ap.validate_model() * (-1)

    def saveOptTrial(self, path):
        # path += '/optrial_' + datetime.now().strftime('%d-%m-%Y_%Hh%Mm') + '.pkl'
        # pickle.dump(self.trials, open(path + '/asetup_trials.pkl', 'wb'))
        pickle.dump(self.trials, open(path+'/trials_setup.pkl', 'wb'))

    def loadOptTrial(self, path):
        # pattern = re.compile(r"optrial_.*?\.pkl")  # mask to get filenames
        # pkl_files = []
        # for root, dirs, files in os.walk(path):  # add .kv files of view/kv in kv_files vector
        #     pkl_files += [root + '/' + file_ for file_ in files if pattern.match(file_)]
        # try:
        #     last_file = max(pkl_files, key=os.path.getmtime).split('/')[-1]
        #     path += '/' + last_file
        # except:
        #     print('no trials optimize on user directory')
        #     path += '/'
        return pickle.load(open(path+'/trials_setup.pkl', 'rb'))


    ######### BKP code versions ################
    # def csp_lda(self, args):
    #     # print('csp_lda: ', args, end='')
    #     filt, fl, fh, tmin, tmax, n_comp, classifier, X, y = args
    #     start = time()  # start timer
    #     if filt['filt']['filt_type'] == 'FFT':  # FFT - Spectral filtering
    #         bmin = int(fl * (fs / nf))
    #         bmax = int(fh * (fs / nf))
    #         XFF = fft(X)
    #         REAL = np.transpose(np.real(XFF)[:, :, bmin:bmax], (2, 0, 1))
    #         IMAG = np.transpose(np.imag(XFF)[:, :, bmin:bmax], (2, 0, 1))
    #         XF0 = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    #         XF = np.transpose(XF0, (1, 2, 0))
    #     elif filt['filt']['filt_type'] == 'IIR':  # IIR - Temporal filtering
    #         if fl == 0: fl = 0.001
    #         Wnl = fl / nf
    #         Wnh = fh / nf
    #         if Wnh >= 1: Wnh = 0.99
    #         b, a = sp.butter(filt['filt']['order'], [Wnl, Wnh], btype='bandpass')  # to filt IIR
    #         XF = sp.lfilter(b, a, X)
    #
    #     if classifier['clf']['type'] == 'LDA':
    #         clf = LDA()
    #
    #     if classifier['clf']['type'] == 'SVM':
    #         clf = SVC(kernel=classifier['clf']['kernel']['ktype'],
    #                   C=10 ** (classifier['clf']['C']))
    #
    #     if classifier['clf']['type'] == 'KNN':
    #         clf = KNeighborsClassifier(n_neighbors=int(classifier['clf']['n_neighbors']),
    #                                    metric='minkowski',
    #                                    p=2)  # minkowski e p=2 -> para usar distancia euclidiana padrão
    #
    #     if classifier['clf']['type'] == 'DTree':
    #         clf = DecisionTreeClassifier(criterion=classifier['clf']['criterion'],
    #                                      max_depth=classifier['clf']['max_depth'],
    #                                      min_samples_split=math.ceil(classifier['clf']['min_samples_split']),
    #                                      random_state=0)  # None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
    #
    #     if classifier['clf']['type'] == 'Bayes':
    #         clf = GaussianNB()
    #
    #     if classifier['clf']['type'] == 'MLP':
    #         clf = MLPClassifier(verbose=False,
    #                             max_iter=10000,
    #                             tol=0.0001,
    #                             activation=classifier['clf']['activation']['act_type'],
    #                             learning_rate_init=10 ** classifier['clf']['eta'],
    #                             alpha=10 ** classifier['clf']['alpha'],
    #                             learning_rate=classifier['clf']['eta_schedule']['eta_type'],
    #                             solver=classifier['clf']['solver']['solver_type'],
    #                             hidden_layer_sizes=(int(classifier['clf']['hidden_n_neurons']), int(classifier['clf']['n_hidden'])))
    #
    #     process = Pipeline([('CSP', CSP(n_comp)), ('classifier', clf)])
    #     # kf = StratifiedShuffleSplit(self.session.dp.n_folds, test_size=self.session.dp.test_perc, random_state=42)
    #     kf = StratifiedKFold(self.session.dp.n_folds, False)
    #     scores = cross_val_score(process, XF, y, cv=kf)
    #     cost = time() - start
    #     acc = np.mean(scores)
    #     print(acc)
    #     return acc, cost
    #
    # def sbcsp(self, args):
    #     # print('sbcsp: ', args, end='')
    #     filt, fl, fh, tmin, tmax, n_comp, n_bands, classifier, X, y = args
    #     # kf = StratifiedShuffleSplit(self.session.dp.n_folds, test_size=self.session.dp.test_perc, random_state=42)
    #     kf = StratifiedKFold(self.session.dp.n_folds, False)
    #
    #     start = time()  # start timer
    #     cross_scores = []
    #     for idx_treino, idx_teste in kf.split(X, y):
    #         XT = X[idx_treino]
    #         XV = X[idx_teste]
    #         yT = y[idx_treino]
    #         yV = y[idx_teste]
    #
    #         if filt['filt']['filt_type'] == 'FFT':  # FFT - Spectral filtering
    #             bmin = int(fl * (fs / nf))
    #             bmax = int(fh * (fs / nf))
    #             filtered = fft(XT)
    #             REAL = np.transpose(np.real(filtered)[:, :, bmin:bmax], (2, 0, 1))
    #             IMAG = np.transpose(np.imag(filtered)[:, :, bmin:bmax], (2, 0, 1))
    #             filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    #             XT_FFT = np.transpose(filtered, (1, 2, 0))
    #             filtered = fft(XV)
    #             REAL = np.transpose(np.real(filtered)[:, :, bmin:bmax], (2, 0, 1))
    #             IMAG = np.transpose(np.imag(filtered)[:, :, bmin:bmax], (2, 0, 1))
    #             filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    #             XV_FFT = np.transpose(filtered, (1, 2, 0))
    #
    #         # Divide sub-bands
    #         if filt['filt']['filt_type'] == 'FFT':
    #             n_bins = len(XT_FFT[0, 0, :])  # ou (fh-fl) * 4 # Número total de bins de frequencia
    #         elif filt['filt']['filt_type'] == 'IIR':
    #             n_bins = fh - fl
    #         overlap = 2
    #         step = int(n_bins / n_bands)
    #         size = int(step * overlap)  # tamanho fixo p/ todas sub bandas. overlap em 50%
    #
    #         # Make sub-bands limits and Temporal/Spectral filtering
    #         SCORE_T = np.zeros((len(XT), n_bands))
    #         SCORE_V = np.zeros((len(XV), n_bands))
    #         for i in range(n_bands):
    #             if filt['filt']['filt_type'] == 'FFT': # Only FFT Spectral filtering
    #                 bin_ini = i * step
    #                 bin_fim = i * step + size
    #                 if bin_fim >= n_bins: bin_fim = n_bins - 1
    #                 XTF = XT_FFT[:, :, bin_ini:bin_fim]
    #                 XVF = XV_FFT[:, :, bin_ini:bin_fim]
    #             elif filt['filt']['filt_type'] == 'IIR': # Only IIR Temporal filtering
    #                 fl_sb = i * step + fl
    #                 fh_sb = i * step + size + fl
    #                 if fl_sb == 0: fl_sb = 0.001
    #                 if fh_sb > fh: fh_sb = fh
    #                 # print(fl_sb, fh_sb, nf, fl_sb/nf, fh_sb/nf)
    #                 if fl_sb == 0: fl_sb = 0.001
    #
    #                 Wnl = fl_sb / nf
    #                 Wnh = fh_sb / nf
    #                 if Wnh >= 1: Wnh = 0.99
    #
    #                 b, a = sp.butter(filt['filt']['order'], [Wnl, Wnh], btype='bandpass')  # to filt IIR
    #                 XTF = sp.lfilter(b, a, XT)  # comment here
    #                 XVF = sp.lfilter(b, a, XV)  # comment here
    #
    #             csp = CSP(n_components=n_comp)
    #             csp.fit(XTF, yT)
    #             XT_CSP = csp.transform(XTF)
    #             XV_CSP = csp.transform(XVF)
    #
    #             clf = LDA()
    #             clf.fit(XT_CSP, yT)
    #             SCORE_T[:, i] = np.ravel(clf.transform(XT_CSP))  # classificações de cada época nas N sub bandas - auto validação
    #             SCORE_V[:, i] = np.ravel(clf.transform(XV_CSP))  # validação
    #
    #         # Meta-classificador Bayesiano
    #         SCORE_T0 = SCORE_T[yT == 0, :]
    #         m0 = np.mean(SCORE_T0, axis=0)  # media classe A
    #         std0 = np.std(SCORE_T0, axis=0)  # desvio padrão classe A
    #
    #         SCORE_T1 = SCORE_T[yT == 1, :]
    #         m1 = np.mean(SCORE_T1, axis=0)
    #         std1 = np.std(SCORE_T1, axis=0)
    #
    #         p0 = norm(m0, std0)  # p0 e p1 representam uma distribuição normal de médias m0 e m1, e desvio padrão std0 e std1
    #         p1 = norm(m1, std1)
    #
    #         META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
    #         META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
    #
    #         # SVM on top of the meta-classifier
    #         # svc = SVC(kernel="linear", C=10 ** Clog)
    #
    #         if classifier['clf']['type'] == 'LDA':
    #             svc = LDA()
    #
    #         if classifier['clf']['type'] == 'SVM':
    #             svc = SVC(kernel=classifier['clf']['kernel']['ktype'],
    #                       C=10 ** (classifier['clf']['C']))
    #
    #         if classifier['clf']['type'] == 'KNN':
    #             svc = KNeighborsClassifier(n_neighbors=int(classifier['clf']['n_neighbors']),
    #                                        metric='minkowski',
    #                                        p=2)  # minkowski e p=2 -> para usar distancia euclidiana padrão
    #
    #         if classifier['clf']['type'] == 'DTree':
    #             svc = DecisionTreeClassifier(criterion=classifier['clf']['criterion'],
    #                                          max_depth=classifier['clf']['max_depth'],
    #                                          min_samples_split=math.ceil(classifier['clf']['min_samples_split']),
    #                                          random_state=0)  # None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
    #
    #         if classifier['clf']['type'] == 'Bayes':
    #             svc = GaussianNB()
    #
    #         if classifier['clf']['type'] == 'MLP':
    #             svc = MLPClassifier(verbose=False,
    #                                 max_iter=10000,
    #                                 tol=0.0001,
    #                                 activation=classifier['clf']['activation']['act_type'],
    #                                 learning_rate_init=10 ** classifier['clf']['eta'],
    #                                 learning_rate=classifier['clf']['eta_schedule']['eta_type'],
    #                                 solver=classifier['clf']['solver']['solver_type'],
    #                                 hidden_layer_sizes=(int(classifier['clf']['hidden_n_neurons']), int(classifier['clf']['n_hidden'])))
    #
    #         svc.fit(META_SCORE_T, yT)
    #         scores = svc.predict(META_SCORE_V)
    #         cross_scores.append(np.mean(scores == yV))
    #
    #     cost = time() - start  # stop timer (cost estimate)
    #     acc = np.mean(cross_scores)
    #
    #     return acc, cost

















    # self.space = ({'filt': hp.choice('filt', [{'design': 'DFT', },
    #                                           # {'design':'IIR', 'iir_order': hp.quniform('iir_order', 1, 8, 1)},
    #                                           # {'design':'FIR', 'fir_order': hp.quniform('fir_order', 2, 7, 1)},
    #                                           ])},
    #               hp.quniform('fl', 0, 10, 1),
    #               hp.quniform('fh', 30, 51, 1),
    #               hp.quniform('tmin', 2, 4, 0.5),
    #               hp.quniform('tmax', 4, 6, 0.5),
    #               hp.quniform('ncomp', 2, 21, 2),  # 21 #14 #116
    #
    #               {'approach': hp.choice('approach', [  ##{'approach' : hp.choice('app_test', [
    #                   {'option': 'csp-lda', },
    #                   {'option': 'sbcsp', 'nbands': hp.quniform('nbands', 2, 50, 1)},
    #                   ## {'option':'sbcsp', 'n_sbands': hp.quniform('sbcsp_n_sbands', 2, 50, 1)},
    #               ])},
    #
    #               {'clf': hp.choice('clf_type', [
    #                   {'type': 'Bayes', },
    #                   {'type': 'LDA', },
    #                   {'type': 'KNN', 'n_neighbors': hp.quniform('n_neighbors', 2, 50, 1), },  # 50 #20
    #                   {'type': 'SVM',
    #                    'C': hp.quniform('Clog', -8, 4, 1),  # hp.lognormal('svm_C', 0, 1),
    #                    'kernel': hp.choice('svm_kernel', [{'ktype': 'linear'}, {'ktype': 'poly'}, {'ktype': 'sigmoid'},
    #                                                       {'ktype': 'rbf',
    #                                                        # 'width': hp.lognormal('svm_rbf_width', 0, 1)
    #                                                        }]), },
    #                   {'type': 'MLP',
    #                    # 'eta': hp.quniform('eta', 0.0001, 0.1, 0.0001),
    #                    # 'eta': hp.choice('eta', [0.1,0.01,0.001,0.0001]),
    #                    'eta': hp.quniform('eta', -8, 1, 1),
    #                    # 'eta_schedule': hp.choice('eta_schedule', [{'eta_type': 'constant'}, {'eta_type': 'invscaling'}, {'eta_type': 'adaptive'}]),
    #                    # 'solver': hp.choice('solver', [{'solver_type': 'adam'}, {'solver_type': 'lbfgs'}, #{'solver_type': 'sgd'},]),
    #                    'alpha': hp.quniform('alpha', -8, 1, 1),  # 'alpha': hp.lognormal('alpha', 0, 1),
    #                    'hidden_n_neurons': hp.quniform('hidden_n_neurons', 50, 500, 50),
    #                    'n_hidden': hp.quniform('n_hidden', 1, 4, 1),
    #                    'activation': hp.choice('activation', [{'act_type': 'identity'},
    #                                                           {'act_type': 'logistic'},
    #                                                           {'act_type': 'tanh'},
    #                                                           {'act_type': 'relu'}, ]), },
    #
    #                   {'type': 'DTree',
    #                    'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
    #                    # 'max_depth': hp.choice('dtree_max_depth', [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
    #                    # 'min_samples_split': hp.quniform('dtree_min_samples_split', 2, 100, 1),
    #                    }, ])})
    #
    # # 'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),},])})













    # filt_, fl, fh, tmin, tmax, ncomp, approach, clf_type = args
    # # print(args)
    # while (tmax - tmin) < 1: tmax += 0.5  # garante janela minima de 1seg
    # if approach['approach']['option'] == 'sbcsp':
    #     if int(approach['approach']['nbands']) > (int(fh) - int(fl)): approach['approach']['nbands'] = (int(fh) - int(
    #         fl)) - 1
    #
    # self.ap.define_approach(True if approach['approach']['option'] == 'sbcsp' else False, self.session.acq.sample_rate,
    #                         int(fl), int(fh), int(ncomp), self.session.dp.class_ids, tmin, tmax,
    #                         filt_['filt']['filt_type'],
    #                         {'type': 'LDA'}, clf_type['clf'],
    #                         None if filt_['filt']['filt_type'] == 'DFT' else filt_['filt']['order'],
    #                         1 if approach['approach']['option'] == 'csp-lda' else int(approach['approach']['nbands']),
    #                         # True, True, 10, 0.2)
    #                         True, False, 2, 0.5)
    # return self.ap.validate_model() * (-1)