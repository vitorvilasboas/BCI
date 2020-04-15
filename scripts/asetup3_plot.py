# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
from hyperopt.plotting import main_plot_history, main_plot_histogram, main_plot_vars, main_plot_1D_attachment
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

plt.rcParams.update({'font.size':12})
cores = ['olive','dimgray','darkorange','firebrick','lime','k','peru','c','purple','m','orange','firebrick','green','gray','hotpink']

ds = 'IV2a' # III3a, III4a, IV2a, IV2b, Lee19, CL, TW 
fs = 250 if ds=='Lee19' else 100 if ds=='III4a' else 125 if ds=='CL' else 250
subjects = range(1,55) if ds=='Lee19' else ['aa','al','av','aw','ay'] if ds=='III4a' else ['K3','K6','L1'] if ds=='III4a' else range(1,10)
classes = [[1,3]] if ds=='III4a' else [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]] if ds in ['IV2a','III3a'] else [[1,2]]

if ds=='Lee19':
    only_cortex = True
    one_session = True
    lee_session = 1
    lee_option = ('_s' + str(lee_session) + '_cortex') if one_session and only_cortex else '_cortex' if only_cortex else ''

path_to_results = './asetup_results/RES_' + ds + ('.pkl' if ds!='Lee19' else (lee_option+'.pkl'))

path_to_figure = './asetup_figures/' # + ds + '_sess' + str(lee_session)
if not os.path.isdir(path_to_figure): os.makedirs(path_to_figure)

R = pd.read_pickle(path_to_results)  
print(ds, R['acc'].mean(), R['acc'].median(), R['acc'].std(), R['acc'].max(), R['acc'].min())

# Smil = pd.read_pickle('./asetup_results/old/BKP_1000iter/RES_' + ds + ('_1000.pkl' if ds!='Lee19' else (lee_option+'_1000.pkl')))
# print(ds, Smil['acc'].mean(), Smil['acc'].median(), Smil['acc'].std(), Smil['acc'].max(), Smil['acc'].min())

# Sold = pd.read_pickle('./asetup_results/old/RES_' + ds + ('.pkl' if ds!='Lee19' else (lee_option+'.pkl')))
# print(ds, Sold['acc'].mean(), Sold['acc'].median(), Sold['acc'].std(), Sold['acc'].max(), Sold['acc'].min(), '\n')
# S.to_csv('./asetup_results/RES_' + ds + '.csv') 

#%%
#print(S[S['classes'] == '1 2'][['subj','acc']])

#%%
acc_ref = R['sb_iir'] # cla,sb
acc_as = R['acc']

# regression_model = LinearRegression()
# regression_model.fit(np.asarray(acc_ref).reshape(-1,1), np.asarray(R['acc']).reshape(-1,1))
# reg = regression_model.predict(np.asarray(acc_ref).reshape(-1,1))

plt.figure(figsize=(10,7), facecolor='mintcream')
# plt.title('Gráfico de dispersão do desempenho da classificação individual para todas as combinações\nde classe (um contra um) no conjunto de dados BCI Competition IV 2a')
plt.scatter(np.asarray(acc_ref).reshape(-1,1), np.asarray(acc_as).reshape(-1,1), facecolors = 'c',
            marker = 'o', s=50, alpha=.9, edgecolors='firebrick', zorder=3)

plt.scatter(round(acc_ref.mean(),2), round(acc_as.mean(),2), 
            facecolors = 'dodgerblue', marker = 'o', s=100, alpha=1, 
            edgecolors='darkblue', label='Acurácia Média', zorder=5)

# plt.plot(acc_ref, reg, color='r', linewidth=.5, linestyle='--', label=('Curva de Regressão'))
plt.plot(np.linspace(0.30, 1.10, 1000), np.linspace(0.30, 1.10, 1000), 
         color='dimgray', linewidth=1, linestyle='--', zorder=0) #linha pontilhada diagonal - limiar 

plt.ylim((0.38, 1.02))
plt.xlim((0.38, 1.02))
plt.xticks(np.arange(0.40, 1.02, 0.05)) # ((0.5, 0.6, 0.7, 0.8, round(acc_ref.mean(), 2), 0.9, 1)) 
plt.yticks(np.arange(0.40, 1.02, 0.05)) # ((0.5, 0.6, 0.7, 0.8, 0.9, round(acc_as.mean(), 2), 1)) 

plt.plot(np.ones(1000)*round(acc_ref.mean(),2), np.linspace(0.30, round(acc_as.mean(),2), 1000), 
         color='dimgray', linewidth=.7, linestyle=':', zorder=0) # linha pontilhada verical - acc média auto setup

plt.plot(np.linspace(0.30, round(acc_ref.mean(),2), 1000), np.ones(1000)*round(acc_as.mean(),2), 
         color='dimgray', linewidth=.7, linestyle=':', zorder=0) # linha pontilhada horizontal - acc média ref

plt.xlabel('Acurácia SBCSP (setup fixo)', fontsize=12)
# plt.xlabel('Acurácia CSP-LDA (setup fixo)', fontsize=12)
plt.ylabel('Acurácia Auto Setup', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
# plt.grid(True, axis='y')
plt.savefig(path_to_figure + ds + '_scatter_as_vs_sbcsp_iir.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


#%%
pairs = ['1 2', '1 3', '1 4', '2 3', '2 4', '2 5']

media_as = R['acc'].mean()
media_sbcsp = R['sb_iir'].mean()
media_classic = R['cla_iir'].mean()

df = pd.DataFrame()
for suj in subjects:
    # print(df.shape[-1])
    df.insert(df.shape[-1], 'S{}'.format(suj), np.asarray(R[R['subj'] == suj]['acc']))
    
plt.figure(figsize=(10,7), facecolor='mintcream')

plt.plot(np.linspace(0, 10, 10), np.ones(10)*R['acc'].mean(), color='orange', linewidth=1.5, alpha=.3, linestyle='--', label='Média Autor Setup', zorder=0)
plt.plot(np.linspace(0, 10, 10), np.ones(10)*R['sb_iir'].mean(), color='aqua', linewidth=1.5, alpha=.3, linestyle='-.', label='Média SBCSP', zorder=0)
plt.plot(np.linspace(0, 10, 10), np.ones(10)*R['cla_iir'].mean(), color='pink', linewidth=1.8, alpha=1, linestyle=':', label='Média CSP-LDA', zorder=0)

plt.boxplot(df.T, boxprops={'color':'b'}, medianprops={'color': 'r'}, whiskerprops={'linestyle':'-.'}, 
            capprops={'linestyle':'-.'}, zorder=5) 

# plt.title('Diagrama de caixa representando a variação da performance de classificação em cada sujeito ao\nconsiderar os seis pares de classes avaliados para o conjunto de dados BCI Competition IV 2a')
plt.xticks(subjects,['S1','S2','S3','S4','S5','S6','S7','S8','S9'])

plt.legend(loc='lower right', fontsize=12)

# plt.yticks(np.linspace(0.6, 1., 5, endpoint=True))
plt.xlabel('Sujeito', size=14)
plt.ylabel('Acurácia (%)', size=14)
plt.yticks(np.arange(0.7, 1.01, step=0.05))
plt.ylim((0.68,1.02))
plt.xlim((0,10))
plt.savefig(path_to_figure + ds + '_boxplot_subj_acc_pairs.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    

#%%
# X = S[['acc', 'cla_dft', 'cla_iir', 'sb_dft', 'sb_iir']]
X = R[['acc', 'cla_iir', 'sb_iir']]

plt.figure(figsize=(10,5), facecolor='mintcream')
plt.boxplot(X.T, boxprops={'color':'b'}, medianprops={'color': 'r'}, whiskerprops={'linestyle':'-.'}, 
            capprops={'linestyle':'-.'}) 

# plt.title('Diagrama de caixa comparativo entre a abordagem proposta e as abordagens clássica e SBCSP, da variação de performance de classificação para todos os sujeitos e todas 6 possíveis combinações binárias entre classes disponíveis no conjunto de dados BCI Competition IV 2a')
# plt.xticks([1,2,3,4,5,6],['Auto-Setup','CSP-LDA DFT','CSP-LDA IIR','SBCSP DFT','SBCSP IIR'])
plt.xticks([1,2,3],['Auto Setup','CSP-LDA','SBCSP'])
#plt.xlabel('Sujeito', size=14)
plt.ylabel('Acurácia (%)', size=14)
plt.yticks(np.arange(0.50, 1.01, step=0.05))
plt.ylim((0.48,1.02))
plt.savefig(path_to_figure + ds + '_boxplot_all_approaches.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
#%%
plt.figure(figsize=(10,7))
n, bins, _ = plt.hist(R['sb_iir'], bins = 10)
plt.title('Frequência das acurácias obtidas para cada sujeito e par de classes no conjunto BCI Competition IV 2a com auto-setup ')
plt.xlabel('Acurácia')
plt.xlabel('Frequência')
plt.savefig(path_to_figure + ds + '_hist_accs_IV2a.pdf', format='pdf', dpi=300, transparent=True, bbox_inches='tight')

#%%
idx_max = []    
for class_ids in classes:
    for suj in subjects:    
        path_to_trials = './asetup_trials/' + ds + ((lee_option + '/') if ds=='Lee19' else '/') + ds + '_' + str(suj) + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
        trials = pickle.load(open(path_to_trials, 'rb'))
        all_loss = [ trials.trials[i]['result']['loss'] * (-1) for i in range(len(trials.trials)) ]
        idx_max.append(np.where(all_loss == np.asarray(all_loss).max())[0][0])

plt.figure(figsize=(10,7))
n, bins, _ = plt.hist(idx_max, range=(0, 2000), bins = 20, color='lightblue', histtype='bar', **{'edgecolor':'dimgray', 'linewidth':.3})
# plt.title('Densidade de iterações executadas pelo algoritmo de auto-setup até encontrar o conjunto de parâmetros de máximo desempenho de classificação ppara todos os sujeitos e pares de classes no conjunto de dados BCI Competition IV 2a')
# plt.xticks(range(0, 1100, 100))
plt.yticks(range(0, 15, 2))
plt.xticks(bins)
plt.xlim((-20,2020))
plt.ylim((0,15.5))
plt.xlabel('Iteração da Máxima Acurácia')
plt.ylabel('Frequência')
plt.savefig(path_to_figure + ds + '_hist_iter_max_acc.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
classifiers = np.unique(R['clf'])
x = np.arange(len(classifiers))
y = [ R.iloc[np.where(R['clf'] == clf)]['subj'].count() for clf in classifiers]
plt.figure(figsize=(10,7))
plt.bar(x,y, color='lightblue', tick_label=classifiers, **{'edgecolor':'dimgray', 'linewidth':.3})
plt.ylim(0,(np.asarray(y).max()+2))
plt.yticks(np.arange((np.asarray(y).max()+4), step=4))
plt.xlabel('Modelo de Classificação')
plt.ylabel('N. ocorrências auto setup')
plt.savefig(path_to_figure + ds + '_bars_clf_events.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
class_ids = [1, 2]
suj = 1
trials = pickle.load(open('./asetup_trials/' + ds + '/' + ds + '_' + str(suj) + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl', 'rb'))
# main_plot_history(trials)
# main_plot_histogram(trials)
main_plot_vars(trials, do_show=False)

x = R['acc'] - R['sb_iir']

x.min()

# for p in pairs:
#     S[S['classes'] == '1 2'][['subj','acc']

#n, bins, _ = plt.hist(, bins = 10, color='r', alpha='')
# plt.figure(figsize=(10,7))
# plt.hist2d(S['sb_iir_acc']*0.01, S['acc'], bins = 5)

# sns.distplot(idx_max, hist = True, kde = False, bins = 10, color = 'blue', 
#              hist_kws={'edgecolor': 'black'}, label='teste')

# plt.scatter(idx, range(1, len(idx)+1))
# plt.scatter(idx, maximos)

# plt.plot(range(1, len(idx)+1),np.ravel(idx), ls='-', linewidth=1, label='sen(x)')

# main_plot_history(trials)
# main_plot_histogram(trials)
# main_plot_vars(trials)

# all_loss = [ trials.trials[i]['result']['loss'] * (-1) for i in range(len(trials.trials)) ]
# all_loss1 = pd.Series(all_loss)
# idx = np.where(all_loss == np.asarray(all_loss).max())

# plt.plot(range(1, 1001),all_loss, ls='-', linewidth=1, label='sen(x)')


#%%

# suj = np.ravel(range(1,10))
# ncomp = np.ravel(range(2, 23, 2))
# sbands = np.ravel(range(1, 23))

# R = pd.read_pickle(os.path.dirname(__file__) + '/FBCSP_IIR_8a30.pickle')

# LH_RH = R.iloc[np.where(R['Classes']=='[1, 2]')]

# #N_CSP = [ [ R.iloc[ np.where((R['N CSP']==i)&(R['N Sbands']==j)) ] for j in sbands ] for i in ncomp ]

# # Melhores N sub-bandas para cada um dos pares de filtros CSP (Para 4 classes um_contra_um = 6 combinações)
# Step1_mean = [ [ R.iloc[ np.where((R['N CSP']==i)&(R['N Sbands']==j)) ]['Acc'].mean() for j in sbands ] for i in ncomp ] 
# Step2_nsb_best = np.ravel([ np.where(Step1_mean[i] == max(Step1_mean[i])) for i in range(len(Step1_mean)) ])
# Step3_acc_best = np.asarray([ Step1_mean[i][j] for i,j in zip(range(len(Step1_mean)), Step2_nsb_best) ])
# Best_Sb_NComp = pd.DataFrame(
#         np.concatenate([ncomp.reshape(11,1),
#                         Step2_nsb_best.reshape(11,1) + 1,
#                         Step3_acc_best.reshape(11,1)
#                         ], axis=1), columns=['N CSP','Best N Sb','Mean Acc'])

# # Melhores N sub-bandas para cada um dos pares de filtros CSP (Para 2 classes LHxRH)
# Step1_mean2 = [ [ LH_RH.iloc[ np.where((LH_RH['N CSP']==i)&(LH_RH['N Sbands']==j)) ]['Acc'].mean() for j in sbands ] for i in ncomp ] 
# Step2_nsb_best2 = np.ravel([ np.where(Step1_mean2[i] == max(Step1_mean2[i])) for i in range(len(Step1_mean2)) ])
# Step3_acc_best2 = np.asarray([ Step1_mean2[i][j] for i,j in zip(range(len(Step1_mean2)), Step2_nsb_best2) ])
# Best_Sb_NComp2 = pd.DataFrame(
#         np.concatenate([ncomp.reshape(11,1),
#                         Step2_nsb_best2.reshape(11,1) + 1,
#                         Step3_acc_best2.reshape(11,1)
#                         ], axis=1), columns=['N CSP','Best N Sb','Mean Acc'])

# # Média Acc por sujeito (4 classes)
# X1 = R.iloc[ np.where((R['N CSP']==10)&(R['N Sbands']==10)) ]
# MeanSuj1 = np.asarray([ X1.iloc[np.where(X1['Subj']==i)]['Acc'].mean() for i in suj ])
# print('Média Total: ',np.mean(MeanSuj1))

# # Média Acc por sujeito (LH x RH)
# X2 = LH_RH.iloc[ np.where((LH_RH['N CSP']==10)&(LH_RH['N Sbands']==10)) ]
# MeanSuj2 = np.asarray([ X2.iloc[np.where(X2['Subj']==i)]['Acc'].mean() for i in suj ])
# print('Média Total: ',np.mean(MeanSuj2)) # np.mean(X2['Acc'])

# S1 = R.iloc[np.nonzero(R['Subj']==1)] # R.iloc[np.where(R['Subj']==1)]

# #SumCost = R.iloc[np.where(R['Subj']==1) ]['Cost'].sum()
# #MeanSB = np.asarray([ LH_RH.iloc[np.where(LH_RH['N Sbands']== i)].iloc[:,3:5].mean() for i in sbands])

