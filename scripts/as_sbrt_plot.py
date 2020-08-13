# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
from hyperopt.plotting import main_plot_history, main_plot_histogram, main_plot_vars, main_plot_1D_attachment
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

ds = 'IV2a' # III3a, III4a, IV2a, IV2b, Lee19, CL, TW 
scenario = '' # '_s1_cortex' or '_s2_cortex'
cortex_only = True # used when ds == Lee19 - True to used only cortex channels

path_to_fig = '../as_results/sbrt20/' + ds + scenario + '/fig/' # PATH TO AUTO SETUP RESULTS FIGURES
# if not os.path.isdir(path_to_fig): os.makedirs(path_to_fig)

R = pd.read_pickle('../as_results/sbrt20/' + ds + scenario + '/RESULTS.pkl')  
# print(ds, R['acc'].mean(), R['acc'].median(), R['acc'].std(), R['acc'].max(), R['acc'].min())
# print(R[R['classes'] == '1 2'][['subj','acc']])

plt.rcParams.update({'font.size':12})
cores = ['olive','dimgray','darkorange','firebrick','lime','k','peru','c','purple','m','orange','firebrick','green','gray','hotpink']

prefix, suffix = '', ''
if ds == 'III3a':
    subjects = ['K3','K6','L1'] 
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]  
elif ds == 'III4a':
    subjects = ['aa','al','av','aw','ay']
    classes = [[1, 3]]
elif ds == 'IV2a':        
    subjects = range(1,10) 
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    prefix = 'A0'
elif ds == 'IV2b': 
    subjects = range(1,10)
    classes = [[1, 2]]
    prefix = 'B0'
elif ds == 'LINCE':
    subjects = ['CL_LR','CL_LF','TL_S1','TL_S2','WL_S1','WL_S2']
    classes = [[1, 2]] 
elif ds == 'Lee19':
    subjects = range(1, 55) 
    classes = [[1, 2]]
    prefix = 'S'

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
# plt.savefig(path_to_fig + 'scatter_as_vs_sbcsp_iir.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
acc_as = R['acc']*100
plt.figure(3, facecolor='mintcream')
plt.subplots(figsize=(10, 12), facecolor='mintcream')
ref = ['cla_iir','sb_iir']
# ref = ['sb_dft','sb_iir']
for i in range(2):
    acc_ref = R[ref[i]]*100
    plt.subplot(2, 1, i+1)
    plt.scatter(np.asarray(acc_ref).reshape(-1,1), np.asarray(acc_as).reshape(-1,1), facecolors = 'c', marker = 'o', s=50, alpha=.9, edgecolors='firebrick', zorder=3)
    plt.scatter(round(acc_ref.mean(),2), round(acc_as.mean(),2), facecolors = 'dodgerblue', marker = 'o', s=100, alpha=1, edgecolors='darkblue', label=r'Acurácia Média', zorder=5)
    plt.plot(np.linspace(40, 110, 1000), np.linspace(40, 110, 1000), color='dimgray', linewidth=1, linestyle='--', zorder=0) #linha pontilhada diagonal - limiar 
    plt.ylim((48, 102))
    plt.xlim((48, 102))
    plt.xticks(np.arange(50, 102, 5))
    plt.yticks(np.arange(50, 102, 5)) 
    plt.plot(np.ones(1000)*round(acc_ref.mean(),2), np.linspace(40, round(acc_as.mean(),2), 1000), color='dimgray', linewidth=.7, linestyle=':', zorder=0) # linha pontilhada verical - acc média auto setup
    plt.plot(np.linspace(40, round(acc_ref.mean(),2), 1000), np.ones(1000)*round(acc_as.mean(),2), color='dimgray', linewidth=.7, linestyle=':', zorder=0) # linha pontilhada horizontal - acc média ref
    plt.xlabel('Acurácia ' + ('CSP-LDA' if i==0 else 'SBCSP' ) + ' (configuração única) (%)', fontsize=12)
    plt.ylabel('Acurácia Auto Setup (%)', fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
# plt.savefig(path_to_fig + 'scatter_y.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

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

plt.plot(np.linspace(0, 10, 10), np.ones(10)*R['acc'].mean(), color='orange', linewidth=1.5, alpha=.3, linestyle='--', label='Média Auto Setup', zorder=0)
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
# plt.savefig(path_to_fig + 'boxplot_subj_acc_pairs.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
#%%
# X = S[['acc', 'cla_dft', 'cla_iir', 'sb_dft', 'sb_iir']]
X = R[['acc', 'sb_iir', 'cla_iir']]*100

plt.figure(figsize=(10,5), facecolor='mintcream')
plt.boxplot(X.T, boxprops={'color':'b'}, medianprops={'color': 'r'}, whiskerprops={'linestyle':'-.'}, 
            capprops={'linestyle':'-.'}) 

# plt.title('Diagrama de caixa comparativo entre a abordagem proposta e as abordagens clássica e SBCSP, da variação de performance de classificação para todos os sujeitos e todas 6 possíveis combinações binárias entre classes disponíveis no conjunto de dados BCI Competition IV 2a')
# plt.xticks([1,2,3,4,5,6],['Auto-Setup','CSP-LDA DFT','CSP-LDA IIR','SBCSP DFT','SBCSP IIR'])
plt.xticks([1,2,3],['Auto Setup','SBCSP','CSP-LDA'])
#plt.xlabel('Sujeito', size=14)
plt.ylabel('Acurácia (%)', size=14)
plt.yticks(np.arange(50, 101, step=5))
plt.ylim((48,102))
plt.savefig(path_to_fig + 'boxplot_approaches.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
#%%
plt.figure(figsize=(10,7))
n, bins, _ = plt.hist(R['sb_iir'], bins = 10)
plt.title('Frequência das acurácias obtidas para cada sujeito e par de classes no conjunto BCI Competition IV 2a com auto-setup ')
plt.xlabel('Acurácia')
plt.xlabel('Frequência')
# plt.savefig(path_to_fig + 'hist_accs_IV2a.pdf', format='pdf', dpi=300, transparent=True, bbox_inches='tight')

#%%
idx_max = []    
for class_ids in classes:
    for suj in subjects:
        sname = prefix + str(suj) + suffix
        path_to_trials = '../as_results/sbrt20/' + ds + '/' + sname + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
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
# plt.savefig(path_to_fig + 'hist_iter_max_acc.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

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
# plt.savefig(path_to_fig + 'bars_clf_events.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
class_ids = [1, 2]
suj = 1
sname = prefix + str(suj) + suffix
trials = pickle.load(open('../as_results/sbrt20/' + ds + '/' + sname + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl', 'rb'))
# main_plot_history(trials)
# main_plot_histogram(trials)
main_plot_vars(trials, do_show=False)

x = R['acc'] - R['sb_iir']

x.min()

# for p in pairs: S[S['classes'] == '1 2'][['subj','acc']
# n, bins, _ = plt.hist(, bins = 10, color='r', alpha='')
# plt.figure(figsize=(10,7))
# plt.hist2d(S['sb_iir_acc']*0.01, S['acc'], bins = 5)
# sns.distplot(idx_max, hist = True, kde = False, bins = 10, color = 'blue', hist_kws={'edgecolor': 'black'}, label='teste')
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