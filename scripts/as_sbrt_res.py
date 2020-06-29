#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:51:30 2020

@author: vboas
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

subjects = range(1,10) 
classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]

FINAL = pd.DataFrame(columns=['as_train','as_train_tune','as_max','as_tune','as_mode','as_pmean',
                              'acc_mlr','as_best','sb_dft','sb_iir','cla_dft','cla_iir'])

# ### 10xRandomSplit  5x5-folds
# R = pd.read_pickle("/home/vboas/cloud/devto/BCI/as_results/sbrt20/cv10_cv5/R9.pkl")
# for suj in subjects:
#     for class_ids in classes:
#         r = R.loc[(R['subj'] == suj) & (R['A'] == class_ids[0]) & (R['B'] == class_ids[1])]
#         FINAL.loc[len(FINAL)] = r.iloc[:,12:].mean()
        
# # REF = pd.DataFrame(columns=['sb_dft','sb_iir','cla_dft','cla_iir'])
# # R = pd.read_pickle("/home/vboas/cloud/devto/BCI/as_results/sbrt20/cv10_ref_half_train/RESULTS_9.pkl")
# # for suj in subjects:
# #     for class_ids in classes:
# #         r = R.loc[(R['subj'] == suj) & (R['A'] == class_ids[0]) & (R['B'] == class_ids[1])]
# #         REF.loc[len(REF)] = r.iloc[:,3:].mean()
# # FINAL['sb_dft'] = REF['sb_dft']
# # FINAL['sb_iir'] = REF['sb_iir']
# # FINAL['cla_dft'] = REF['cla_dft']
# # FINAL['cla_iir'] = REF['cla_iir']

# 10xFixSplit  5x5folds       

FINAL = pd.DataFrame(columns=['as_train','as_train_tune','as_max','as_tune','as_mode',
                              'as_pmean','as_best','sb_dft','sb_iir','cla_dft','cla_iir'])

for suj in subjects:
    for class_ids in classes:
        r = pd.DataFrame(columns=['as_train','as_train_tune','as_max','as_tune','as_mode','as_pmean','as_best','sb_dft','sb_iir','cla_dft','cla_iir'])
        for i in range(10):
            R = pd.read_pickle("/home/vboas/cloud/devto/BCI/as_results/sbrt20/master_cv5/RESULTS_"+str(i)+".pkl")
            a = R.loc[(R['subj'] == suj) & (R['A'] == class_ids[0]) & (R['B'] == class_ids[1])]
            b = a.iloc[:,12:]
            r.loc[len(r)] = b.mean()
        
        FINAL.loc[len(FINAL)] = r.mean()
            
print(FINAL.std())
               
##%% PLOT GRAFIC #####################################################################
acc_as = FINAL['as_best']
ref = ['cla_dft','sb_dft']
plt.rcParams.update({'font.size':12})
plt.figure(3, facecolor='mintcream')
plt.subplots(figsize=(10, 12), facecolor='mintcream')
for i in range(2):
    acc_ref = FINAL[ref[i]]
    plt.subplot(2, 1, i+1)
    plt.scatter(np.asarray(acc_ref).reshape(-1,1), np.asarray(acc_as).reshape(-1,1), facecolors = 'c', marker = 'o', s=50, alpha=.9, edgecolors='firebrick', zorder=3)
    plt.scatter(acc_ref.mean(), acc_as.mean(), facecolors = 'dodgerblue', marker = 'o', s=100, alpha=1, edgecolors='darkblue', label=r'Acurácia Média', zorder=5)
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
plt.savefig('/home/vboas/Desktop/scatter_y_9.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# X = S[['acc', 'cla_dft', 'cla_iir', 'sb_dft', 'sb_iir']]
X = FINAL[['as_best', 'sb_dft', 'cla_dft']]

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
plt.savefig('/home/vboas/Desktop/boxplot_approaches_9.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    