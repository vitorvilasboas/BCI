#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue Jun 16 11:39:21 2020
# @author: vboas

import pandas as pd
path = '/home/vboas/cloud/results/'

# IV2A = pd.DataFrame(columns=['subj', 'A', 'B', 'fl', 'fh', 'tmin', 'tmax', 'nbands', 'ncsp', 'csp_list', 'clf', 'clf_details',
#                               'as_max', 'as_tune', 'as_mode', 'as_pmean', 'as_best', 'sb_iir', 'cla_iir', 'as_on', 'sb_on', 'cla_on', 
#                               'as_command', 'sb_command', 'cla_command', 'as_comlist', 'sb_comlist', 'cla_comlist'])
# LEE19 = IV2A.copy(deep=False)

# for s in range(1,10): IV2A.loc[len(IV2A)] = pd.read_pickle(path+'IV2a/R_'+str(s)+'.pkl').iloc[0]  
# pd.to_pickle(IV2A, path+'FINAL_IV2a.pkl')

# for s in range(1,55): LEE19.loc[len(LEE19)] = pd.read_pickle(path+'Lee19/R_'+str(s)+'.pkl').iloc[0]     
# pd.to_pickle(LEE19, path+'FINAL_Lee19.pkl')

IV2A = pd.read_pickle(path+'FINAL_IV2a_moderate.pkl')
LEE19 = pd.read_pickle(path+'FINAL_Lee19.pkl')

print(f"========= IV2a ========="); print(IV2A.iloc[:,12:25].mean(), '\n')
print(f"========= Lee19 ========"); print(LEE19.iloc[:,12:25].mean())