# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 22:43:10 2016

@author: kevin
"""
#%%
import numpy as np
import pandas as pd

pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
figsize(15, 5)

#%%
df_failed = pd.read_csv('data/failed_cycle123_2.txt.gz')
df_normal = pd.read_csv('data/normal_cycle123_2.txt.gz')
df_failed.dtypes
df_normal.dtypes

#%%
plt.figure(1)
called_int_cols = df_failed.columns[df_failed.columns.str.match('Called')]
df_failed_byLaneCycle = df_failed.groupby(['RunFolder', 'Lane', 'Cycle','Read'])[called_int_cols].mean()
ax1 = plt.subplot(1,2,1)
df_failed_byLaneCycle.boxplot()

df_normal_byLaneCycle = df_normal.groupby(['RunFolder', 'Lane', 'Cycle','Read'])[called_int_cols].mean()
plt.subplot(1,2,2, sharey=ax1)

df_normal_byLaneCycle.boxplot()


#%%


#%%

