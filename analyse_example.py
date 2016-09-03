# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 22:43:10 2016

@author: kevin
"""
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
plotly.tools.set_credentials_file(username='kevyin', api_key='n3c33j5hac')
from ggplot import *

pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier

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
# plotly


# trace0 = Scatter(
#     x=[1, 2, 3, 4],
#     y=[10, 15, 13, 17]
# )
# trace1 = Scatter(
#     x=[16, 12, 13, 14],
#     y=[16, 5, 11, 9]
# )
# data = Data([trace0, trace1])
#
# py.plot(data, filename = 'basic-line')

#%%

# ggplot
ggplot(diamonds, aes(x='price', color='clarity')) + \
    geom_density() + \
    scale_color_brewer(type='div', palette=7) + \
    facet_wrap('cut')