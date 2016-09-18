# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 18:21:41 2016

@author: kevin
"""

import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
plotly.tools.set_credentials_file(username='kevyin', api_key='n3c33j5hac')
from ggplot import *
import dateutil.parser

lsfile = 'lsdata.txt'

# read ls -l --full-time output
def parselslist(lsfile):
    with open(lsfile, 'r') as f:
        for line in f:
            squished_line = re.sub( '\s+', ' ', line ).strip()
            split_line = squished_line.split(' ')
            date_str = ' '.join(split_line[5:8])
            #date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            date = dateutil.parser.parse(date_str)
            yield date
#%%         
it = parselslist(lsfile)    

s = pd.Series(numpy.array(list(it)))

#%%
#s.sort_values(inplace=True)
min_date = s.min()
print min_date
s_diff = s.apply(lambda x: (x - min_date).total_seconds()/2600)
s_diff.sort_values(inplace=True)
s_diff.index = range(1,len(s_diff) + 1)


#%%
#print s
print s_diff
d = {'index' : pd.Series(np.arange(0,s.size)), 'timediff': s_diff}
df = pd.DataFrame(d)
#print df
#print s
print "FWEF"
df.head()
p = ggplot(df, aes(x='index', y='timediff', color = 'timediff') )
print 'afwef'
p + geom_point()