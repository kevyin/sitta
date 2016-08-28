# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 21:45:11 2016

@author: kevin
"""
#%%

from __future__ import division
from numpy import *
from sympy import *
x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)

expr = ( x + y ) ** 3
expr
expr.expand()


#%%

import os

print os.getcwd()
print os.listdir('data')

#%%
def average(a, b):
    """
    Given two numbers a and b, return their average value.

    Parameters
    ----------
    a : number
      A number
    b : number
      Another number

    Returns
    -------
    res : number
      The average of a and b, computed using 0.5*(a + b)

    Example
    -------
    >>> average(5, 10)
    7.5

    """

    return (a + b) * 0.5

#%%
%matplotlib qt
import pylab
pylab.plot(range(10), 'o')