import numpy as np
import pandas as pd
import sys
import itertools as it
import math
from copy import copy, deepcopy
import datetime
import os

import IPython.display as ipd
digits = 3
pd.options.display.chop_threshold = 10**-(digits+1)
pd.options.display.float_format = lambda x: '{0:.{1}f}'.format(x,digits)
pd.options.display.show_dimensions = True
def display(X):
    if isinstance(X, pd.Series) or (isinstance(X, np.ndarray) and X.ndim <=2):
        ipd.display(pd.DataFrame(X))
    else:
        ipd.display(X)

import matplotlib.pyplot as plt
plt.style.use("classic")
plt.style.use("fivethirtyeight")
plt.rc("figure", figsize=(5,5))
from mpl_toolkits.mplot3d import Axes3D

#Below is a collection of utility functions I have found useful on prior projects.
#Not all of them will be used in this project.

def listify(L):
    x = type(L)
    if x is 'tuple': 
        return list(L)
    elif x is 'list':
        return L
    elif x is 'np.ndarray':
        return L.tolist()
    else:
        try:
            return [L]
        except:
            return L

def masarray(v):
    v = np.asarray(v)
    if(v.ndim == 0):
        v = v.reshape(1)
    return v
       
def tile_rows(v,n):
    return np.tile(v,(n,1))

def tile_cols(v,n):
    return np.tile(v[:,np.newaxis],(1,n))

def margins(A):
    df = pd.DataFrame(A).copy()
    df.loc['TOTAL'] = df.sum(axis=0)
    df['TOTAL'] = df.sum(axis=1)
    df.ix['TOTAL','TOTAL'] = df['TOTAL'].sum()
    return df

def get_summary_stats(v):    
    ss = pd.DataFrame(v).describe().T
    ss['SE'] = ss['std'] / np.sqrt(ss['count'])
    ss['count'] = int(ss['count'])#.astype('np.uint16')
    return ss

def solve_quadratic(a, b, c):
    with np.errstate(invalid='ignore', divide='ignore'):
        d = b**2 - 4*a*c
        d[d<0] = np.inf
        e = np.sqrt(d)
        x = np.stack([(-b-e)/(2*a), (-b+e)/(2*a)])
        idx = (a==0)
        x_a = -1*c/b
        x_a = np.stack([x_a, x_a])
        x[:,idx] = x_a[:,idx]
        return x


DIV1 = "*" * 80
DIV2 = "#" * 80


def wedge(a,b):
    return np.outer(b,a)-np.outer(a,b)

def rbroadcast(a,b):
    a = np.asarray(a)
    b = np.asarray(b)
    a_sh_old = a.shape
    b_sh_old = b.shape
    d_dim = len(a_sh_old) - len(b_sh_old)
    if(d_dim > 0):
        b_sh_new = list(b_sh_old)
        b_sh_new.extend([1]*d_dim)
        b = b.reshape(b_sh_new)
    elif(d_dim < 0):
        a_sh_new = list(a_sh_old)
        a_sh_new.extend([1]*(-d_dim))
        a = a.reshape(a_sh_new)
    try:
        return np.broadcast_arrays(a,b)
    except:
        raise Exception("Can't make %s and %s compatible"%(a_sh_old, b_sh_old))
    
def mmul(a,b):
    a,b = rbroadcast(a,b)
    c = a*b
    return c

def mdot(a,b,axis=0):
    A = mmul(a,b).sum(axis=axis)
    return A

def mag(v,axis=0):
    A = mdot(v,v,axis)
    B = np.sqrt(A)
    return B

def unit(v,axis=0):
    m = mag(v,axis)
    v /= m
    return v