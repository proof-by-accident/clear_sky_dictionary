import os
import itertools as it

import numpy as np
import numpy.linalg as npla

import pandas as pd

import scipy as sp
from scipy import ndimage as ndi
import scipy.signal as ss

import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.animation as ani
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D

import time as t
import pickle as p
import progressbar

from datetime import datetime as dt
from datetime import timezone as tz
from datetime import timedelta as td
import pytz

import gc

from read_funcs import *

DATA_DIR = os.path.abspath('/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs')
CAM_DIRS = [os.path.join(DATA_DIR,'11'), os.path.join(DATA_DIR,'12') ]
YEAR_DIRS = [ os.path.join(cd,y) for cd in CAM_DIRS for y in os.listdir(cd) ]
MONTH_DIRS = [ os.path.join(y,m) for y in YEAR_DIRS for m in os.listdir(y) ]
DAY_DIRS = [ os.path.join(m,d)+'/' for m in MONTH_DIRS for d in os.listdir(m) ]

DATA_DIR = os.path.abspath('/home/peter/Cloud_Dynamics/Data/clearsky_index')
PICKLE_DIR = '/home/peter/Cloud_Dynamics/Python/pickle_dumps'
comps = pd.read_csv(DATA_DIR+'/csi_comps.csv')
comps.columns = ['date','mst','ghi','ghi1','ghi2','getr','dni','detr']

def fname_to_time(fname):
    year,mon,day,time = fname.split('/')[-4:]
    time = time.split('.')[0]
    
    tstamp = dt(*[int(_) for _ in [year,mon,day,time[:2],time[2:4],time[4:]]])
    return( tstamp.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('US/Mountain')) )


def min_round(t):
    s = t.second
    t -= td(seconds=s)
    return(t)

def str_to_tstamp(t):
    time = t[:-6]
    tz = t[-5:].split(':')
    tz = tz[0] + tz[1]
    return(dt.strptime(time+' -'+tz, "%Y-%m-%d %H:%M:%S %z"))

def rsqr(x,y):
    lin = np.polyfit(x,y,1)
    y_hat = x*lin[0] + lin[1]
    ssr = np.sum((y - y_hat)**2)
    sst = np.sum((y - np.mean(y))**2)
    return(1 - (ssr/sst))
    
try:
    csi = pd.read_csv(DATA_DIR+'/csi.csv')
    csi.time = csi.time.map(str_to_tstamp)

except:
    csi = pd.DataFrame({'gcsi': comps.ghi/comps.getr})
    tlist = [0]*csi.shape[0]
    for i in range(csi.shape[0]):
        date = comps.date[i].split('/')
        time = comps.mst[i].split(':')
        
        tstamp = dt(*[int(_) for _ in [date[2],date[0],date[1],time[0],time[1]]])
        tlist[i] = tstamp.replace(tzinfo=pytz.timezone('US/Mountain'))
        
    csi.insert(1,'time',tlist)
    csi.to_csv(DATA_DIR+'/csi.csv', index=False)

day = DAY_DIRS[1]
test_fnames = [day + _ for _ in os.listdir(day)]

piks = os.listdir(PICKLE_DIR)[:-5]
rsqs = []

mask = crop_mask(1536,1536)
def rsq_pik(pf):
    test = p.load(open(PICKLE_DIR+'/'+pf,'rb'))
    test['time'] = test.fname.map(fname_to_time)

    test.time = test.time.astype('object')
    csi_test = test.merge(csi,on='time')

    return([rsqr(csi_test['mean'],csi_test['gcsi']),
            rsqr(csi_test['median'],csi_test['gcsi']),
            rsqr(csi_test['sd'],csi_test['gcsi']),
            rsqr(csi_test['max'],csi_test['gcsi']),
            rsqr(csi_test['clearsky_index'],csi_test['gcsi'])])

test = pd.DataFrame( np.array([rsq_pik(_) for _ in piks]))
test.columns = ['rbr_mean','rbr_median','rbr_sd','rbr_max','rbr_my_csi']
test['file'] = piks

perfs = test.loc[:,test.columns!='file'].apply(np.mean,0)

test_file = piks[4]
test_file = p.load(open(PICKLE_DIR+'/'+test_file,'rb'))
test_file['time'] = test_file.fname.map(fname_to_time).astype(object)

csi_test = test_file.merge(csi,on='time')
comp_var ='mean'
plt.scatter(csi_test.loc[:,comp_var],csi_test.gcsi)
plt.xlabel('RBR '+comp_var)
plt.ylabel('Clearness Index')
plt.show()

csd = p.load(open(PICKLE_DIR+'./clearsky_dict.p','rb'))


full_stats = p.load(open(PICKLE_DIR+'/full_stats.p','rb'))
