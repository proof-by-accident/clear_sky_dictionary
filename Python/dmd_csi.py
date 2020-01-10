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
import matplotlib.image as image
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

from mpl_toolkits.mplot3d import Axes3D

def scale(x):
    return((x-np.mean(x))/np.sqrt(np.var(x)))

def del_T(t1,t2):
    t1 = str_to_tstamp(t1)
    t2 = str_to_tstamp(t2)

    dt = t1 - t2
    d = dt.days
    s = dt.seconds
    return(60.*24.*d + s/60.)

def fname_to_time(fname):
    year,mon,day,time = fname.split('/')[-4:]
    time = time.split('.')[0]
    tstamp = dt(*[int(_) for _ in [year,mon,day,time[:2],time[2:4],time[4:]]])
    return(tstamp.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('US/Mountain')) + td(hours=1))

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

def mode_crop(img, thresh=.5, step=5):
    mode_x, mode_y = np.where(img==img.max())
    p1 = [int(np.min(mode_x)),int(np.min(mode_y))]
    p2 = [int(np.max(mode_x)),int(np.max(mode_y))]
    img_crop = img[mode_x,mode_y]
    denom = img.sum()
    while (img_crop.sum()/denom) < thresh:
        p1[0] -= step
        p1[1] -= step
        p2[0] += step
        p2[1] += step
        img_crop = img[p1[0]:p2[0],p1[1]:p2[1]]

    #grad = imgrad_norm(img_crop)
    not_flat = 1# - (grad==0)

    return(not_flat*img_crop)

def mode_sub(img,mode):
    mode = mode.reshape((1536,1536),order='F')
    mode = mode/mode.max()
    return(img - img*mode)

def phi_filter(img,Phi):
    sub_modes = Phi[:,1:]
    for k in range(sub_modes.shape[1]):
        img = mode_sub(img,sub_modes[:,k])
    mode = Phi[:,0]
    mode = mode.reshape((1536,1536),order='F')
    mode = mode/mode.max()
    return(img*mode)

def phi_filter2(img,Phi):
    mode = Phi[:,0]
    mode = mode.reshape((1536,1536),order='F')
    mode = mode/mode.max()
    return(img*mode)

def implot3d(img):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    dim = img.shape[0]
    X,Y = np.meshgrid(np.linspace(0,dim,dim),np.linspace(0,dim,dim))
    ax.plot_wireframe(X,Y,img)
    plt.show()

def imgrad_norm(img):
    grad = np.gradient(img)
    return( np.sqrt(grad[0]**2 + grad[1]**2))


if __name__=='__main__':
    #exec(open('sun_position.py').read())
    IMAGE_DIR = './example_images/some_cloud/'
    DATA_DIR = os.path.abspath('/home/peter/Cloud_Dynamics/Data/clearsky_index')
    PICKLE_DIR = '/home/peter/Cloud_Dynamics/Python/pickle_dumps'

    try:
        #assert False
        csi = pd.read_csv(DATA_DIR+'/csi.csv')

    except:
        comps = pd.read_csv(DATA_DIR+'/csi_comps.csv')
        comps.columns = ['date','mst','ghi','ghi1','ghi2','getr','dni','detr']
        csi = pd.DataFrame({'gcsi': comps.ghi/comps.getr, 'dcsi': comps.dni/comps.detr})
        tlist = [0]*csi.shape[0]
        for i in range(csi.shape[0]):
            date = comps.date[i].split('/')
            time = comps.mst[i].split(':')
            tstamp = dt(*[int(_) for _ in [date[2],date[0],date[1],time[0],time[1]]])
            tlist[i] = tstamp.replace(tzinfo=pytz.timezone('US/Mountain'))

        csi.insert(1,'time',tlist)
        csi.to_csv(DATA_DIR+'/csi.csv', index=False)

    day ='/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs/11/2019/04/15'

test_fnames = []
for f in os.listdir(IMAGE_DIR):
    f = f.split('_')[0][-6:]
    test_fnames.append(day+'/'+f+'.jpg')

    mean_rbr = []
    sd_rbr = []
    med_rbr = []
    max_rbr = []
    business = []
    times = []
    mask = crop_mask(1536,1536)
    cyc = np.floor(len(test_fnames)/len(sun_modes))

    for i in range(len(test_fnames)):
        print(i/float(len(test_fnames)))
        try:
            Phi = curr_level[int(np.floor(i/cyc))]['Phi'][:(1536**2),:]
        except IndexError:
            Phi = curr_level[-1]['Phi'][:(1536**2),:]
        img = mode_crop(phi_filter2(mask*rbr_read(test_fnames[i]), np.abs(Phi)), thresh=.3)
        mean_rbr.append(np.mean(img))
        sd_rbr.append(np.sqrt(np.var(img)))
        med_rbr.append(np.median(img))
        max_rbr.append(np.max(img))
        business.append(np.mean(imgrad_norm(img)))
        times.append(fname_to_time(test_fnames[i]))

    csi_rbr = np.array(mean_rbr)*np.array(sd_rbr)

    im_stats = pd.DataFrame({'mean_rbr': mean_rbr,
                             'sd_rbr': sd_rbr,
                             'med_rbr': med_rbr,
                             'max_rbr': max_rbr,
                             'my_csi': csi_rbr,
                             'bus': business,
                             'time': [str(_) for _ in times]})#[str(_.replace(second=0)) for _ in times]})

    csi_test = im_stats.merge(csi,on='time')

    perfs = {'mean_rbr': rsqr(csi_test.mean_rbr,csi_test.dcsi),
             'sd_rbr': rsqr(csi_test.sd_rbr,csi_test.dcsi),
             'med_rbr': rsqr(csi_test.med_rbr,csi_test.dcsi),
             'max_rbr': rsqr(csi_test.max_rbr,csi_test.dcsi),
             'my_csi': rsqr(csi_test.my_csi,csi_test.dcsi),
             'bus': rsqr(csi_test.bus,csi_test.dcsi)}



N = len(test_fnames)
fig,ax=plt.subplots(5,10)
ax = ax.reshape((50))
for i in range(N):
    img = mode_crop(rbr_read(test_fnames[2*i]), thresh=.3)
    ax[i].imshow(img)
    ax[i].set_title(np.round(csi_test.mean_rbr[i],2))
#plt.show()
#
#T1 = csi.time
#T2 = im_stats.time
#T_incl = [np.any(_ == np.array(T1)) for _ in T2]
#nearTs =[]
#thresh = 30
#foo = 0.
#for t in T1:
#    if foo%10000==0:
#        print(foo/T)
#    foo += 1
#    for _t in T2:
#        if np.abs(del_T(t,_t))<thresh:
#            nearTs.append(t)
#            break
