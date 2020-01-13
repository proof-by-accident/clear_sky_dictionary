import os

import numpy as np
import numpy.linalg as npla

import pandas as pd

import scipy as sp
from scipy import ndimage as ndi

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

import gc

def plot_bgr(fname, size=5000):
    img = img.imread(fname)
    img_flat = np.reshape(img,(img.shape[0]**2,img.shape[-1]))
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x = np.random.choice(img_flat[:,0], size=size)
    y = np.random.choice(img_flat[:,1], size=size)
    z = np.random.choice(img_flat[:,2], size=size)
    ax.scatter(x,y,z)               
    ax.set_xlabel('Blue')
    ax.set_ylabel('Green')
    ax.set_zlabel('Red')
    plt.show()

def fname_to_datetime(fname):
    year,month,day,time = fname.split('/')[-4:]
    time = time.split('.')[0]
    return(pd.to_datetime(year+month+day+time,format='%Y%m%d%H%M%S'))

def rgb_read(fname, progbar=None):
    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass
    
    return(cv2.imread(fname))

def hsv_read_resize(fname, size, progbar=None):
    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass
    return(cv2.resize(cv2.imread(fname,cv2.COLOR_BGR2HSV),size).astype(float)/255.)


def hsv_read1(fname):
    rgb = cv2.imread(fname)[...,[2,1,0]]
    hsv = col.rgb_to_hsv(rgb.astype(float)/255.)
    hsv[...,2] = hsv[...,2]/hsv[...,2].max()
    return(255.*hsv)

    
def hsv_read2(fname):
    return(cv2.resize(cv2.imread(fname,cv2.COLOR_BGR2HSV),(1536,1536)).astype(float)/255.)
    

def rgb_plot(fname, progbar=None):
    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass

   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    rgb = rgb_read(fname)

    h,w = (1536,1536)
    mask = crop_mask(h,w)
    
    red = (rgb[:,:,2]*mask).flatten()
    green = (rgb[:,:,1]*mask).flatten()
    blue = (rgb[:,:,0]*mask).flatten()

    red = red[red!=0]
    green = green[green!=0]
    blue = blue[blue!=0]
    
    n_subset = 1000
    subset = np.random.choice(range(len(red)), n_subset)
    
    (red,green,blue) = (red[subset]/max(red[subset]),
                        green[subset]/max(green[subset]),
                        blue[subset]/max(blue[subset]))
    
    c = np.array([red,green,blue]).T
    
    ax.scatter(red,green,blue, c=c)
    
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    
    plt.show()
    plt.close()

def hsv_plot(fname):   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    rgb = rgb_read(fname)
    hsv = col.rgb_to_hsv(rgb[:,:,[2,1,0]].astype(float)/255.)
    #hsv = hsv_read_resize(fname,(1536,1536))
       
    
    h,w = (1536,1536)
    mask = crop_mask(h,w)
    
    red = (rgb[:,:,2]*mask).flatten()
    green = (rgb[:,:,1]*mask).flatten()
    blue = (rgb[:,:,0]*mask).flatten()    
    
    hue = (hsv[:,:,0]*mask).flatten()
    sat = (hsv[:,:,1]*mask).flatten()
    val = (hsv[:,:,2]*mask).flatten()
    
    n_subset = 1000
    subset = np.random.choice(range(len(red)), n_subset)
    
    (hue,sat,val) = (hue[subset]/max(hue[subset]),
                     sat[subset]/max(sat[subset]),
                     val[subset]/max(val[subset]))

    (red,green,blue) = (red[subset]/max(red[subset]),
                        green[subset]/max(green[subset]),
                        blue[subset]/max(blue[subset]))
    
    c = np.array([red,green,blue]).T
    
    ax.scatter(hue,sat,val,c=c)
    
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')
    
    plt.show()
    plt.close()

def animate_csd_sv(csd, intvl=100):    
    fig = plt.figue()
    ims = []

    n_subset = 1000
    subset = np.random.choice(range(1536**2), n_subset)

    for (hr,fname) in zip(csd.keys(),csd.values()):        
        rgb = rgb_read(fname)
        hsv = col.rgb_to_hsv(rgb[:,:,[2,1,0]])
        
        h,w = (1536,1536)
        mask = crop_mask(h,w)
        
        red = (rgb[:,:,2]*mask).flatten()
        green = (rgb[:,:,1]*mask).flatten()
        blue = (rgb[:,:,0]*mask).flatten()    
        
        hue = (hsv[:,:,0]*mask).flatten()
        sat = (hsv[:,:,1]*mask).flatten()
        val = (hsv[:,:,2]*mask).flatten()
        
        (hue,sat,val) = (hue[subset]/max(hue[subset]),
                         sat[subset]/max(sat[subset]),
                         val[subset]/max(val[subset]))
        
        (red,green,blue) = (red[subset]/max(red[subset]),
                            green[subset]/max(green[subset]),
                            blue[subset]/max(blue[subset]))

        
        im = plt.scatter(sat,val,c=np.array([red,green,blue]).T)
        ims.append([im])

    an = ani.ArtistAnimation(fig,ims,interval=intvl,blit=True,repeat_delay=100)
    plt.show()

    
def rgb_to_rbr(rgb, progbar=None):
    ret = np.ones(rgb.shape[:2])
    ret = rgb[:,:,2]/(ret+rgb[:,:,0])

    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass
    
    return(ret)

def rbr_read(fname, progbar=None):
    return(rgb_to_rbr(rgb_read(fname,progbar)))

def crop_mask(h,w,crop_factor=.98):
    x1 = int( w-(1-crop_factor)*w/2 )
    y1 = int( h-(1-crop_factor)*h/2 ) 

    x0 = int( (1-crop_factor)*w/2 )
    y0 = int( (1-crop_factor)*h/2 )

    mask = Image.new('L',(h,w),0)
    draw = ImageDraw.Draw(mask)
    draw.pieslice([x0,y0,x1,y1],0,360,fill=1)
    return( np.array(mask) )

    
def day_clearsky_dict_make(day_dir, mask, progbar=None):
    image_files = [os.path.join(day_dir,f) for f in os.listdir(day_dir)]

    img = None
    stats = [[None]*6]*len(image_files)
    for i in range(len(image_files)):
        f = image_files[i]
        hr = fname_to_datetime(f).hour
        
        img = mask*rbr_read(f,progbar)
        
        stats[i] = [f,hr,np.mean(img),np.median(img),np.max(img),np.std(img)]
        

    stats = pd.DataFrame(stats)        
    stats.columns = ['fname','hour','mean','median','max','sd']
    stats['clearsky_index'] = stats['mean']*stats['sd']

    PICKLE_DIR = os.path.abspath('/home/peter/Cloud_Dynamics/Python/pickle_dumps')
    with open(os.path.join(PICKLE_DIR,day_dir.split('/')[-4]+'_'+day_dir.split('/')[-1].split('.')[0]+'.p'),'wb') as f:
        p.dump(stats,f)

    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass

    rgb_images = None
    rbr_images = None
    gc.collect()
    return(stats)

DATA_DIR = os.path.abspath('/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs')
CAM_DIRS = [os.path.join(DATA_DIR,'11'), os.path.join(DATA_DIR,'12') ]
YEAR_DIRS = [ os.path.join(cd,y) for cd in CAM_DIRS for y in os.listdir(cd) ]
MONTH_DIRS = [ os.path.join(y,m) for y in YEAR_DIRS for m in os.listdir(y) ]
DAY_DIRS = [ os.path.join(m,d) for m in MONTH_DIRS for d in os.listdir(m) ]

d=2
fnames = [DAY_DIRS[d] + '/' + _ for _ in os.listdir(DAY_DIRS[d])]
test_fname = DAY_DIRS[0] + '/' + os.listdir(DAY_DIRS[0])[0]
clearsky_dict = p.load(open('./pickle_dumps/clearsky_dict.p','rb'))

if __name__=='__main__':    
    #assert False

    prog_len = len([f for dd in DAY_DIRS for f in os.listdir(dd)]) + len(DAY_DIRS)

    h,w = (1536,1536)
    mask = crop_mask(h,w)

    with progressbar.ProgressBar(max_value=prog_len) as progbar:
        full_stats = []
        for dd in DAY_DIRS:
            day_stats = day_clearsky_dict_make(dd,mask,progbar=progbar)
            full_stats.append(day_stats)

    with open(os.path.join('/home/peter/Cloud_Dynamics/Python/pickle_dumps','full_stats.p'),'wb') as f:
        p.dump(full_stats,f)

    full_stats = pd.concat(full_stats).reset_index()

    clearsky_ids = full_stats.groupby('hour').idxmin()['clearsky_index']
    clearsky_dict = {k: full_stats['fname'][int(clearsky_ids[k])] for k in clearsky_ids.keys()}

    with open(os.path.join('clearsky_dict.p'),'wb') as f:
        p.dump(clearsky_dict,f)
