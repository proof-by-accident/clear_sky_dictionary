import sys
import os
import itertools as it

import numpy as np
import numpy.linalg as npla

import pandas as pd

import scipy as sp
from scipy import ndimage as ndi
import scipy.signal as ss
import scipy.io

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


DATA_DIR = os.path.abspath('/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs')
CAM_DIRS = [os.path.join(DATA_DIR,'11'), os.path.join(DATA_DIR,'12') ]
YEAR_DIRS = [ os.path.join(cd,y) for cd in CAM_DIRS for y in os.listdir(cd) ]
MONTH_DIRS = [ os.path.join(y,m) for y in YEAR_DIRS for m in os.listdir(y) ]
DAY_DIRS = [ os.path.join(m,d)+'/' for m in MONTH_DIRS for d in os.listdir(m) ]
GIF_DIR = '/home/peter/Cloud_Dynamics/Python/gifs'


############################
# Functions to read in the TSI images located in DATA_DIR
############################
# Function names follow format XX_read where XX indicates the returned image type (eg. RGB values, grayscale, etc). Image values are generally scaled to range [0,1]
# Arguments:
#   `fname` filename of image, relative to DATA_DIR (so to read the image taken by camera 12 on April 15th, 2019 at 11:30 am you would input fname /11/2019/04/15/113000.jpg
#   `progbar` defaults to None in all functions. If a progress bar is provided it will increment its value by 1
#
# Author: Peter Shaffery
# Email: pshaff2@gmail.com
# License: None
###########################

# reads image specificed by `fname` into RGB data volume
def rgb_read(fname, progbar=None):
    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass
    return(cv2.imread(fname).astype(float)[...,[2,1,0]]/255.)

# returns grayscale image
def grayscale_read(fname, progbar=None):
    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass
    return(cv2.imread(fname,0).astype(float)/255.)


# the CV2 package calculates grayscale differently from the matplotlib library (as well as from MATLAB)
def grayscale_read_matlab(fname):
    im = hsv_read(fname)
    im[:,:,[0,1]] = 0.
    return(col.hsv_to_rgb(im)*255)

# reads grayscale or RGB image and resizes to provided size tuple
def grayscale_read_resize(fname, size, progbar=None):
    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass
    return(cv2.resize(cv2.imread(fname,0),size).astype(float)/255.)


def rgb_read_resize(fname, size, progbar=None):
    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass
    return(cv2.resize(cv2.imread(fname),size).astype(float)[:,:,[2,1,0]]/255.)

# returns HSV image
def hsv_read(fname, progbar=None):
    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass
    # return(cv2.imread(fname,cv2.COLOR_BGR2HSV).astype(float)/255.)
    return(col.rgb_to_hsv(rgb_read(fname)))


def hsv_read_resize(fname, size, progbar=None):
    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass
    return(cv2.resize(hsv_read(fname),size))


# creates a 2D circular mask for a square image of size `h`x`w` which sets all values outside the circle radius to 0. Radius determined by `crop_factor`*h
def crop_mask(h,w,crop_factor=.98):
    x1 = int( w-(1-crop_factor)*w/2 )
    y1 = int( h-(1-crop_factor)*h/2 ) 

    x0 = int( (1-crop_factor)*w/2 )
    y0 = int( (1-crop_factor)*h/2 )

    mask = Image.new('L',(h,w),0)
    draw = ImageDraw.Draw(mask)
    draw.pieslice([x0,y0,x1,y1],0,360,fill=1)
    return( np.array(mask) )


# same as rgb_read but also applies crop_mask
def mask_rgb_read(fname, progbar=None):
    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass
    img = cv2.imread(fname).astype(float)[:,:,[2,1,0]]/255.
    mask = crop_mask(img.shape[0], img.shape[1])
    img[:,:,0] = mask*img[:,:,0]
    img[:,:,1] = mask*img[:,:,1]
    img[:,:,2] = mask*img[:,:,2]
    return(img)


# converts RGB data volume to 2D Red-Blue Difference (RBD) image
def rgb_to_rbd(rgb, progbar=None):
    ret = np.ones(rgb.shape[:2])
    ret = (rgb[:,:,2]-rgb[:,:,0])/(ret+rgb[:,:,2]+rgb[:,:,0])
    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass   
    return(ret)


# converts RGB data volume to 2D Red-Blue Ratio (RBR) image
def rgb_to_rbr(rgb, progbar=None):
    ret = np.ones(rgb.shape[:2])
    ret = (rgb[:,:,0])/(ret+rgb[:,:,2])
    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass   
    return(ret)


# functions to read various color channels from image `fname`
def sat_read(fname):
    return(cv2.imread(fname, cv2.COLOR_BGR2HSV).astype(float)[:,:,1]/255.)

def sat_read2(fname):
    return(col.rgb_to_hsv(rgb_read(fname)).astype(float)[:,:,1])

def hue_read(fname):
    return(cv2.imread(fname, cv2.COLOR_BGR2HSV).astype(float)[:,:,0]/255.)

def val_read(fname):
    return(cv2.imread(fname, cv2.COLOR_BGR2HSV).astype(float)[:,:,2]/255.)

def rbd_read(fname, progbar=None):
    return(rgb_to_rbd(rgb_read(fname,progbar)))

def rbr_read(fname, progbar=None):
    return(rgb_to_rbr(rgb_read(fname,progbar)))

# compute various image statistics of image `fname`
def mean_sat_read(fname):
    return( np.mean( cv2.imread(fname,cv2.COLOR_RGB2HSV)[:,:,1] ) )

def sd_sat_read(fname):
    return( np.std( cv2.imread(fname,cv2.COLOR_RGB2HSV)[:,:,1] ) )

def frac_sat_read(fname, thresh=.2):
    hsv = cv2.imread(fname,cv2.COLOR_RGB2HSV)
    thresh = thresh*255
    return( np.sum(hsv[:,:,1] < thresh)/float(hsv.shape[0]*hsv.shape[1]) )

