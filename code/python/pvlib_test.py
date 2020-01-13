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
from datetime import datetime as dt
import pytz as tz
import pickle as p
import progressbar

import gc

import pvlib as pv

from read_funcs import *

def fname_to_datetime(fname):
    fname = fname.split('/')
    time = fname.pop().split('.')[0]
    day = fname.pop()
    month = fname.pop()
    year = fname.pop()

    hour = time[0:2]
    minute = time[2:4]
    sec = time[4:]
    
    datetime = dt(int(year),int(month),int(day),int(hour),int(minute),int(sec))
    datetime = tz.timezone('UTC').localize(datetime).astimezone(tz.timezone('US/Pacific'))

    return(datetime)

def solpix_loc(fname):
    im = rgb_read(fname)

    latlong = (39.743373,-105.179152)
    solpos = pv.solarposition.spa_python(fname_to_datetime(fname), latlong[0], latlong[1])
    zen,azi = (float(solpos.apparent_zenith), float(solpos.azimuth))

    h,w = im.shape[:2]

    deg2rad = lambda x: (x/360.)*2*np.pi

    L = int((h/2.)*(zen/180.))
    i = int( (1536/2.) + L*np.sin(deg2rad(azi)) )
    j = int( L*np.cos(deg2rad(azi)) )

    return(i,j)

#timestamp is 08:42:04
fname = '/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs/11/2019/04/06/154200.jpg'
#timestamp is 13:55:24
fname = '/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs/11/2019/04/07/205530.jpg'

print(solpix_loc(fname))
plt.imshow(rgb_read(fname))
plt.show()
