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
from matplotlib.animation import FFMpegWriter
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D

import time as t
import pickle as p
import progressbar

import gc

from read_funcs import *

DATA_DIR = os.path.abspath('/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs')
CAM_DIRS = [os.path.join(DATA_DIR,'11'), os.path.join(DATA_DIR,'12') ]
YEAR_DIRS = [ os.path.join(cd,y) for cd in CAM_DIRS for y in os.listdir(cd) ]
MONTH_DIRS = [ os.path.join(y,m) for y in YEAR_DIRS for m in os.listdir(y) ]
DAY_DIRS = [ os.path.join(m,d)+'/' for m in MONTH_DIRS for d in os.listdir(m) ]
GIF_DIR = '/home/peter/Cloud_Dynamics/Python/gifs'

test_fname = DAY_DIRS[0] + '/' + os.listdir(DAY_DIRS[0])[0]
clearsky_dict = p.load(open('./pickle_dumps/clearsky_dict.p','rb'))

def normalize(x):
    return(x/np.max(np.abs(x)))

# FUNCTIONS TO ANIMATE IMAGE STATISTICS
def anim_init():
    l1.set_data([],[])
    l2.set_data([],[])
    l3.set_data([],[])

    return([l1, l2, l3])


def crop(img,mask):
    if len(img.shape)<3:
        return(img*mask)

    else:
        for i in range(img.shape[-1]):
            img[:,:,i] = img[:,:,i]*mask

        return(img)

def gif_make(frames, out_name):
    red_frames, green_frames, blue_frames, hue_frames, sat_frames, val_frames, img_frames = frames

    gridsize = (2,4)
    fig = plt.figure(figsize=(12,6))

    ax1 = plt.subplot2grid(gridsize, (0,0), colspan=2, rowspan=2)

    ax2 = plt.subplot2grid(gridsize, (0,2))
    ax2.set_xlabel('Saturation')
    ax2.set_ylabel('Hue')

    ax3 = plt.subplot2grid(gridsize, (1,2))
    ax3.set_xlabel('Saturation')
    ax3.set_ylabel('Value')

    ax4 = plt.subplot2grid(gridsize, (0,3))
    ax4.set_xlabel('Blue')
    ax4.set_ylabel('Red')

    ax5 = plt.subplot2grid(gridsize, (1,3))
    ax5.set_xlabel('Blue')
    ax5.set_ylabel('Green')

    p1 = ax1.imshow(img_frames[0])
    p2 = ax2.scatter([],[], s=1)
    p3 = ax3.scatter([],[], s=1)
    p4 = ax4.scatter([],[], s=1)
    p5 = ax5.scatter([],[], s=1)

    plot_objs = (p1,p2,p3,p4,p5)

    plot_dat = (img_frames,
                red_frames,
                green_frames,
                blue_frames,
                hue_frames,
                sat_frames,
                val_frames)

    fig.tight_layout()
    anim = ani.FuncAnimation(fig, anim_iter,
                             fargs = (plot_objs,plot_dat),
                             frames=len(img_frames),
                             interval=20,
                             blit=True)
    anim.save(out_name, writer='imagemagick', fps=20)
    plt.close('all')

def anim_iter(i, plot_objs, plot_dat):
    p1,p2,p3,p4,p5 = plot_objs
    img_frames,red_frames,green_frames,blue_frames,hue_frames,sat_frames,val_frames = plot_dat

    p1.set_data(img_frames[i])

    c = np.array([red_frames[i],green_frames[i],blue_frames[i]]).T

    p2.set_offsets(np.array([sat_frames[i], hue_frames[i]]).T)

    p3.set_offsets(np.array([sat_frames[i], val_frames[i]]).T)

    p4.set_offsets(np.array([blue_frames[i], red_frames[i]]).T)

    p5.set_offsets(np.array([blue_frames[i], green_frames[i]]).T)

    return([p1,p2,p3,p4,p5])


## animate_dat
# Function to animate various TSI statistics over the course of a given day. For speed reasons this function should be used in conjuction with the combine_gifs.sh script in the `bash` code directory. Because the TSIs are so high-res, it is faster and more memory efficient to generate many short gifs and combine them with ImageMagick than it is to write one large GIF.
# _Arguments_
# i: day to animate
# gif_length: number of frames in one GIF, defaults to 50

def animate_day(i,gif_length=50):
    day = DAY_DIRS[i]
    fnames = [day + _ for _ in os.listdir(day) ]
    fnames = [f for f,i in zip(fnames,range(len(fnames))) if i%2 == 0]

    K = int(np.floor(len(fnames)/float(gif_length)))
    for k in range(K):
        img_frames = [ rgb_read_resize(f,(512,512)) for f in fnames[(50*k):(50*(k+1))] ]
        h,w,_ = img_frames[0].shape
        mask = crop_mask(h,w)

        img_frames = [crop(img,mask) for img in img_frames]

        red_frames = [(img[:,:,0]).flatten() for img in img_frames]
        green_frames = [(img[:,:,1]).flatten() for img in img_frames]
        blue_frames = [(img[:,:,2]).flatten() for img in img_frames]

        hsv_frames = [crop(hsv_read_resize(f,(512,512)), mask) for f in fnames[(50*k):(50*(k+1))]]
        hue_frames = [img[:,:,0].flatten() for img in hsv_frames]
        sat_frames = [img[:,:,1].flatten() for img in hsv_frames]
        val_frames = [img[:,:,2].flatten() for img in hsv_frames]

        frames = (red_frames,
                  green_frames,
                  blue_frames,
                  hue_frames,
                  sat_frames,
                  val_frames,
                  img_frames)

        gif_make(frames, GIF_DIR+'/{}_day_{}.gif'.format(k,i))
        print(k)
