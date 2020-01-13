import os
import itertools as it

import numpy as np
import numpy.linalg as npla

import pandas as pd

import scipy as sp
from scipy import ndimage as ndi
import scipy.signal as ss

import sklearn.mixture as mix

import ot

import networkx as nx
import node2vec as nv

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

from read_funcs import *

def crop_image(x):
    crop = crop_mask(x.shape[0], x.shape[1], .95)
    
    if len(x.shape) == 2:
        ret = x*crop
        ret[ret==0] = np.nan
        return(ret)

    else:
        for i in range(x.shape[2]):
            x[...,i] = crop*x[...,i]

        x[x==0] = np.nan
        return(x)

    
def process_key(event):
    fig = event.canvas.figure

    if event.key=='down':
        pan_down(fig)

    elif event.key=='up':
        pan_up(fig)

    elif event.key=='left':
        pan_left(fig)

    elif event.key=='right':
        pan_right(fig)

    elif event.key=='z':
        zoom_out(fig)

    elif event.key=='x':
        zoom_in(fig)

    fig.canvas.draw()
        
def zoom_out(fig):
    grid_size = fig.grid_size
    grid_n = fig.grid_n
    i = fig.i
    j = fig.j
    
    if (i*(grid_size+50))<=1536 and (j*(grid_size+50))<=1536:
        grid_size += 50
        grid_n = int(rbr_example.shape[0]/grid_size) + 1    
    
    else:
        grid_size += 50
        grid_n = int(rbr_example.shape[0]/grid_size) + 1    
        i -= 1
        j -= 1
        
    fig.grid_size = grid_size
    fig.grid_n = grid_n
    fig.i = i
    fig.j=j
    
    sub_rbr = rbr_example[(i*grid_size):((i+1)*grid_size),(j*grid_size):((j+1)*grid_size)]
    sub_rgb = rgb_example[(i*grid_size):((i+1)*grid_size),(j*grid_size):((j+1)*grid_size),:]

    print(sub_rgb.shape)
    
    fig.get_axes()[0].images[0].set_data(sub_rgb)
    ax = fig.get_axes()[1]
    ax.cla()
    ax.hist(sub_rbr.flatten(), bins=np.linspace(0,1,30), density=True)


def zoom_in(fig):
    grid_size = fig.grid_size
    grid_n = fig.grid_n
    i = fig.i
    j = fig.j
    
    grid_size -= 50
    grid_n = int(rbr_example.shape[0]/grid_size) + 1

    fig.grid_size = grid_size
    fig.grid_n = grid_n
    
    sub_rbr = rbr_example[(i*grid_size):((i+1)*grid_size),(j*grid_size):((j+1)*grid_size)]
    sub_rgb = rgb_example[(i*grid_size):((i+1)*grid_size),(j*grid_size):((j+1)*grid_size),:]
    
    fig.get_axes()[0].images[0].set_data(sub_rgb)
    ax = fig.get_axes()[1]
    ax.cla()
    ax.hist(sub_rbr.flatten(), bins=np.linspace(0,1,30), density=True)
    

def pan_left(fig):
    grid_size = fig.grid_size
    grid_n = fig.grid_n
    i = fig.i
    j = fig.j

    if (j-1)>=0:
        j -= 1
        fig.j = j

    else:
        pass
    
    sub_rbr = rbr_example[(i*grid_size):((i+1)*grid_size),(j*grid_size):((j+1)*grid_size)]
    sub_rgb = rgb_example[(i*grid_size):((i+1)*grid_size),(j*grid_size):((j+1)*grid_size),:]
    
    fig.get_axes()[0].images[0].set_data(sub_rgb)
    ax = fig.get_axes()[1]
    ax.cla()
    ax.hist(sub_rbr.flatten(), bins=np.linspace(0,1,30), density=True)

def pan_right(fig):
    grid_size = fig.grid_size
    grid_n = fig.grid_n
    i = fig.i
    j = fig.j
    
    if ((j+1)*grid_size)<=1536:
        j += 1
        fig.j=j

    else:
        pass
    
    sub_rbr = rbr_example[(i*grid_size):((i+1)*grid_size),(j*grid_size):((j+1)*grid_size)]
    sub_rgb = rgb_example[(i*grid_size):((i+1)*grid_size),(j*grid_size):((j+1)*grid_size),:]
    
    fig.get_axes()[0].images[0].set_data(sub_rgb)
    ax = fig.get_axes()[1]
    ax.cla()
    ax.hist(sub_rbr.flatten(), bins=np.linspace(0,1,30), density=True)


def pan_up(fig):
    grid_size = fig.grid_size
    grid_n = fig.grid_n
    i = fig.i
    j = fig.j    

    if (i-1)>=0:
        i -= 1
        fig.i=i

    else:
        pass
    
    sub_rbr = rbr_example[(i*grid_size):((i+1)*grid_size),(j*grid_size):((j+1)*grid_size)]
    sub_rgb = rgb_example[(i*grid_size):((i+1)*grid_size),(j*grid_size):((j+1)*grid_size),:]
    
    fig.get_axes()[0].images[0].set_data(sub_rgb)
    ax = fig.get_axes()[1]
    ax.cla()
    ax.hist(sub_rbr.flatten(), bins=np.linspace(0,1,30), density=True)

    
def pan_down(fig):
    grid_size = fig.grid_size
    grid_n = fig.grid_n
    i = fig.i
    j = fig.j    

    if ((i+1)*grid_size)<=1536:
        i += 1
        fig.i = i

    else:
        pass
    
    sub_rbr = rbr_example[(i*grid_size):((i+1)*grid_size),(j*grid_size):((j+1)*grid_size)]
    sub_rgb = rgb_example[(i*grid_size):((i+1)*grid_size),(j*grid_size):((j+1)*grid_size),:]
    
    fig.get_axes()[0].images[0].set_data(sub_rgb)
    ax = fig.get_axes()[1]
    ax.cla()
    ax.hist(sub_rbr.flatten(), bins=np.linspace(0,1,30), density=True)

    
# cloudy - /home/peter/Cloud_Dynamics/Data/TSI_images/jpegs/11/2019/04/05/213200.jpg
if __name__ == '__main__':
    stats = pd.concat(p.load(open('./pickle_dumps/full_stats.p','rb'))[:-1]).reset_index()
    
    stats['clearsky_index'] = (stats['mean']/np.max(stats['mean'])) * (stats['sd']/np.max(stats['sd']))

    csi_mins = stats.groupby('hour').idxmin()['clearsky_index']
    mean_mins = stats.groupby('hour').idxmin()['mean']
    sd_mins = stats.groupby('hour').idxmin()['sd']
    
    csi_dict = {k: stats['fname'][int(csi_mins[k])] for k in csi_mins.keys()}
    mean_dict = {k: stats['fname'][int(mean_mins[k])] for k in mean_mins.keys()}
    sd_dict = {k: stats['fname'][int(sd_mins[k])] for k in sd_mins.keys()}
    
    mean_not_csi = [_ for _ in list(mean_mins) if _ not in list(csi_mins)]
    sd_not_csi = [_ for _ in list(sd_mins) if _ not in list(csi_mins)]  
    
    non_csl_clearsky = '/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs/11/2019/04/19/154600.jpg'
    test_size = 40
    #test_names = list(np.random.choice(stats.fname,test_size)) + [non_csl_clearsky]
    #p.dump(test_names,open('test_names.p','wb'))
    test_names = p.load(open('./pickle_dumps/test_names.p','rb'))
    csi_names = [csi_dict[k] for k in csi_dict.keys()]

    im_example = '/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs/12/2019/04/06/165230.jpg'
    #im_example = '/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs/11/2019/04/06/154200.jpg'
    rbr_example = crop_image(rbr_read(im_example))
    rgb_example = crop_image(rgb_read(im_example))

    grid_size = 1536
    grid_n = int(rbr_example.shape[0]/grid_size) + 1

    i = 0
    j = 0
        
    sub_rbr = rbr_example[(i*grid_size):((i+1)*grid_size),(j*grid_size):((j+1)*grid_size)]
    sub_rgb = rgb_example[(i*grid_size):((i+1)*grid_size),(j*grid_size):((j+1)*grid_size),:]

    fig_grid = (2,3)    
    fig = plt.figure(figsize=(8,6))
    fig.i = i
    fig.j = j
    fig.grid_size = grid_size
    fig.grid_n = grid_n
    
    ax = plt.subplot2grid(fig_grid,(0,0),rowspan=2,colspan=2)
    ax.imshow(sub_rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = plt.subplot2grid(fig_grid,(0,2),rowspan=2,colspan=1)
    ax.hist(sub_rbr.flatten(), bins=np.linspace(0,1,30),density=True)
    ax.set_xlabel('RBR')
    ax.set_xticks([0.,.25,.5,.75,1.])
    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()
