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

import gc

from read_funcs import *

# zero-mean centers input `x`
def center(x):
    return(x - np.mean(x))

# given two filenames `f1` and `f2` this reads the saturation channels for each image and computes the hellinger distance between their histograms
def sat_hue_hell(f1,f2):
    h1 = np.histogram2d( rbd_read(f1).flatten(), sat_read2(f1).flatten() )
    h2 = np.histogram2d( rbd_read(f2).flatten(), sat_read2(f2).flatten(), bins=[h1[1],h1[2]] )

    d1 = h1[0]/np.sum(h1[0])
    d2 = h2[0]/np.sum(h2[0])  

    return( np.sqrt(np.sum((np.sqrt(d1)-np.sqrt(d2))**2))/np.sqrt(2))


def crop_image(x):
    crop = crop_mask(x.shape[0], x.shape[1])
    
    if len(x.shape) == 2:
        return(x*crop)

    else:
        ret = []
        for i in range(x.shape[2]):
            ret.append( crop*x[:,:,i] )
            
        return(np.array(ret))
    

# given two inputs `x` and `y` this computes the linear correlation coefficient between them
def corrcoef(x,y):
    return( np.mean(center(x)*center(y))/(np.std(x)*np.std(y)) )

# these compute the FFT of various color channels
def fft_grayscale(fname):
    return( np.fft.rfft2( center(cv2.imread(fname,0)) ) )

def fft_rgb(fname):
    return( np.fft.rfftn(center(rgb_read(fname))) )
    
def fft_rbr(fname):
    return( np.fft.rfft2(center(rbr_read(fname))) )

def fft_rbd(fname):
    return( np.fft.rfft2(center(rbd_read(fname))) )

def fft_sat(fname):
    return( np.fft.rfft2((sat_read(fname))) )

def fft_hue(fname):
    return( np.fft.rfft2(center(hue_read(fname))) )

# FFT comparison types
def mean_masked_spec_corr(s1,s2):
    if len(s1.shape)==2:
        corr = np.fft.irfft2(s1*s2)

    else:
        corr = np.fft.irfftn(s1*s2)

    return( np.mean(crop_image(corr)) )

def max_masked_spec_corr(s1,s2):
    if len(s1.shape)==2:
        corr = np.fft.irfft2(s1*s2)

    else:
        corr = np.fft.irfftn(s1*s2)

    return( np.max(crop_image(corr)) )

def linear_spec_corr(s1,s2):
    l1 = np.log(np.abs(s1))
    l1[np.abs(l1)==np.inf] = 0

    l2 = np.log(np.abs(s2))
    l2[np.abs(l2)==np.inf] = 0
    
    return( corrcoef( l1, l2 ) )


# compares `spec` to all FFTs in the `csl_spec` using whatever method was supplied as `comp`
# since the output of this procedure is a list, it then uses `return_func` to produce a scalar value from the list, which defaults to the max list value
def csl_comp(spec,csl_spec,comp,return_func=max):
    corrs = [comp(spec,_) for _ in csl_spec if not np.all(_ == spec)]
    return(return_func(corrs))


if __name__ == '__main__':
    stats = pd.concat(p.load(open('./pickle_dumps/full_stats.p','rb'))).reset_index()
    
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
    test_names = list(np.random.choice(stats.fname,50)) + [non_csl_clearsky]
    
    #channels=[fft_grayscale, fft_rbr, fft_rbd, fft_sat, fft_hue, fft_rgb]
    #comparisons=[max_masked_spec_corr, mean_masked_spec_corr, linear_spec_corr]

    channels=[fft_grayscale]
    comparisons=[linear_spec_corr]
    
    chan_comp_pairs = it.product(channels, comparisons)

    for fft,comp in chan_comp_pairs:    
        csl_spec = [fft(_) for _ in [csi_dict[k] for k in csi_dict.keys()] +
                    [mean_dict[k] for k in mean_dict.keys()] +
                    [sd_dict[k] for k in sd_dict.keys()]]

        test_spec = [fft(_) for _ in test_names]
    
        test_dists = [csl_comp(_, csl_spec, comp) for _ in test_spec]
        csl_dists = [csl_comp(_, csl_spec, comp) for _ in csl_spec]
        
        try:
            plt.hist(test_dists)
            # plot a vertical line at the comparison value of the "planted" clearsky image
            plt.axvline(x=test_dists[-1])
            plt.hist(csl_dists)
            plt.savefig(fft.__name__+'_'+comp.__name__+'.png')
            plt.close()

        except:
            p.dump([test_dists,csl_dists], open(fft.__name__+'_'+comp.__name__+'_error'+'.p','wb'))
