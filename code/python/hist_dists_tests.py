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


def rbd_hell(f1,f2,nbins=20):
    h1 = np.histogram( rbd_read(f1).flatten(), bins=nbins )
    h2 = np.histogram( rbd_read(f2).flatten(), bins=h1[1] )
    
    d1 = h1[0]/np.sum(h1[0])
    d2 = h2[0]/np.sum(h2[0])  

    return( np.sqrt(np.sum((np.sqrt(d1)-np.sqrt(d2))**2))/np.sqrt(2))


def rbd_kl(f1,f2):
    h1 = np.histogram( rbd_read(f1).flatten() )
    h2 = np.histogram( rbd_read(f2).flatten(), bins=h1[1] )
    
    d1 = h1[0]/np.sum(h1[0])
    d2 = h2[0]/np.sum(h2[0])  

    return( np.dot(d1 * np.log(d1/d2)) )

def rbd_hist_inter(f1,f2,nbins=20):
    h1 = np.histogram( crop_image(rbd_read(f1)).flatten(), bins=nbins )
    h2 = np.histogram( crop_image(rbd_read(f2)).flatten(), bins=h1[1] )
    
    d1 = h1[0]/np.sum(h1[0])
    d2 = h2[0]/np.sum(h2[0])

    return( np.sum(d1,d2) )


def rbd_wasserstein(f1,f2,nbins=30):
    bins = np.linspace(-1,1,nbins)
    h1 = np.histogram( rbd_read(f1).flatten(), bins=bins )
    h2 = np.histogram( rbd_read(f2).flatten(), bins=bins )
    
    d1 = h1[0]/np.sum(h1[0])
    d2 = h2[0]/np.sum(h2[0]) 

    return(sp.stats.wasserstein_distance(d1,d2))


def sat_wasserstein(f1,f2,nbins=30):
    bins = np.linspace(0,1,nbins)
    h1 = np.histogram( sat_read2(f1).flatten(), bins=bins )
    h2 = np.histogram( sat_read2(f2).flatten(), bins=bins )
    
    d1 = h1[0]/np.sum(h1[0])
    d2 = h2[0]/np.sum(h2[0]) 

    return(sp.stats.wasserstein_distance(d1,d2))


def im_wasserstein(im1,im2,nbins=30):
    bins = np.linspace(0,1,nbins)
    h1 = np.histogram( im1.flatten(), bins=bins )
    h2 = np.histogram( im2.flatten(), bins=bins )
    
    d1 = h1[0]/np.sum(h1[0])
    d2 = h2[0]/np.sum(h2[0]) 

    return(sp.stats.wasserstein_distance(d1,d2))


def im_hellinger(im1,im2,nbins=30):
    bins = np.linspace(0,1,nbins)
    h1 = np.histogram( im1.flatten(), bins=bins )
    h2 = np.histogram( im2.flatten(), bins=bins )
    
    d1 = h1[0]/np.sum(h1[0])
    d2 = h2[0]/np.sum(h2[0])  

    return( np.sqrt(np.sum((np.sqrt(d1)-np.sqrt(d2))**2))/np.sqrt(2))


def gauss_wasserstein(m1,v1,m2,v2):
    term1 = np.abs(m1-m2)**2
    term2 = v1 + v2 - 2*np.sqrt(v1*v2)
    return(term1+term2)


def rbd_wasserstein_approx1(f1,f2):
    im1 = rbd_read(f1).flatten()
    im2 = rbd_read(f2).flatten()

    return(gauss_wasserstein(np.mean(im1),np.var(im1), np.mean(im2), np.var(im2)))

def rbd_wasserstein_approx2(f1,f2):
    im1 = rbd_read(f1).flatten().reshape(-1,1)
    im2 = rbd_read(f2).flatten().reshape(-1,1)

    gauss1 = mix.GaussianMixture(2).set_params(tol=1e-1).fit(im1)
    gauss2 = mix.GaussianMixture(2).set_params(tol=1e-1).fit(im2)

    m11,m12 = gauss1.means_.flatten()
    v11,v12 = gauss1.covariances_.flatten()
    p11,p12 = gauss1.weights_.flatten()

    m21,m22 = gauss2.means_.flatten()
    v21,v22 = gauss2.covariances_.flatten()
    p21,p22 = gauss2.weights_.flatten()
    
    d1 = np.array([p11,p12,0,0])
    d2 = np.array([0,0,p21,p22])

    m = np.array([m11,m12,m21,m22])
    v = np.array([v11,v12,v21,v22])
    
    weight_matrix = np.zeros((len(d1),len(d2)))
    for i in range(len(d1)):
        for j in range(len(d1)):
            weight_matrix[i,j] = gauss_wasserstein(m[i],v[i],m[j],v[j])

    return(ot.emd(d1,d2,weight_matrix))
            
    
def crop_image(x):
    crop = crop_mask(x.shape[0], x.shape[1])
    
    if len(x.shape) == 2:
        return(x*crop)

    else:
        ret = []
        for i in range(x.shape[2]):
            ret.append( crop*x[:,:,i] )
            
        return(np.array(ret))
    

# compares `spec` to all FFTs in the `csl_spec` using whatever method was supplied as `comp`
# since the output of this procedure is a list, it then uses `return_func` to produce a scalar value from the list, which defaults to the max list value
def csl_comp(spec,csl_spec,comp,return_func=max):
    corrs = [comp(spec,_) for _ in csl_spec if not np.all(_ == spec)]
    return(return_func(corrs))


def csl_dist_comp(fname,csl_fnames,dist,dist_summ=np.max):
    dists = [dist(fname,_) for _ in csl_fnames]
    return(dist_summ(dists))


def n2v_comp(csl_names,test_names,dist):
    N = len(csl_names)
    M = len(test_names)
    adj_mat = np.zeros((N,M))

    for i in range(N):
        for j in range(M):
            adj_mat[i,j] = dist(csl_names[i],test_names[j])
        
            g = nx.from_numpy.matrix(adj_mat)


def rgb_row_plot(i,fname):
    rgb = rgb_read(fname)

    r = rgb[i,:,0]
    g = rgb[i,:,1]
    b = rgb[i,:,2]

    plt.plot(r,c='red')
    plt.plot(g,c='green')
    plt.plot(b,c='blue')

    plt.show()


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

    assert False

    csi_ims = [rbr_read(_) for _ in csi_names]
    test_ims = [rbr_read(_) for _ in test_names]

    dist_func = im_hellinger

    rbr_hell_csi = np.array([csl_dist_comp(_,csi_ims,dist_func,lambda x: x) for _ in csi_ims])
    rbr_hell_test = np.array([csl_dist_comp(_,csi_ims,dist_func,lambda x: x) for _ in test_ims])

    dist_func = im_wasserstein
   
    rbr_emd_csi = np.array([csl_dist_comp(_,csi_ims,dist_func,lambda x: x) for _ in csi_ims])
    rbr_emd_test = np.array([csl_dist_comp(_,csi_ims,dist_func,lambda x: x) for _ in test_ims])
    
    csi_ims = [sat_read2(_) for _ in csi_names]
    test_ims = [sat_read2(_) for _ in test_names]

    dist_func = im_hellinger

    sat_hell_csi = np.array([csl_dist_comp(_,csi_ims,dist_func,lambda x: x) for _ in csi_ims])
    sat_hell_test = np.array([csl_dist_comp(_,csi_ims,dist_func,lambda x: x) for _ in test_ims])
    
    sat_emd_csi = np.array([csl_dist_comp(_,csi_ims,dist_func,lambda x: x) for _ in csi_ims])
    sat_emd_test = np.array([csl_dist_comp(_,csi_ims,dist_func,lambda x: x) for _ in test_ims])

    
    gridsize = (1,3)
    fig = plt.figure(figsize=(12,6))
    
    ax1 = plt.subplot2grid(gridsize, (0,0))
    ax1.imshow(rbr_hell_test)
    ax1.set_title('RBR (Hellinger)')
    ax1.set_xlabel('CSD')
    ax1.set_ylabel('Test Images')
    
    ax2 = plt.subplot2grid(gridsize, (0,2))
    ax2.set_title('Sat (EMD)')
    ax2.imshow(sat_emd_test)
    ax2.set_xlabel('CSD')
    ax2.set_ylabel('Test Images')
    
    ax3 = plt.subplot2grid(gridsize, (0,1))
    ax3.set_title('RBR (EMD)')
    ax3.imshow(rbr_emd_test)
    ax3.set_xlabel('CSD')
    ax3.set_ylabel('Test Images')
    
    plt.tight_layout()
    plt.show()
