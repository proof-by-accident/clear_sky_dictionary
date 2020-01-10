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

i = 1
day = DAY_DIRS[i]
fnames = [day+ _ for _ in os.listdir(day)]
imgs = [grayscale_read_resize(_,(100,100)).flatten() for _ in fnames]

# POD 
X = np.array(imgs).T
#U,S,V = npla.svd(X)

center = lambda x: x-np.mean(x)
for i in range(X.shape[1]):
    X[:,i] = center(X[:,i])

# DMD
X1 = X[:,:-1]
X2 = X[:,1:]
U,S,V = npla.svd(X1, full_matrices=False)
A_tilde = np.dot(U.T, np.dot(X2, np.dot(V.T,np.diag(1/S))))
r = npla.matrix_rank(A_tilde)

lam, W = npla.eig(A_tilde)
lam_order = np.argsort(np.abs(lam))[::-1]
lam = lam[lam_order]
W = W[:,lam_order]

phi = np.dot(X2, np.dot(V.T,np.dot(np.diag(1/S),W)))

foo = np.sum(np.abs(phi),1)
plt.imshow(foo.reshape((100,100)))
plt.show()

omega = lambda t: np.exp( np.diag(np.log(lam))*t )
b = np.dot(np.dot(npla.inv(np.dot(phi.T,phi)),phi.T), X[:,0])
X_dmd = np.array([ np.dot(phi, np.dot(omega(t),b)) for t in range(X.shape[1])]).T

im = X[:,1].reshape((100,100))
modes = [_.reshape((100,100)) for _ in phi[:,:3].T]
gridsize = (1,4)
fig = plt.figure(figsize=(12,6))
ax1 = plt.subplot2grid(gridsize, (0,0))
ax2 = plt.subplot2grid(gridsize, (0,1))
ax3 = plt.subplot2grid(gridsize, (0,2))
ax4 = plt.subplot2grid(gridsize, (0,3))
ax1.imshow(im)
ax2.imshow(np.abs(modes[0]))
ax3.imshow(np.abs(modes[1]))
ax4.imshow(np.abs(modes[2]))
fig.tight_layout()
plt.savefig('dmd_compare.png')
plt.show()

#fig = plt.figure()
#ims = []
#for i in range(X_dmd[:,:-75].shape[1]):
#    im = plt.imshow(np.abs(X_dmd[:,i].reshape((100,100))))
#    ims.append([im])
#
#an = ani.ArtistAnimation(fig,ims,interval=50,blit=True,repeat_delay=100)
#an.save('dmd.gif',writer='ImageMagick')
#plt.show()

#full = rgb_read(fnames[45])
#x0 = y0 = 686
#delta = 200
#seg = full[x0:(x0+delta),y0:(y0+delta),:]
#plt.imshow(seg); plt.show()
