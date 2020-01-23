import os
import h5py
import argparse as ap
import cv2
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

def rgb_read(fname, progbar=None):
    """
    Reads an NxN pixel RGB image into a numpy Nd-Array
    Parms:
    ----------
    fname     location of image, string
    progbar   progressbar to be updated, progressBar (pkg: progressbar2)
    """
    rgb = cv2.imread(fname)

    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass
    
    return(rgb)


def fname_to_datetime(fname):
    year,month,day,time = fname.split('/')[-4:]
    time = time.split('.')[0]
    return(pd.to_datetime(year+month+day+time,format='%Y%m%d%H%M%S'))


def write_hdf5(images,timestamps,out):
    """ 
    Stores an array of images to HDF5.
    Parameters:
    ---------------
    images       images array, (N, 32, 32, 3) to be stored
    timestamps   timestamps array, (N, 1) to be stored
    """

    num_images = len(images)

    # Create a new HDF5 file
    outfile = h5py.File(out, "w")

    # Create a dataset in the file
    dataset = outfile.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(timestamps), h5py.h5t.STD_U8BE, data=timestamps
    )
    outfile.close()
    
if __name__=='__main__':    
    DATA_DIR = os.path.abspath('/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs/')
    
    CAM_DIRS = [os.path.join(DATA_DIR,'11'), os.path.join(DATA_DIR,'12') ]
    
    YEAR_DIRS = [ os.path.join(cd,y) for cd in CAM_DIRS for y in os.listdir(cd) ]
    MONTH_DIRS = [ os.path.join(y,m) for y in YEAR_DIRS for m in os.listdir(y) ]
    DAY_DIRS = [ os.path.join(m,d) for m in MONTH_DIRS for d in os.listdir(m) ]
    
    for day_dir in DAY_DIRS[-1:]:
        image_files = [os.path.join(day_dir,f) for f in os.listdir(day_dir)]

        images = np.stack([rgb_read(f) for f in image_files])
        timestamps = np.array([fname_to_datetime(f) for f in image_files])

        out = day_dir
        out = out.split('/')
        out[-5] = 'hdf5'
        out = '/'.join(out)
        
        write_hdf5(images,timestamps,out)
