import sys
sys.path.append('..')

from read_funcs import *
from plot_funcs import *
from sun_position import *

############################
# Author: Peter Shaffery
# Email: pshaff2@gmail.com
# License: None
###########################

PAWAR_THRESH1 = (15e3)/49065
PAWAR_THRESH2 = .6
PAWAR_THRESH3 = 50/49065
PAWAR_THRESH4 = .06
PAWAR_THRESH5 = .04

TEST_IM = DAY_DIRS[-2] + '192300.jpg'
TEST_IM = DAY_DIRS[-2] + '222300.jpg'

def spa_calc(i_sp,j_sp,i,j):
    h,w = (1536,1536)

    y_sp = i_sp - (h/2.)
    x_sp = j_sp - (w/2.)

    y = i - (h/2.)
    x = j - (w/2.)

    spd = np.sqrt((x-x_sp)**2 + (y-y_sp)**2)
    R = h/2.

    spa = 2*np.arcsin(spd/(2*R))
    return(spa)

def iza_calc(i,j):
    h,w = (1536,1536)

    y = i - (h/2.)
    x = j - (w/2.)

    izd = np.sqrt(x**2 + y**2)
    R = h/2.

    iza = np.arcsin(izd/R)
    return(iza)

def test1(im,thresh=PAWAR_THRESH1):
    max_G = np.nanmax(im[:,:,1]/np.nanmax(im))
    return(max_G>thresh)

def test2(im,thresh=PAWAR_THRESH2):
    im = im/np.nanmax(im)
    rbr = im[:,:,0]/(im[:,:,2]+1)
    return(np.nanmean(rbr)<thresh)

def test3(im,im_prev,thresh1=PAWAR_THRESH3,thresh2=PAWAR_THRESH4):
    rcd = im[:,:,0] - im_prev[:,:,0]
    rcd_prop =  np.sum(rcd>thresh1)/(1536**2)
    return(rcd_prop<thresh2)

def test4(im,im_prev,thresh1=PAWAR_THRESH3,thresh2=PAWAR_THRESH5):
    rcd = im[:,:,0] - im_prev[:,:,0]
    rcd_prop =  np.sum(rcd>thresh1)/(1536**2)
    return(rcd_prop<thresh2)

def test_image(fname,fname_prev,mask,iza_mask):
    rgb = rgb_read(fname)
    rgb_prev = rgb_read(fname_prev)

    for i in range(3):
        rgb[:,:,i] = rgb[:,:,i]*mask
        rgb_prev[:,:,i] = rgb_prev[:,:,i]*mask

    if test1(rgb):
        pass
    else:
        return(False)

    # clear circumsolar region:
    solpix = solpix_loc(fname)
    horizon = np.empty(rgb.shape)
    horizon[:] = np.nan
    horizon_prev = np.empty(rgb.shape)
    horizon_prev[:] = np.nan

    no_horizon = np.empty(rgb.shape)
    no_horizon[:] = np.nan
    no_horizon_prev = np.empty(rgb.shape)
    no_horizon_prev[:] = np.nan

    for i in range(3):
        horizon[:,:,i] = iza_mask*rgb[:,:,i]
        no_horizon[:,:,i] = rgb[:,:,i] - np.nan_to_num(horizon[:,:,i])

        horizon_prev[:,:,i] = iza_mask*rgb_prev[:,:,i]
        no_horizon_prev[:,:,i] = rgb_prev[:,:,i] - np.nan_to_num(horizon_prev[:,:,i])

    i_range = range(np.max([0,solpix[0]-500]),np.min([1536,solpix[0]+500]))
    j_range = range(np.max([0,solpix[1]-500]),np.min([1536,solpix[1]+500]))
    for i in i_range:
        for j in j_range:
            spa = spa_calc(solpix[0],solpix[1],i,j)
            spa /= 2*np.pi
            spa *= 360
            if spa < 15:
                rgb[i,j,0] = np.nan
                rgb[i,j,1] = np.nan
                rgb[i,j,2] = np.nan
            else:
                pass


    if test2(rgb):
        pass
    else:
        return(False)

    if test3(no_horizon,no_horizon_prev):
        pass
    else:
        return(False)

    if test4(horizon,horizon_prev):
        pass
    else:
        return(False)

    return(True)


# when csd_make.py is run (either from the python REPL or from the command line it will create a CSD (in the form of a python `dictionary` object and save it to a pickle file
if __name__=='__main__':
    # Compiling the CSD takes some time, so the script displays progress using an in-line progress bar
    # Max lengths of progress bar
    prog_len = len([f for dd in DAY_DIRS for f in os.listdir(dd)]) + len(DAY_DIRS)

    # call crop mask
    h,w = (1536,1536)
    mask = crop_mask(h,w,crop_factor=.9).astype('float')
    mask[mask==0] = np.nan

    iza_mask = 1.*np.ones((h,w))
    for i in range(h):
        for j in range(w):
            iza = iza_calc(i,j)
            iza /= 2*np.pi
            iza *= 360
            if iza > 80:
                iza_mask[i,j] = 1.
            else:
                iza_mask[i,j] = np.nan

    csd = []
    # for each day in DAY_DIRS compute image statistics used to select CSD
    with progressbar.ProgressBar(max_value=prog_len) as progbar:
        for dd in DAY_DIRS:
            for f_prev,f in zip(os.listdir(dd)[:-1],os.listdir(dd)[1:]):
                fname = os.path.join(dd,f)
                fname_prev = os.path.join(dd,f_prev)
                csd_element = test_image(fname,fname_prev,mask,iza_mask)
                if csd_element:
                    csd.append(fname)

                else:
                    pass

                progbar.update(progbar.value+1)

    # save CSD in a pickle file
    with open(os.path.join(PICKLE_DIR,'pawar_clearsky_dict.p'),'wb') as f:
        p.dump(csd,f)
