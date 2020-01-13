from datetime import datetime as dt
from datetime import timezone as tz
from datetime import timedelta as td
import pytz
import pvlib as pv
import sys
sys.path.insert(1,'..')
exec(open('./mrDMD.py').read())
from read_funcs import *
import scipy.io

def phi_filter(im, phi_flati,nx,ny):
    phi = phi_flat.reshape((nx,ny),order='F')
    return(im - im*phi)

def phi_flat_plot(phi,nx,ny):
    plt.imshow(np.abs(phi.reshape((nx,ny),order='F')))
    plt.show()

if __name__ == '__main__':
    IMAGE_DIR = './example_images/some_cloud/'
    im_fnames = [IMAGE_DIR+_ for _ in os.listdir(IMAGE_DIR)]

    nx = ny = 1536
    n = nx*ny
    #Xraw = np.array([grayscale_read_matlab(f)[:,:,0].flatten('F') for f in im_fnames]).T
    #Xraw -= Xraw.mean(0)
    Xraw = scipy.io.loadmat('Xraw.mat')['Xraw']

    nfiles = len(im_fnames)

    L = 4
    r = 10
    dt = 30
    T = nfiles*dt

    print('running mrDMD...')
    mrdmd = mrDMD(Xraw,dt,r,2,L)
    print('...done')

    prev_level = [mrdmd]
    for l in range(L-2):
        curr_level = []
        for ch in prev_level:
            if len(ch['children'])>0:
                curr_level += ch['children']
            else:
                break
        prev_level = curr_level

    sun_modes = [np.abs(ch['Phi'][:n,0].reshape((nx,ny),order='F')) for ch in curr_level]
    #cloud_modes = [np.abs(ch['Phi'][:n,1:].reshape((nx,ny),order='F')) for ch in curr_level]
    modes = [np.abs(_['Phi']) for _ in curr_level]

#    omega = curr_level[-1]['omega']
#    b = curr_level[-1]['b']
#    phi = curr_level[-1]['Phi']
#    foo = np.dot(phi,np.dot(np.diag(np.exp(omega)),b))
#    bar = np.abs(foo[:,0].reshape((nx,ny), order='F'))
#    fig,ax = plt.subplots(2,1)
#    ax[0].imshow(test_orig - test_orig*bar/np.max(bar))
#    ax[1].imshow(test_orig)
#    plt.show()
#


def solpix_loc(fname):
    latlong = (39.743057,-105.178950)
    solpos = pv.solarposition.spa_python(fname_to_time(fname), latlong[0], latlong[1])
    zen,azi = (float(solpos.apparent_zenith), float(solpos.azimuth))

    h,w = (1536,1536)

    deg2rad = lambda x: (x/360.)*2*np.pi

    L = int((h/2.)*(zen/90.))
    i = int( (h/2.) - L*np.cos(deg2rad(azi)) )
    j = int( (w/2.) - L*np.sin(deg2rad(azi)) )

    return(i,j)

def fname_to_time(fname):
    year,mon,day,time = fname.split('/')[-4:]
    time = time.split('.')[0]
    tstamp = dt(*[int(_) for _ in [year,mon,day,time[:2],time[2:4],time[4:]]])
    return(tstamp.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('US/Mountain')))# + td(hours=1))

if __name__=='alt':
    #timestamp is 08:42:04
    #fname = '/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs/11/2019/04/06/154200.jpg'
    #timestamp is 13:55:24
    #fname = '/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs/11/2019/04/07/205530.jpg'
    dname = '/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs/11/2019/04/15/'

    test_files = [dname+_ for _ in os.listdir(dname)[1:1500:60]]

    fig,axs = plt.subplots(5,5)
    axs = axs.reshape((25,))
    
    nx = ny = 1536
    for ax,fn in zip(axs,test_files):
        loc = solpix_loc(fn)
        solpix = np.zeros((nx,ny))
        solpix[(loc[0]-50):(loc[0]+50),(loc[1]-50):(loc[1]+50)] = 1
        ax.imshow(rbr_read(fn)-solpix)

    plt.show()
