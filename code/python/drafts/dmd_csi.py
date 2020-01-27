from dmd import *
from read_funcs import *
from plot_funcs import *

def scale(x):
    return((x-np.mean(x))/np.sqrt(np.var(x)))


def logit(x,a=1,b=0):
    return(1/(1+np.exp(-a*x -b)))


def del_T(t1,t2):
    t1 = str_to_tstamp(t1)
    t2 = str_to_tstamp(t2)

    dt = t1 - t2
    d = dt.days
    s = dt.seconds
    return(60.*24.*d + s/60.)


def fname_to_time(fname):
    year,mon,day,time = fname.split('/')[-4:]
    time = time.split('.')[0]
    tstamp = dt(*[int(_) for _ in [year,mon,day,time[:2],time[2:4],time[4:]]])
    return(tstamp.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('US/Mountain')) + td(hours=1))


def fname_to_time2(fname):
    year,mon,day,time = fname.split('/')[-4:]
    time = time.split('.')[0]
    tstamp = dt(*[int(_) for _ in [year,mon,day,time[:2],time[2:4],time[4:]]])
    return(tstamp.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('US/Mountain')))


def min_round(t):
    s = t.second
    t -= td(seconds=s)
    return(t)


def str_to_tstamp(t):
    time = t[:-6]
    tz = t[-5:].split(':')
    tz = tz[0] + tz[1]
    return(dt.strptime(time+' -'+tz, "%Y-%m-%d %H:%M:%S %z"))


def rsqr(x,y):
    lin = np.polyfit(x,y,1)
    y_hat = x*lin[0] + lin[1]
    ssr = np.sum((y - y_hat)**2)
    sst = np.sum((y - np.mean(y))**2)
    return(1 - (ssr/sst))


def mode_sub(img,mode):
    mode = mode.reshape((1536,1536),order='F')
    mode = mode/mode.max()
    return(img - img*mode)


def phi_filter2(img,Phi):
    sub_modes = Phi[:,1:]
    for k in range(sub_modes.shape[1]):
        img = mode_sub(img,sub_modes[:,k])
    mode = Phi[:,0]
    mode = mode.reshape((1536,1536),order='F')
    mode = mode/mode.max()
    return(img*mode)


def phi_filter(img,Phi):
    mode = Phi[:,0]
    mode = mode.reshape((1536,1536),order='F')
    mode = mode/mode.max()
    return(img*mode)


def implot3d(img):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    dim = img.shape[0]
    X,Y = np.meshgrid(np.linspace(0,dim,dim),np.linspace(0,dim,dim))
    ax.plot_wireframe(X,Y,img)
    plt.show()


def imgrad_norm(img):
    grad = np.gradient(img)
    return( np.sqrt(grad[0]**2 + grad[1]**2))

def mode_crop(img, thresh=.5, step=5):
    mode_x, mode_y = np.where(img==img.max())
    p1 = [int(np.min(mode_x)),int(np.min(mode_y))]
    p2 = [int(np.max(mode_x)),int(np.max(mode_y))]
    img_crop = img[mode_x,mode_y]
    denom = img.sum()
    while (img_crop.sum()/denom) < thresh:
        p1[0] -= step
        p1[1] -= step
        p2[0] += step
        p2[1] += step
        img_crop = img[p1[0]:p2[0],p1[1]:p2[1]]

    #grad = imgrad_norm(img_crop)
    not_flat = 1# - (grad==0)

    return(not_flat*img_crop)

def image_stats(im_list,fns):
    mean_rbr = []
    sd_rbr = []
    med_rbr = []
    max_rbr = []
    busyness = []
    times = []

    cyc = np.floor(len(test_fnames)/len(sun_modes))
    for i in range(len(im_list)):
        img = im_list[i]
        mean_rbr.append(np.mean(img))
        sd_rbr.append(np.sqrt(np.var(img)))
        med_rbr.append(np.median(img))
        max_rbr.append(np.max(img))
        busyness.append(np.mean(imgrad_norm(img)))
        times.append(fname_to_time(fns[i]))

    csi_rbr = np.array(mean_rbr)*np.array(sd_rbr)

    im_stats = pd.DataFrame({'mean_rbr': mean_rbr,
                             'sd_rbr': sd_rbr,
                             'med_rbr': med_rbr,
                             'max_rbr': max_rbr,
                             'my_csi': csi_rbr,
                             'busy': busyness,
                             'time': [str(_.replace(second=0)) for _ in times]})
    return(im_stats)

if __name__=='__main__':
    TEST_IMAGE_DIR = './example_images/some_cloud/'
    CSI_DATA_DIR = os.path.abspath('/home/peter/Cloud_Dynamics/Data/clearsky_index')
    TEST_DAY_DIR ='/home/peter/Cloud_Dynamics/Data/TSI_images/jpegs/11/2019/04/15'


    # read in DMD test data
    im_fnames = [TEST_IMAGE_DIR+_ for _ in os.listdir(TEST_IMAGE_DIR)]
    nx = ny = 1536
    n = nx*ny
    Xraw = np.array([grayscale_read_matlab(f)[:,:,0].flatten('F') for f in im_fnames]).T
    Xraw -= Xraw.mean(0)

    nfiles = len(im_fnames)

    L = 4
    r = 10
    dt = 30
    T = nfiles*dt
    print('running mrDMD...')
    mrdmd = mrDMD(Xraw,dt,r,2,L)
    print('...done')

    # unpack DMD modes
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
    cloud_modes = [np.abs(ch['Phi'][:n,1:].reshape((nx,ny),order='F')) for ch in curr_level]
    modes = [np.abs(_['Phi']) for _ in curr_level]

    # read in true CSI values
    try:
        csi = pd.read_csv(CSI_DATA_DIR+'/csi.csv')

    except:
        comps = pd.read_csv(CSI_DATA_DIR+'/csi_comps.csv')
        comps.columns = ['date','mst','ghi','ghi1','ghi2','getr','dni','detr']
        csi = pd.DataFrame({'gcsi': comps.ghi/comps.getr, 'dcsi': comps.dni/comps.detr})
        tlist = [0]*csi.shape[0]
        for i in range(csi.shape[0]):
            date = comps.date[i].split('/')
            time = comps.mst[i].split(':')
            tstamp = dt(*[int(_) for _ in [date[2],date[0],date[1],time[0],time[1]]])
            tlist[i] = tstamp.replace(tzinfo=pytz.timezone('US/Mountain'))

        csi.insert(1,'time',tlist)
        csi.to_csv(CSI_DATA_DIR+'/csi.csv', index=False)

    # read and process test images
    test_fnames = []
    for f in os.listdir(TEST_IMAGE_DIR):
        f = f.split('_')[0][-6:]
        test_fnames.append(TEST_DAY_DIR+'/'+f+'.jpg')

    cyc = np.floor(len(test_fnames)/len(sun_modes))

    # create lists of differently processed images
    solpix_ims = []
    solpix_ims_mode_filtered1 = []
    solpix_ims_mode_filtered2 = []

    mask = crop_mask(1536,1536)
    for i in range(len(test_fnames)):
        print(i/float(len(test_fnames)))
        # get DMD modes
        #try:
        #    Phi = np.abs(curr_level[int(np.floor(i/cyc))]['Phi'][:(1536**2),:])
        #except IndexError:
        #    Phi = np.abs(curr_level[-1]['Phi'][:(1536**2),:])
        #Phi = np.abs(mrdmd['Phi'])
        #Phi = Phi[:,[2,0,1,3,4,5]]


        fn = test_fnames[i]
        img = mask*rbr_read(fn)

        # crop image to contain mainly sun pixels
        solpix_ims.append(solpix_crop(img,fn))
        # use sun DMD modes to highlight sun
        #solpix_ims_mode_filtered1.append(solpix_crop(phi_filter(img,Phi),fn))
        # filter out non-sun DMD modes
        #solpix_ims_mode_filtered2.append(solpix_crop(phi_filter2(img,Phi),fn))


    ims = image_stats(solpix_ims, test_fnames)
    ims_filtered1 = image_stats(solpix_ims_mode_filtered1, test_fnames)
    ims_filtered2 = image_stats(solpix_ims_mode_filtered2, test_fnames)

    csi_comp = ims.merge(csi,on='time')
    perfs = {'mean_rbr': rsqr(csi_comp.mean_rbr,csi_comp.dcsi),
             'sd_rbr': rsqr(csi_comp.sd_rbr,csi_comp.dcsi),
             'med_rbr': rsqr(csi_comp.med_rbr,csi_comp.dcsi),
             'max_rbr': rsqr(csi_comp.max_rbr,csi_comp.dcsi),
             'my_csi': rsqr(csi_comp.my_csi,csi_comp.dcsi),
             'busy': rsqr(csi_comp.busy,csi_comp.dcsi)}
    fig,axs = plt.subplots(2,3,figsize=(10,8),constrained_layout=True)
    fig.suptitle('Unfiltered, Cropped',y=1)
    axs[0,0].scatter(csi_comp.mean_rbr, csi_comp.dcsi)
    axs[0,0].set_xlabel('Mean RBR')
    axs[0,0].set_title(np.round(perfs['mean_rbr'],2))
    
    axs[0,1].scatter(csi_comp.sd_rbr, csi_comp.dcsi)
    axs[0,1].set_xlabel('SD RBR')
    axs[0,1].set_title(np.round(perfs['sd_rbr'],2))

    axs[0,2].scatter(csi_comp.med_rbr, csi_comp.dcsi)
    axs[0,2].set_xlabel('Med RBR')
    axs[0,2].set_title(np.round(perfs['med_rbr'],2))
    
    axs[1,0].scatter(csi_comp.max_rbr, csi_comp.dcsi)
    axs[1,0].set_xlabel('Max RBR')
    axs[1,0].set_title(np.round(perfs['max_rbr'],2))
    
    axs[1,1].scatter(csi_comp.my_csi, csi_comp.dcsi)
    axs[1,1].set_xlabel('My CSI')
    axs[1,1].set_title(np.round(perfs['my_csi'],2))
    
    axs[1,2].scatter(csi_comp.busy, csi_comp.dcsi)
    axs[1,2].set_xlabel('Mean RBR Gradient')
    axs[1,2].set_title(np.round(perfs['busy'],2))
    fig.tight_layout()
    fig.savefig('perfs.png')
    plt.close()
    
    assert False
    csi_comp = ims_filtered1.merge(csi,on='time')
    perfs_filtered1 = {'mean_rbr': rsqr(csi_comp.mean_rbr,csi_comp.dcsi),
             'sd_rbr': rsqr(csi_comp.sd_rbr,csi_comp.dcsi),
             'med_rbr': rsqr(csi_comp.med_rbr,csi_comp.dcsi),
             'max_rbr': rsqr(csi_comp.max_rbr,csi_comp.dcsi),
             'my_csi': rsqr(csi_comp.my_csi,csi_comp.dcsi),
             'busy': rsqr(csi_comp.busy,csi_comp.dcsi)}
    fig,axs = plt.subplots(2,3,figsize=(10,8),constrained_layout=True)
    fig.suptitle('Sun Mode Masked, Cropped',y=1)
    axs[0,0].scatter(csi_comp.mean_rbr, csi_comp.dcsi)
    axs[0,0].set_xlabel('Mean RBR')
    axs[0,0].set_title(np.round(perfs_filtered1['mean_rbr'],2))
    
    axs[0,1].scatter(csi_comp.sd_rbr, csi_comp.dcsi)
    axs[0,1].set_xlabel('SD RBR')
    axs[0,1].set_title(np.round(perfs_filtered1['sd_rbr'],2))

    axs[0,2].scatter(csi_comp.med_rbr, csi_comp.dcsi)
    axs[0,2].set_xlabel('Med RBR')
    axs[0,2].set_title(np.round(perfs_filtered1['med_rbr'],2))
    
    axs[1,0].scatter(csi_comp.max_rbr, csi_comp.dcsi)
    axs[1,0].set_xlabel('Max RBR')
    axs[1,0].set_title(np.round(perfs_filtered1['max_rbr'],2))
    
    axs[1,1].scatter(csi_comp.my_csi, csi_comp.dcsi)
    axs[1,1].set_xlabel('My CSI')
    axs[1,1].set_title(np.round(perfs_filtered1['my_csi'],2))
    
    axs[1,2].scatter(csi_comp.busy, csi_comp.dcsi)
    axs[1,2].set_xlabel('Mean RBR Gradient')
    axs[1,2].set_title(np.round(perfs_filtered1['busy'],2))
    fig.tight_layout()
    fig.savefig('perfs_filt1.png')
    plt.close()
    
    csi_comp = ims_filtered2.merge(csi,on='time')
    perfs_filtered2 = {'mean_rbr': rsqr(csi_comp.mean_rbr,csi_comp.dcsi),
             'sd_rbr': rsqr(csi_comp.sd_rbr,csi_comp.dcsi),
             'med_rbr': rsqr(csi_comp.med_rbr,csi_comp.dcsi),
             'max_rbr': rsqr(csi_comp.max_rbr,csi_comp.dcsi),
             'my_csi': rsqr(csi_comp.my_csi,csi_comp.dcsi),
             'busy': rsqr(csi_comp.busy,csi_comp.dcsi)}
    fig,axs = plt.subplots(2,3, figsize=(10,8),constrained_layout=True)
    fig.suptitle('Cloud Modes Removed, Cropped',y=1)
    axs[0,0].scatter(csi_comp.mean_rbr, csi_comp.dcsi)
    axs[0,0].set_xlabel('Mean RBR')
    axs[0,0].set_title(np.round(perfs_filtered2['mean_rbr'],2))
    
    axs[0,1].scatter(csi_comp.sd_rbr, csi_comp.dcsi)
    axs[0,1].set_xlabel('SD RBR')
    axs[0,1].set_title(np.round(perfs_filtered2['sd_rbr'],2))

    axs[0,2].scatter(csi_comp.med_rbr, csi_comp.dcsi)
    axs[0,2].set_xlabel('Med RBR')
    axs[0,2].set_title(np.round(perfs_filtered2['med_rbr'],2))
    
    axs[1,0].scatter(csi_comp.max_rbr, csi_comp.dcsi)
    axs[1,0].set_xlabel('Max RBR')
    axs[1,0].set_title(np.round(perfs_filtered2['max_rbr'],2))
    
    axs[1,1].scatter(csi_comp.my_csi, csi_comp.dcsi)
    axs[1,1].set_xlabel('My CSI')
    axs[1,1].set_title(np.round(perfs_filtered2['my_csi'],2))
    
    axs[1,2].scatter(csi_comp.busy, csi_comp.dcsi)
    axs[1,2].set_xlabel('Mean RBR Gradient')
    axs[1,2].set_title(np.round(perfs_filtered2['busy'],2))
    fig.tight_layout()
    fig.savefig('perfs_filt2.png')
    plt.close()
