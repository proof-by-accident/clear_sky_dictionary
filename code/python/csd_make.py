from read_funcs import *
from plot_funcs import *
from sun_position import *

############################
# Script to read all TSIs in DATA_DIR and construct a clearsky dictionary (CSD)
# Core algorithm described at overleaf.com/1577517535shnfkqjdbrcm
############################
# Author: Peter Shaffery
# Email: pshaff2@gmail.com
# License: None
###########################

# `day2stats` calculates several statistics for all images taken in a given day of the TSI data. Returns a Pandas dataframe (df) where each row corresponds to one TSI's stats
# _Arguments_
# day_dir: the directory corresponding to the day of images for which to compute stats
# mask: the zero mask used to crop image
# progbar: defaults to `None`, otherwise increments the provided progress bar by one
def day2stats(day_dir, mask, progbar=None):
    
    # get a list of filenames corresponding to TSIs in day `day_dir`
    image_files = [os.path.join(day_dir,f) for f in os.listdir(day_dir)]

    # preallocating variable names
    img = None
    stats = [[None]*6]*len(image_files)

    for i in range(len(image_files)):
        f = image_files[i]
        hr = fname_to_time(f).hour #fname_to_time is found in `sun_position.py`

        img = mask*rbr_read(f,progbar) #this sets all image content outside of the fisheye lens to 0

        # compile row of day-level image statistics
        stats[i] = [f,hr,np.mean(img),np.median(img),np.max(img),np.std(img)]


    # convert list of rows to Pandas DataFrame
    stats = pd.DataFrame(stats)
    stats.columns = ['fname','hour','mean','median','max','sd']
    stats['clearsky_index'] = stats['mean']*stats['sd']

    # save DataFrame to a serialized format (pickle)
    with open(os.path.join(PICKLE_DIR,day_dir.split('/')[-4]+'_'+day_dir.split('/')[-1].split('.')[0]+'.p'),'wb') as f:
        p.dump(stats,f)

    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass

    return(stats)

# when csd_make.py is run (either from the python REPL or from the command line it will create a CSD (in the form of a python `dictionary` object and save it to a pickle file
if __name__=='__main__':
    # Compiling the CSD takes some time, so the script displays progress using an in-line progress bar
    # Max lengths of progress bar
    prog_len = len([f for dd in DAY_DIRS for f in os.listdir(dd)]) + len(DAY_DIRS)

    # call crop mask
    h,w = (1536,1536)
    mask = crop_mask(h,w)

    # for each day in DAY_DIRS compute image statistics used to select CSD
    with progressbar.ProgressBar(max_value=prog_len) as progbar:
        full_stats = []
        for dd in DAY_DIRS:
            day_stats = day2stats(dd,mask,progbar=progbar)
            full_stats.append(day_stats)

    # for backup (among other things) save the CSD statistics before computing CSD
    with open(os.path.join(PICKLE_DIR,'csd_stats_save.p'),'wb') as f:
        p.dump(full_stats,f)

    # combine list of day-level dfs into one df
    full_stats = pd.concat(full_stats).reset_index()

    # compute, for each hour of the TSI data, the minimum clearsky_index (CSI) to find CSD
    clearsky_ids = full_stats.groupby('hour').idxmin()['clearsky_index']
    clearsky_dict = {k: full_stats['fname'][int(clearsky_ids[k])] for k in clearsky_ids.keys()}

    # save CSD in a pickle file
    with open(os.path.join(PICKLE_DIR,'clearsky_dict.p'),'wb') as f:
        p.dump(clearsky_dict,f)
