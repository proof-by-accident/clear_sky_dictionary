from read_funcs import *
from plot_funcs import *

############################
# Script to read all TSIs in DATA_DIR and construct a clearsky dictionary (CSD)
############################
# Author: Peter Shaffery
# Email: pshaff2@gmail.com
# License: None
###########################

# Calculates relevant statistics for all images taken in a given day of the TSI data. Returns a Pandas dataframe (df) where each row corresponds to one TSI
# _Arguments_
# day_dir: the directory corresponding to the day of images for which to compute stats
# mask: the zero mask used to crop image
# progbar: defaults to `None`, otherwise increments the provided progress bar by one
def day2stats(day_dir, mask, progbar=None):
    image_files = [os.path.join(day_dir,f) for f in os.listdir(day_dir)]

    img = None
    stats = [[None]*6]*len(image_files)
    for i in range(len(image_files)):
        f = image_files[i]
        hr = fname_to_datetime(f).hour

        img = mask*rbr_read(f,progbar)

        stats[i] = [f,hr,np.mean(img),np.median(img),np.max(img),np.std(img)]


    stats = pd.DataFrame(stats)
    stats.columns = ['fname','hour','mean','median','max','sd']
    stats['clearsky_index'] = stats['mean']*stats['sd']

    with open(os.path.join(PICKLE_DIR,day_dir.split('/')[-4]+'_'+day_dir.split('/')[-1].split('.')[0]+'.p'),'wb') as f:
        p.dump(stats,f)

    if progbar:
        progbar.update(progbar.value+1)
    else:
        pass

    return(stats)


if __name__=='__main__':
    # Max lengths of progress bar
    prog_len = len([f for dd in DAY_DIRS for f in os.listdir(dd)]) + len(DAY_DIRS)

    # call crop mask
    h,w = (1536,1536)
    mask = crop_mask(h,w)

    # for each day in DAY_DIRS compute the statistics used to select CSD
    with progressbar.ProgressBar(max_value=prog_len) as progbar:
        full_stats = []
        for dd in DAY_DIRS:
            day_stats = day2stats(dd,mask,progbar=progbar)
            full_stats.append(day_stats)

    # save all statistics
    with open(os.path.join(PICKLE_DIR,'csd_stats_save.p'),'wb') as f:
        p.dump(full_stats,f)

    # combine list of dfs into one df
    full_stats = pd.concat(full_stats).reset_index()

    # compute, for each hour of the TSI data, the minimum clearsky_index (CSI) to find CSD
    clearsky_ids = full_stats.groupby('hour').idxmin()['clearsky_index']
    clearsky_dict = {k: full_stats['fname'][int(clearsky_ids[k])] for k in clearsky_ids.keys()}

    # save CSD in a pickle file
    with open(os.path.join(PICKLE_DIR,'clearsky_dict.p'),'wb') as f:
        p.dump(clearsky_dict,f)
