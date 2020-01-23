import os
import argparse as ap
from unpack_images import *

if __name__=='__main__':    
    img_dirs = ['/projects/pesh5067/NREL/clearsky_dict/Data/TSI_images/cloud_images/20190404/11',
                '/projects/pesh5067/NREL/clearsky_dict/Data/TSI_images/cloud_images/20190405/11',
                '/projects/pesh5067/NREL/clearsky_dict/Data/TSI_images/cloud_images/20190406/11',
                '/projects/pesh5067/NREL/clearsky_dict/Data/TSI_images/cloud_images/20190407/11']
    
    for im in img_dirs:
        unpack_images(os.path.join(os.path.curdir,'cloud_images',im))
