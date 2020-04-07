# Introduction
This subdirectory contains the bulk of the scripts used to process the TSIs and create a CSD, as well as some additional scripts which perform other analyses that didn't make it into the writeup.

# Contents
The subdirectory contains three utility scripts: `read_funcs.py`, `plot_funcs.py`, and `sun_position.py`. The `read_funcs` are a collection of functions which take in an image filename and return either an RGB data volume or some image computed from the RGB (eg. RBR image). `plot_funcs` contains plotting functions used throughout the other scripts. Finally, `sun_position` contains functions used to convert an image's filename to a timestamp, as well as to locate the center of the sun in an image, given its timestamp.

The `make_csd.py` script actually compiles the CSD.  Its results are stored as a Python pickle file (a serialized format) in the `/pickles` subdirectory. The `dmd.py` script contains functions to perform the Dynamic Mode Decomposition (DMD) on a series of images. While these functions are not used by `make_csd`, they are used by some scripts found in the `/drafts` subdirectory. This contains some code which performs various analyses that aren't really complete.

Finally, all package requirements are listed in the `reqs` file. I advise that all scripts here are run either within a python `virtualenv` (through the module `venv`) or in an Anaconda environment. If doing the former, packages can be installed by running `pip install -r ./reqs` (after activating the `virtualenv`). If using Anaconda I believe the proper command would be `conda install --file ./reqs`, however you will also need to specify the Anaconda enviroment into which you are installing the packages.
