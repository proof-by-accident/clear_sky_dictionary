from datetime import datetime as dt
from datetime import timezone as tz
from datetime import timedelta as td
import pytz
import pvlib as pv

from read_funcs import *


# `solpix_loc` takes the filename of a TSI which it parses into a time of day. In then passes this time
# to the solar position algorithm provided by `pvlib` package and converts the output into a tuple which
# corresponds to the solar pixel location in `fname`
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

# parses the TSI filename into its timestamp
def fname_to_time(fname):
    year,mon,day,time = fname.split('/')[-4:]
    time = time.split('.')[0]
    tstamp = dt(*[int(_) for _ in [year,mon,day,time[:2],time[2:4],time[4:]]])
    return(tstamp.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('US/Mountain')))
