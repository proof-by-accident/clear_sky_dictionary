from read_funcs import *
from plot_funcs import *
from sun_position import *

with open(PICKLE_DIR+'/csd_stats_save.p','rb') as f:
    stats = pd.concat(p.load(f)).rename(columns={'mean':'rbr_mean',
                                                 'sd':'rbr_sd',
                                                 'median':'rbr_median',
                                                 'max':'rbr_max',
                                                 'clearsky_index':'clearsky_pseudoindex'})

def fname_to_time_clip_sec(fname):
    time =fname_to_time(fname)
    return(time.replace(second=0))

stats.insert(len(stats.columns),'time',stats.fname.apply(fname_to_time_clip_sec))

with open(os.path.abspath(os.path.join(DATA_DIR,'../clearsky_index/csi.csv')),'r') as f:
    csi = pd.read_csv(f)

def csi_time_parse(s):
    s = s[:-5] + ''.join(s[-5:].split(':'))
    t = dt.strptime(s,'%Y-%m-%d %H:%M:%S%z')
    return(t.astimezone(pytz.timezone('US/Mountain')))

csi = csi.rename(columns={'time':'tstamp'})
csi.insert(len(csi.columns),'time',csi.tstamp.apply(csi_time_parse))
csi.insert(len(csi.columns),'Direct CHP1-1 [W/m^2]',csi_comps.iloc[:,6])
csi.insert(len(csi.columns),'Direct Extraterrestrial (calc) [W/m^2]',csi_comps.iloc[:,7])

stats = pd.merge(stats,csi,on='time',how='left')

clearsky_ids = stats.groupby('hour').idxmin()['clearsky_pseudoindex']
clearsky_rows = stats.iloc[clearsky_ids]
dcsi_ids = stats.groupby('hour').idxmax()['dcsi']
dcsi_rows = stats.iloc[dcsi_ids]
mean_ids = stats.groupby('hour').idxmin()['rbr_mean']
mean_rows = stats.iloc[mean_ids]
sd_ids = stats.groupby('hour').idxmin()['dcsi']
sd_rows = stats.iloc[sd_ids]

ex1 = clearsky_rows.fname.iloc[4]
ex2 = clearsky_rows.fname.iloc[10]

FIGURE_DIR = os.path.abspath(os.path.join(PYTHON_DIR,'../../figures'))
plt.imshow(rgb_read(ex1))
plt.savefig(os.path.join(FIGURE_DIR,'ex1.png'))
plt.close()

plt.imshow(rgb_read(ex2))
plt.savefig(os.path.join(FIGURE_DIR,'ex2.png'))
plt.close()

plt.hist(stats.dcsi,alpha=0.5,label='TSIs',normed=True,stacked=False)
plt.hist(clearsky_rows.dcsi.iloc[:-1],alpha=0.5,label='CSD',color='red',normed=True,stacked=False)
plt.legend(fontsize=20)
plt.xlabel('DCSI',size=20)
plt.ylabel('Percent of TSIs',size=20)
plt.xticks(size = 14)
plt.yticks(size = 14)
#plt.show()
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,'csd_dcsi_hist.png'))
plt.close()

x = stats.clearsky_pseudoindex
y = stats.dcsi
keep = np.where((1-np.isnan(stats.dcsi))*stats.dcsi>0.01)
x = x.iloc[keep]
y = y.iloc[keep]
lm = scipy.stats.linregress(x,y)

plt.scatter(stats.clearsky_pseudoindex,stats.dcsi,s=.5)
#plt.scatter(x,y)
xgrid = np.linspace(np.min(x),np.max(x),1000)
plt.plot(xgrid,lm[1] + lm[0]*xgrid,color='red')
plt.xlabel('Pseudo-index',size=20)
plt.ylabel('DCSI',size=20)
plt.xticks(size = 14)
plt.yticks(size = 14)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,'csd_dcsi_scatter.png'))
plt.close()


clearsky_qs = [np.sum(stats.dcsi<_)/len(stats.dcsi) for _ in clearsky_rows.dcsi]
qs = np.arange(0,1,.01)
dcsi_qs = np.nanquantile(stats.dcsi,qs)
plt.plot(dcsi_qs,qs)
plt.scatter(clearsky_rows.dcsi[1:-1],clearsky_qs[1:-1],marker='*',c='red',s=200)
plt.xlabel('DCSI',size=20)
plt.ylabel('Quantile',size=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,'csd_dcsi_quants.png'))
plt.close()

def to_day(t):
    try:
        return(float(t[8:10]))
    except:
        return(0)


foo = stats.loc[np.logical_and(stats.hour>7,stats.hour<17)]
days = [to_day(_) for _ in foo.time]
plt.scatter(foo.iloc[:,7],foo.dcsi,s=.5,c=days,cmap='viridis')
grid = np.linspace(np.min(x),np.max(x),1000)
plt.xlabel('Pseudo-index',size=20)
plt.ylabel('DCSI',size=20)
plt.xticks(size = 14)
plt.yticks(size = 14)
plt.tight_layout()
#plt.show()
plt.close()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(foo.clearsky_pseudoindex,foo.hour,foo.dcsi,s=.5,c=days,cmap='viridis')
xgrid = np.linspace(np.min(x),np.max(x),1000)
ax.set_xlabel('Pseudo-index',size=20)
ax.set_ylabel('Hour',size=20)
ax.set_zlabel('DCSI',size=20)
#plt.show()
plt.close()

xgrid = np.arange(clearsky_rows.dcsi[1:-1].min(),clearsky_rows.dcsi[1:-1].max(),.01)
plt.plot(xgrid,xgrid,c='blue')
plt.scatter(clearsky_rows.dcsi[1:-1],dcsi_rows.dcsi[1:-1],c='red',s=200)
plt.xlabel('CSD DCSI',size=20)
plt.ylabel('Best Achievable DCSI',size=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,'csd_dcsi_perf.png'))
plt.show()
plt.close()


xgrid = np.arange(foo.dcsi[1:-1].min(),foo.dcsi[1:-1].max(),.01)
plt.plot(xgrid,xgrid,c='blue')
plt.scatter(clearsky_rows.dcsi[1:-1],dcsi_rows.dcsi[1:-1],c='red',s=200)
plt.xlabel('CSD DCSI',size=20)
plt.ylabel('Best Achievable DCSI',size=20)
plt.tight_layout()
plt.show()
