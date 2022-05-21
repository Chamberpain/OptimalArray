from OptimalArray.Data.__init__ import ROOT_DIR
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR as PLOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from GeneralUtilities.Data.Filepath.search import find_files
from GeneralUtilities.Data.pickle_utilities import load
from OptimalArray.Utilities.CM4Mat import CovCM4Global
from GeneralUtilities.Compute.list import GeoList
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os
import numpy as np

data_file_handler = FilePathHandler(ROOT_DIR,'OptimalArray')
plot_file_handler = FilePathHandler(PLOT_DIR,'final_figures')
plt.rcParams['font.size'] = '22'
plt.rcParams['text.usetex'] = True

depth = 2
cov_holder = CovCM4Global.load(depth_idx = depth)

def decode_filename(filename):
	basename = os.path.basename(filename)
	dummy,depth_idx,number = basename.split('_')
	return number


filenames = find_files(data_file_handler.tmp_file(''),'*')
filenames.remove(data_file_handler.tmp_file('.DS_Store'))
number_list = [int(x.split('_')[-1]) for x in filenames]
filenames = [x for _,x in sorted(zip(number_list,filenames))]
gs = GridSpec(2, 2, width_ratios=[7, 1], height_ratios=[3,1]) 
fig = plt.figure(figsize=(14,8))
ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
ax1.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
ax2 = fig.add_subplot(gs[1])
ax2.annotate('b', xy = (-0.37,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
ax3 = fig.add_subplot(gs[2])
ax3.annotate('c', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

filename = filenames[-1]
number = decode_filename(filename)
float_array,p_hat = load(filename)
pos_list = GeoList([cov_holder.trans_geo.total_list[x] for x in float_array])
float_lats,float_lons = pos_list.lats_lons()
plot_holder = GlobalCartopy(ax=ax1,adjustable=True)
XX,YY,ax = plot_holder.get_map()
ax.scatter(float_lons,float_lats)
ax2.hist(float_lats,bins=np.arange(-90,91,5),orientation="horizontal")
ax2.set_yticks([-60,-30,0,30,60])
ax2.set_yticklabels(['$60^\circ S$','$30^\circ S$','$0^\circ$','$30^\circ N$','$60^\circ N$'])
ax2.set_ylim([-90,90])
ax2.set_xlabel('Float Count')
ax3.hist(float_lons,bins=np.arange(-180,181,5))
ax3.set_xlim([-180,180])
ax3.set_xticks([-180,-120,-60,0,60,120,180])
ax3.set_xticklabels(['$180^\circ$','$120^\circ W$','$60^\circ W$','$0^\circ$','$60^\circ E$','$120^\circ E$','$180^\circ$'])

ax3.set_ylabel('Float Count')

plt.savefig(plot_file_handler.out_file('Figure_8'))
plt.close()
