from OptimalArray.Data.__init__ import ROOT_DIR
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR as PLOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from GeneralUtilities.Data.Filepath.search import find_files
from GeneralUtilities.Data.pickle_utilities import load
from OptimalArray.Utilities.ComputeOptimal import CovCM4Optimal
from GeneralUtilities.Compute.list import GeoList
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib

data_file_handler = FilePathHandler(ROOT_DIR,'OptimalArray')
plot_file_handler = FilePathHandler(PLOT_DIR,'final_figures')
plt.rcParams['font.size'] = '22'
plt.rcParams['text.usetex'] = True

cov_holder = CovCM4Optimal.load()

def decode_filename(filename):
	basename = os.path.basename(filename)
	dummy,depth_idx,number = basename.split('_')
	return number


filenames = find_files(data_file_handler.tmp_file(''),'optimal*')
number_list = [int(x.split('_')[-1]) for x in filenames]
filenames = [x for _,x in sorted(zip(number_list,filenames))]
gs = GridSpec(2, 2, width_ratios=[7, 1], height_ratios=[3,1]) 
fig = plt.figure(figsize=(14,8))
ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
ax1.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
plot_holder = GlobalCartopy(ax=ax1,adjustable=True)
XX,YY,ax1 = plot_holder.get_map()
ax2 = fig.add_subplot(gs[1])
ax2.annotate('b', xy = (-0.37,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
ax3 = fig.add_subplot(gs[2])
ax3.annotate('c', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

def float_plot(filename,zorder,color):
	number = decode_filename(filename)
	float_array,p_hat = load(filename)
	pos_list = GeoList([cov_holder.trans_geo.total_list[x] for x in float_array])
	float_lats,float_lons = pos_list.lats_lons()
	ax1.scatter(float_lons,float_lats,zorder=zorder,color=color,label=str(number)+' Floats')
	domain_lats,domain_lons = cov_holder.trans_geo.total_list.lats_lons()
	domain_counts, bins = np.histogram(domain_lats,bins=np.arange(-90,91,4))
	domain_counts[domain_counts == 0] = 1
	float_counts, bins = np.histogram(float_lats,bins=np.arange(-90,91,4))
	ax2.hist(bins[:-1], bins, weights=float_counts/domain_counts,orientation="horizontal",zorder=zorder,color=color)
	ax2.set_yticks([-60,-30,0,30,60])
	ax2.set_yticklabels(['$60^\circ S$','$30^\circ S$','$0^\circ$','$30^\circ N$','$60^\circ N$'])
	ax2.set_ylim([-90,90])
	ax2.set_xlabel('Floats/Grid Cell')
	domain_counts, bins = np.histogram(domain_lons,bins=np.arange(-180,181,4))
	domain_counts[domain_counts == 0] = 1
	float_counts, bins = np.histogram(float_lons,bins=np.arange(-180,181,4))
	ax3.hist(bins[:-1], bins, weights=float_counts/domain_counts,zorder=zorder,color=color)
	ax3.set_xlim([-180,180])
	ax3.set_xticks([-180,-120,-60,0,60,120,180])
	ax3.set_xticklabels(['$180^\circ$','$120^\circ W$','$60^\circ W$','$0^\circ$','$60^\circ E$','$120^\circ E$','$180^\circ$'])
	ax3.set_xlabel('Floats/Grid Cell')

cmap = matplotlib.cm.get_cmap('plasma_r')
# for k,(filename,zorder) in enumerate([(filenames[800],17),(filenames[700],18),(filenames[600],19),(filenames[500],20),(filenames[400],21),(filenames[300],22),(filenames[200],23),(filenames[100],24)]):
for k,(filename,zorder) in enumerate([(filenames[1000],16),(filenames[900],17),(filenames[800],17),(filenames[700],18),(filenames[600],19),(filenames[500],20),(filenames[400],21),(filenames[300],22),(filenames[200],23),(filenames[100],24)]):

	norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
	color = cmap(norm(k))
	float_plot(filename,zorder,color)

ax1.legend(bbox_to_anchor=(0.55, 1.25), loc='upper center',ncol=5,columnspacing = 0.01,handletextpad=0.1,labelspacing = 0.01)
plt.savefig(plot_file_handler.out_file('Figure_8'))
plt.close()


def make_movie():
	float_array,p_hat = load(filenames[0])
	base_cov = cov_holder.trans_geo.transition_vector_to_plottable(p_hat)
	plot_file_handler = FilePathHandler(PLOT_DIR,'optimal_array')
	for k,floatnum in enumerate(np.arange(0,1001,10)):
		fig = plt.figure(figsize=(14,8))
		ax1 = fig.add_subplot(projection=ccrs.PlateCarree())
		plot_holder = GlobalCartopy(ax=ax1,adjustable=True)
		XX,YY,ax1 = plot_holder.get_map()
		XX,YY = cov_holder.trans_geo.get_coords()
		file_ = filenames[floatnum]
		float_array,p_hat = load(file_)
		pos_list = GeoList([cov_holder.trans_geo.total_list[x] for x in float_array])
		try:
			float_lats,float_lons = pos_list.lats_lons()
		except ValueError:
			float_lats = []
			float_lons = []
		pcm = ax1.pcolor(XX,YY,cov_holder.trans_geo.transition_vector_to_plottable(p_hat)/base_cov,cmap='YlOrBr',vmin=0,vmax=1)
		fig.colorbar(pcm,ax=[ax1],label='Mapping Error',location='right',fraction=0.032, pad=0.04)
		ax1.scatter(float_lons,float_lats,s=70,c='b')
		plt.savefig(plot_file_handler.tmp_file(str(k)))
		plt.close()