from OptimalArray.Utilities.CM4Mat import CovCM4Global
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy

from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list,full_argo_list
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from OptimalArray.Data.__init__ import ROOT_DIR
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR as PLOT_DIR
import cartopy.crs as ccrs
import geopy
from GeneralUtilities.Data.pickle_utilities import load
from TransitionMatrix.Utilities.TransGeo import get_cmap
import datetime
import matplotlib.pyplot as plt
import numpy as np
from TransitionMatrix.Utilities.ArgoData import Core,BGC
from GeneralUtilities.Compute.list import GeoList
from OptimalArray.Utilities.Plot.Figure_9 import random_decode_filename
from GeneralUtilities.Data.Filepath.search import find_files
import copy

plt.rcParams['font.size'] = '18'
plot_handler = FilePathHandler(PLOT_DIR,'final_figures')
data_file_handler = FilePathHandler(ROOT_DIR,'OptimalArray')
cov_holder = CovCM4Global.load(depth_idx = 2)



annotate_list = ['b','c','d','e','f']
colorbar_label = ['$(^\circ C)^2$','$(PSU)^2$','','$(kg\ m^{-3})^2$','$(mol\ m^{-3})^2$']

filenames = find_files(data_file_handler.tmp_file(''),'random*')
random_floatnum_list = []
random_depth_list = []
for filename in filenames:
	float_num,depth_idx,kk = random_decode_filename(filename)
	if float_num>500:
		continue
	if depth_idx>24:
		print(filename)
		continue
	random_floatnum_list.append(float_num)
	random_depth_list.append(depth_idx)
random_floatnum_sorted = sorted(np.unique(random_floatnum_list))
random_depth_sorted = sorted(np.unique(random_depth_list))
depths = cov_holder.get_depths().data[random_depth_sorted,0]

XX,YY = np.meshgrid(random_floatnum_sorted,random_depth_sorted)
idx = [(float_num,float_depth) for float_num,float_depth in zip(XX.flatten(),YY.flatten())]
deep_dict_list = dict(zip(idx,[[] for x in idx]))


XX,YY = np.meshgrid(random_floatnum_sorted,[x for x in random_depth_sorted if x < cov_holder.chl_depth_idx])
idx = [(float_num,float_depth) for float_num,float_depth in zip(XX.flatten(),YY.flatten())]
shallow_dict_list = dict(zip(idx,[[] for x in idx]))



deep_dict = np.zeros([len(random_floatnum_sorted),len(random_depth_sorted)]).flatten().tolist()
data_dict = dict(zip(cov_holder.trans_geo.variable_list,[{},{},{},{},{}]))

for filename in filenames:
	temp_var_list = copy.deepcopy(cov_holder.trans_geo.variable_list)
	float_num,depth_idx,kk = random_decode_filename(filename)
	if depth_idx >= cov_holder.chl_depth_idx:
		temp_var_list.remove('chl')
	if float_num>500:
		continue
	if depth_idx>24:
		print(filename)
		continue
	idx = (float_num,depth_idx)


	H_array,p_hat = load(filename)
	for var,data in zip(temp_var_list,np.split(p_hat,len(temp_var_list))):
		try:
			data_dict[var][idx].append(data.sum())
		except KeyError:
			data_dict[var][idx] = [data.sum()]

H_array, p_hat = load('//Users/paulchamberlain/Projects/OptimalArray/Data/OptimalArray/tmp/random_300_2_6')
float_list = GeoList(cov_holder.trans_geo.total_list[x] for x in H_array)
data = np.split(p_hat,len(cov_holder.trans_geo.variable_list))[cov_holder.trans_geo.variable_list.index('ph')]

x_offset = 4
label_offset_list = [(x_offset,0.5),(x_offset,0.5),(x_offset,0.5),(x_offset,0.6),(x_offset,0.5)]
ax_list = []

fig = plt.figure(figsize=(14,14))
lats,lons = float_list.lats_lons()
ax = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
plot_holder = GlobalCartopy(ax=ax,adjustable=True)
XX,YY,ax0 = plot_holder.get_map()
XX,YY = cov_holder.trans_geo.get_coords()
data = cov_holder.trans_geo.transition_vector_to_plottable(data)
ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
pcm = ax0.pcolormesh(XX,YY,data)
ax0.scatter(lons,lats,c='orange',s=Core.marker_size,zorder=11,label = 'Random')
cbar = fig.colorbar(pcm, orientation="horizontal", pad=0.14)
cbar.ax.set_xlabel('Unconstrained Variance')
cbar.ax.xaxis.set_label_coords(0.5,1.6)
ax0.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#           ncol=3, mode="expand", borderaxespad=0.)
for kk,temp_dict in enumerate(data_dict.values()):
	ax = fig.add_subplot(6,2,(kk+7))
	ax_list.append(ax)
	floatnum_list = []
	depth_list = []	
	for (float_num,depth_idx),data in temp_dict.items():
		floatnum_list.append(float_num)
		depth_list.append(depth_idx)
	unique_depth_list = sorted(np.unique(depth_list))
	unique_floatnum_list = sorted(np.unique(floatnum_list))
	XX,YY = np.meshgrid(unique_floatnum_list,depths[:len(unique_depth_list)])
	dummy_array = np.zeros([len(unique_depth_list),len(unique_floatnum_list)])
	for (float_num,depth_idx),data in temp_dict.items():
		f_idx = unique_floatnum_list.index(float_num)
		d_idx = unique_depth_list.index(depth_idx)
		dummy_array[d_idx,f_idx] = np.mean(data)
	ax.pcolor(XX,YY,dummy_array)
	ax.annotate(annotate_list[kk], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.gca().invert_yaxis()
	ax.get_xaxis().set_visible(False)
	ax.set_ylabel('Depth (m)')
	cb = plt.colorbar(pcm)
	cb.ax.set_ylabel(colorbar_label[kk],rotation=90)
	cb.ax.yaxis.set_label_coords(label_offset_list[kk][0],label_offset_list[kk][1])
ax.get_xaxis().set_visible(True)
ax.set_xlabel('Deployed Floats')
ax_list[3].get_xaxis().set_visible(True)
ax_list[3].set_xlabel('Deployed Floats')
plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=.45)
plt.savefig(plot_handler.out_file('Figure_10'))
plt.close()