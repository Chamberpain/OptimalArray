from OptimalArray.Utilities.MOM6Mat import CovMOM6CCS
import scipy.sparse.linalg
import gc
import os
import shutil
from OptimalArray.Utilities.CorMat import CovElement
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from GeneralUtilities.Data.Filepath.instance import make_folder_if_does_not_exist
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
from TransitionMatrix.Utilities.TransGeo import get_cmap
from OptimalArray.Utilities.CM4Mat import CovCM4GlobalSubsample
from netCDF4 import Dataset
import numpy as np
import matplotlib.colors as colors
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR as PLOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler

plt.rcParams['font.size'] = '22'
plt.rcParams['text.usetex'] = True
plot_handler = FilePathHandler(PLOT_DIR,'final_figures')

def return_var(depth):
	mean_removed_list = []
	scaled_data_list = []
	depth_number = CovCM4GlobalSubsample.get_depths()[depth,0]
	print('depth idx is '+str(depth))
	dummy = CovCM4GlobalSubsample(depth_idx = depth)
	make_folder_if_does_not_exist(dummy.trans_geo.make_diagnostic_plot_folder())
	master_list = dummy.get_filenames()
	array_variable_list = []
	data_scale_list = []
	for variable,files in master_list:
		time_list = []
		holder_list = []
		if dummy.trans_geo.depth_idx>=dummy.chl_depth_idx:
			if variable=='chl':
				continue
		for file in files:
			print(file)
			dh = Dataset(file)
			time_list.append(dh['time'][0])
			var_temp = dh[variable][:,dummy.trans_geo.depth_idx,:,:]
			holder_list.append(var_temp[:,dummy.trans_geo.truth_array].data)
		holder_total_list = np.vstack([x for _,x in sorted(zip(time_list,holder_list))])
		if variable=='chl':
			assert (holder_total_list>0).all()
			holder_total_list = np.log(holder_total_list)
			mean_removed,holder_total_list,data_scale = dummy.normalize_data(holder_total_list)
			print(holder_total_list.var().max())
		
		else:
			mean_removed,holder_total_list,data_scale = dummy.normalize_data(holder_total_list)				
			print(holder_total_list.var().max())
		mean_removed_list.append(mean_removed.var(axis=0))
		scaled_data_list.append(holder_total_list.var(axis=0))
	return (mean_removed_list,scaled_data_list)

dummy = CovCM4GlobalSubsample(depth_idx = 2)
mean_removed_list_2,scaled_var_list_2 = return_var(2)
mean_removed_list_8,scaled_var_list_8 = return_var(8)
mean_removed_list_26,scaled_var_list_26 = return_var(26)



fig = plt.figure(figsize=(14,14))

ax0 = fig.add_subplot(3,2,1,projection=ccrs.PlateCarree())
plot_holder = GlobalCartopy(ax=ax0,adjustable=True)
XX,YY,ax0 = plot_holder.get_map()
plottable = dummy.trans_geo.transition_vector_to_plottable(mean_removed_list_2[0])
pcm = ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
XX,YY = dummy.trans_geo.get_coords()
pcm = ax0.pcolormesh(XX,YY,plottable,norm=colors.LogNorm())
plt.colorbar(pcm)
ax0.annotate('a', xy = (0.23,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

ax1 = fig.add_subplot(3,2,2,projection=ccrs.PlateCarree())
plot_holder = GlobalCartopy(ax=ax1,adjustable=True)
XX,YY,ax1 = plot_holder.get_map()
plottable = dummy.trans_geo.transition_vector_to_plottable(mean_removed_list_2[1])
pcm = ax1.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
XX,YY = dummy.trans_geo.get_coords()
pcm = ax1.pcolormesh(XX,YY,plottable,norm=colors.LogNorm())
plt.colorbar(pcm)
ax1.annotate('b', xy = (0.23,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)


ax2 = fig.add_subplot(3,2,3,projection=ccrs.PlateCarree())
plot_holder = GlobalCartopy(ax=ax2,adjustable=True)
XX,YY,ax2 = plot_holder.get_map()
plottable = dummy.trans_geo.transition_vector_to_plottable(mean_removed_list_8[2])
pcm = ax2.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
XX,YY = dummy.trans_geo.get_coords()
pcm = ax2.pcolormesh(XX,YY,plottable,norm=colors.LogNorm())
plt.colorbar(pcm)
ax2.annotate('c', xy = (0.23,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)


ax3 = fig.add_subplot(3,2,4,projection=ccrs.PlateCarree())
plot_holder = GlobalCartopy(ax=ax3,adjustable=True)
XX,YY,ax3 = plot_holder.get_map()
plottable = dummy.trans_geo.transition_vector_to_plottable(mean_removed_list_2[3])
pcm = ax3.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
XX,YY = dummy.trans_geo.get_coords()
pcm = ax3.pcolormesh(XX,YY,plottable,norm=colors.LogNorm())
plt.colorbar(pcm)
ax3.annotate('d', xy = (0.23,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)


ax4 = fig.add_subplot(3,2,5,projection=ccrs.PlateCarree())
plot_holder = GlobalCartopy(ax=ax4,adjustable=True)
XX,YY,ax4 = plot_holder.get_map()
plottable = dummy.trans_geo.transition_vector_to_plottable(mean_removed_list_26[3])
pcm = ax4.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
XX,YY = dummy.trans_geo.get_coords()
pcm = ax4.pcolormesh(XX,YY,plottable,norm=colors.LogNorm())
plt.colorbar(pcm)
ax4.annotate('e', xy = (0.23,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
