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

for covclass in [CovCM4GlobalSubsample]:
	# for depth in [8,26]:
	mean_removed_list = []
	scaled_data_list = []
	for depth in [2,6,14,18]:
		depth_number = covclass.get_depths()[depth,0]
		print('depth idx is '+str(depth))
		dummy = covclass(depth_idx = depth)
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


	fig = plt.figure(figsize=(14,14))
	ax0 = fig.add_subplot(2,1,1,projection=ccrs.PlateCarree())
	raw_var_combine_plot = dummy.trans_geo.transition_vector_to_plottable(np.stack([x/np.median(x) for x in mean_removed_list]).sum(axis=0))
	plot_holder = GlobalCartopy(ax=ax0,adjustable=True)
	XX,YY,ax0 = plot_holder.get_map()
	pcm = ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
	XX,YY = dummy.trans_geo.get_coords()
	pcm = ax0.pcolormesh(XX,YY,raw_var_combine_plot,norm=colors.LogNorm())
	plt.colorbar(pcm,label='combined raw data variance')
	ax0.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	scaled_var_combine_plot = dummy.trans_geo.transition_vector_to_plottable(np.stack(scaled_data_list).sum(axis=0))
	ax1 = fig.add_subplot(2,1,2,projection=ccrs.PlateCarree())
	plot_holder = GlobalCartopy(ax=ax1,adjustable=True)
	XX,YY,ax1 = plot_holder.get_map()
	pcm = ax1.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
	XX,YY = dummy.trans_geo.get_coords()
	pcm = ax1.pcolormesh(XX,YY,scaled_var_combine_plot,norm=colors.LogNorm())
	plt.colorbar(pcm,label='combined scaled data variance')
	ax1.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig(plot_handler.out_file('Sup_2'))	
	plt.close()
