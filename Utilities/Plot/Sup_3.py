from OptimalArray.Utilities.CM4Mat import CovCM4Global,CovCM4GlobalSubsample,CovCM4Indian,CovCM4SO,CovCM4NAtlantic,CovCM4TropicalAtlantic,CovCM4SAtlantic,CovCM4NPacific,CovCM4TropicalPacific,CovCM4SPacific,CovCM4GOM,CovCM4CCS
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
from netCDF4 import Dataset
import numpy as np

for covclass in [CovCM4GlobalSubsample]:
	# for depth in [8,26]:
	mean_removed_list = []
	scaled_data_list = []
	variable_list = []
	depth_list = []
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
			variable_list.append(variable)
			depth_list.append(depth_number)
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

cov_holder = CovCM4Global.load(depth_idx = 2)

fig,axs = plt.subplots(5, 2,figsize=(14,14))
for mr,tl,var,depth in zip(mean_removed_list,scaled_data_list,variable_list,depth_list):
	idx = cov_holder.trans_geo.variable_list.index(var)
	axs[idx,0].plot(sorted(mr),label=(str(depth)+' m'))
	axs[idx,1].plot(sorted(tl))


for k,ylab in enumerate(['$(^\circ C)^2$','$(PSU)^2$','','$(kg\ m^{-3})^2$','$(mol\ m^{-3})^2$']):
	axs[k,0].set_ylabel(ylab)
	axs[k,0].set_yscale('log')

for k,an in enumerate(['a','b','c','d','e','f','g','h','i','j']):
	axs.flatten()[k].annotate(an, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

axs[0,0].legend(bbox_to_anchor=(1, 1.4), loc='upper center',ncol=4)
plt.savefig(plot_handler.out_file('Sup_3'))	
plt.close()