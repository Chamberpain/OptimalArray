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
import geopy
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR as PLOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler

plt.rcParams['font.size'] = '22'
plt.rcParams['text.usetex'] = True
plot_handler = FilePathHandler(PLOT_DIR,'final_figures')

for covclass in [CovCM4GlobalSubsample]:
	# for depth in [8,26]:
	black_sea = []
	western_trop_pac = []
	eastern_top_pac = []
	western_trop_atl = []

	variable_list = []
	depth_list = []
	cov_holder = CovCM4GlobalSubsample.load(depth_idx = 2)

	black_sea_idx = cov_holder.trans_geo.total_list.index(geopy.Point(30, 164))
	western_trop_pac_idx = cov_holder.trans_geo.total_list.index(geopy.Point(2, 144))
	eastern_trop_pac_idx = cov_holder.trans_geo.total_list.index(geopy.Point(2, -100))
	western_trop_atl_idx = cov_holder.trans_geo.total_list.index(geopy.Point(2, -40))

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
				try:
					dh = Dataset(file)
				except OSError:
					continue
				time_list.append(dh['time'][0])
				var_temp = dh[variable][:,dummy.trans_geo.depth_idx,:,:]
				holder_list.append(var_temp[:,dummy.trans_geo.truth_array].data)
			holder_total_list = np.vstack([x for _,x in sorted(zip(time_list,holder_list))])
			black_sea.append(holder_total_list[:,black_sea_idx])
			western_trop_pac.append(holder_total_list[:,western_trop_pac_idx])
			eastern_top_pac.append(holder_total_list[:,eastern_trop_pac_idx])
			western_trop_atl.append(holder_total_list[:,western_trop_atl_idx])

fig,axs = plt.subplots(5, 4,figsize=(18,18))
v_list = [7,.8,0.1,6*10**(-7),0.035]
for bs,wtp,etp,wta,var,depth in zip(black_sea,western_trop_pac,eastern_top_pac,western_trop_atl,variable_list,depth_list):
	# if depth ==15:
	# 	continue
	v = v_list[cov_holder.trans_geo.variable_list.index(var)]
	idx = cov_holder.trans_geo.variable_list.index(var)
	axs[idx,0].plot(bs-bs.mean(),label=(str(depth)+' m'))
	axs[idx,0].set_ylim(-v,v)
	axs[idx,1].plot(wtp-wtp.mean())
	axs[idx,1].set_ylim(-v,v)
	axs[idx,2].plot(etp-etp.mean())
	axs[idx,2].set_ylim(-v,v)
	axs[idx,3].plot(wta-wta.mean())
	axs[idx,3].set_ylim(-v,v)

axs[4,0].set_xlabel('Time (Days)')
axs[4,1].set_xlabel('Time (Days)')
axs[4,2].set_xlabel('Time (Days)')
axs[4,3].set_xlabel('Time (Days)')



for k,ylab in enumerate(['$^\circ C$','$PSU$','','$kg\ m^{-3}$','$mol\ m^{-3}$']):
	axs[k,0].set_ylabel(ylab)
	# axs[k,0].set_yscale('log')

for k,an in enumerate(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t']):
	axs.flatten()[k].annotate(an, xy = (0.23,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

axs[0,0].legend(bbox_to_anchor=(1, 1.6), loc='upper center',ncol=4)
plt.subplots_adjust(hspace=.45)
plt.savefig(plot_handler.out_file('sup_4'))	
plt.close()