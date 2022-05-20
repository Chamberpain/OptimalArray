from OptimalArray.Utilities.Plot.Figure_11_15 import make_recent_float_H
from OptimalArray.Utilities.CM4Mat import CovCM4Global
from OptimalArray.Utilities.Plot.Figure_22 import get_units
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR as PLOT_DIR
from OptimalArray.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.pickle_utilities import save,load
from OptimalArray.Utilities.Utilities import make_P_hat
import gc
import matplotlib.pyplot as plt
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader
from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import full_argo_list, aggregate_argo_list
import numpy as np
from OptimalArray.Utilities.CM4Mat import CovCM4


plt.rcParams['font.size'] = '22'
plt.rcParams['text.usetex'] = True
data_file_handler = FilePathHandler(ROOT_DIR,'CurrentArgo')
plot_handler = FilePathHandler(PLOT_DIR,'final_figures')

cov_holder = CovCM4Global.load(depth_idx = 2)
XX,YY = cov_holder.trans_geo.get_coords()

data_dict = {}
depth_list = [2,4,6,8,10,12,14,16,18,20,22,24]
depths = cov_holder.get_depths().data[depth_list,0]

for depth in depth_list:
	var_list = cov_holder.trans_geo.variable_list[:]
	filename = data_file_handler.tmp_file(str(depth)+'_present_base')
	try: 
		out = load(filename)
		print('loaded data')
	except FileNotFoundError:
		print('could not load, calculating...')
		cov_holder = CovCM4Global.load(depth_idx = depth)
		H_array = make_recent_float_H(cov_holder.trans_geo)
		p_hat_array = make_P_hat(cov_holder.cov,H_array,noise_factor=4)
		out_temp = np.split(p_hat_array.diagonal(),len(cov_holder.trans_geo.variable_list))
		out = [cov_holder.trans_geo.transition_vector_to_plottable(x) for x in out_temp]
		save(filename,out)
		del p_hat_array
		del out_temp
		gc.collect(generation=2)
	if depth>cov_holder.chl_depth_idx:
		var_list.remove('chl')
	for var,data in zip(var_list,out):
		try:
			data_dict[var].append((depth,data))
		except KeyError:
			data_dict[var]=[]
			data_dict[var].append((depth,data))


antarctic_mask = YY<=-60
S_midlatitudes_mask = (YY>-60)&(YY<=-20)
tropics_mask= (YY>-20)&(YY<=20)
N_midlatitudes_mask = (YY>20)&(YY<=60)
arctic_mask = YY>60
mask_list = [antarctic_mask,S_midlatitudes_mask,tropics_mask,N_midlatitudes_mask,arctic_mask]
name_list = ['Southern Ocean','South Midlatitudes','Tropics','North Midlatitudes','Artic']
units_list = CovCM4.get_units(cov_holder)
colorbar_label = ['$(^\circ C)^2$','$(PSU)^2$','','$(kg\ m^{-3})^2$','$(mol\ m^{-3})^2$']
annotate_list = ['a','b','c','d','e']
fig = plt.figure(figsize=(14,14))
for kk,var in enumerate(cov_holder.trans_geo.variable_list):
	ax = fig.add_subplot(5,1,(kk+1))
	dummy_array = np.zeros([len(data_dict[var]),len(mask_list)])
	XXX,YYY = np.meshgrid([0,1,2,3,4,5],depths[:len(data_dict[var])])

	for ii,(depth_idx,data) in enumerate(data_dict[var]):
		for jj,mask in enumerate(mask_list):
			dummy_array[ii,jj]=data[mask].mean()
	pcm = ax.pcolor(XXX,YYY,dummy_array,snap=True)
	plt.colorbar(pcm,label = colorbar_label[kk])
	ax.annotate(annotate_list[kk], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.gca().invert_yaxis()
	ax.get_xaxis().set_visible(False)
	ax.set_ylabel('Depth (m)')
ax.get_xaxis().set_visible(True)
ax.set_xticks([0,0.5, 1.5, 2.5,3.5,4.5,5])
ax.set_xticklabels(['']+name_list+[''],rotation = 20)
plt.subplots_adjust(hspace=0.05)
plt.savefig(plot_handler.out_file('Figure_16'))