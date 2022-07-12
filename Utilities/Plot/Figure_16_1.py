from OptimalArray.Utilities.Plot.Figure_11_15 import make_recent_float_H
from OptimalArray.Utilities.CM4Mat import CovCM4Global
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
from OptimalArray.Utilities.CorGeo import InverseGlobal



plt.rcParams['font.size'] = '22'
plt.rcParams['text.usetex'] = True
data_file_handler = FilePathHandler(ROOT_DIR,'CurrentArgo')
plot_handler = FilePathHandler(PLOT_DIR,'final_figures')

cov_holder = CovCM4Global.load(depth_idx = 2)
XX,YY = cov_holder.trans_geo.get_coords()

data_dict = {}
depth_list = [2,4,6,8,10,12,14,16,18,20]
depths = cov_holder.get_depths().data[depth_list,0]

for depth in depth_list:
	var_list = cov_holder.trans_geo.variable_list[:]
	filename = data_file_handler.tmp_file(str(depth)+'_present_base')
	dummy = InverseGlobal(depth_idx = depth)
	datascale = load(dummy.make_datascale_filename())
	data,var = zip(*datascale)
	datascale_dict = dict(zip(var,data))
	cov_holder = CovCM4Global.load(depth_idx = depth)
	try: 
		out = load(filename)
		print('loaded data')
	except FileNotFoundError:
		print('could not load, calculating...')
		H_array = make_recent_float_H(cov_holder.trans_geo)
		p_hat_array = make_P_hat(cov_holder.cov,H_array,noise_factor=4)
		out_temp = np.split(p_hat_array.diagonal(),len(cov_holder.trans_geo.variable_list))
		out = [cov_holder.trans_geo.transition_vector_to_plottable(x) for x in out_temp]
		save(filename,out)
		del p_hat_array
		del out_temp
		gc.collect(generation=2)
	total_cov = [cov_holder.trans_geo.transition_vector_to_plottable(x) for x in np.split(cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list))]
	out = [x/y for x,y in zip(out,total_cov)]

	if (depth>=cov_holder.chl_depth_idx)&('chl' in var_list):
		var_list.remove('chl')
	for var,data in zip(var_list,out):
		data = data
		try:
			data_dict[var].append((depth,data))
		except KeyError:
			data_dict[var]=[]
			data_dict[var].append((depth,data))

cov_holder = CovCM4Global.load(depth_idx = 2)

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
	XXX,YYY = np.meshgrid([0,1,2,3,4],depths[:len(data_dict[var])])

	for ii,(depth_idx,data) in enumerate(data_dict[var]):
		for jj,mask in enumerate(mask_list):
			dummy_array[ii,jj]=data[mask].mean()
	pcm = ax.pcolor(XXX,YYY,dummy_array,snap=True,cmap='YlOrBr',vmin=0,vmax=1)
	ax.set_yscale('log')
	plt.colorbar(pcm,label = colorbar_label[kk],pad=.07)
	ax.annotate(annotate_list[kk], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.gca().invert_yaxis()
	ax.get_xaxis().set_visible(False)
	ax.set_ylabel('Depth (m)')
ax.get_xaxis().set_visible(True)
ax.set_xticks([0, 1, 2,3,4])
ax.set_xticklabels(name_list,rotation = 20)
plt.subplots_adjust(hspace=0.15)
plt.savefig(plot_handler.out_file('Figure_16_1'))