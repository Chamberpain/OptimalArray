from OptimalArray.Utilities.CM4Mat import CovCM4Global
import datetime
from OptimalArray.Data.__init__ import ROOT_DIR
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR as PLOT_DIR
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
from GeneralUtilities.Data.pickle_utilities import save,load
from TransitionMatrix.Utilities.TransGeo import get_cmap
import numpy as np
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
import matplotlib.pyplot as plt
from OptimalArray.Utilities.CM4Mat import CovCM4
from OptimalArray.Utilities.CorGeo import InverseGlobal
from matplotlib.gridspec import GridSpec

def make_plot():
	plt.rcParams['font.size'] = '22'
	plt.rcParams['text.usetex'] = True
	data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgo/base')
	plot_handler = FilePathHandler(PLOT_DIR,'final_figures')
	depth_idx_list = [2,4,6,8,10,12,14,16,18,20,22,24]

	base_cov = CovCM4Global.load(depth_idx = 2)
	initial_var_dict = dict(zip(base_cov.trans_geo.variable_list,[[],[],[],[],[]]))
	mapping_error_var_dict = dict(zip(base_cov.trans_geo.variable_list,[[],[],[],[],[]]))
	for depth_idx in depth_idx_list:
		print('depth is ',depth_idx)
		cov_holder = CovCM4Global.load(depth_idx = depth_idx)
		if depth_idx>8:
			try:
				cov_holder.trans_geo.variable_list.remove('chl')
			except:
				pass
		datascale = load(cov_holder.trans_geo.make_datascale_filename())
		data,var1 = zip(*datascale)
		datascale_dict = dict(zip(var1,data))
		p_diag = np.split(cov_holder.cov.diagonal(),len(cov_holder.cov.diagonal())/len(cov_holder.trans_geo.total_list))
		for i,var in enumerate(cov_holder.trans_geo.variable_list):
			print('var is ',var)

			filename = data_file_handler.tmp_file(str(depth_idx)+'_'+str(0)+'_base')
			data = load(filename)
			assert len(data[0])==len(p_diag[0])
			initial_var_dict[var].append((data[i]*datascale_dict[var]).mean())
			for k in range(16):
				filename = data_file_handler.tmp_file(str(depth_idx)+'_'+str(k)+'_base')
				data = load(filename)
				calc_data = (data[i]/p_diag[i]).mean()
				assert calc_data<1
				mapping_error_var_dict[var].append((depth_idx,k,calc_data))


	depths = cov_holder.get_depths().data[depth_idx_list, 0]
	time_list = [x for x in range(3,3*16+1,3)]
	annotate_list = ['a','b','c','d','e']
	colorbar_label = ['$(^\circ C)^2$','$(PSU)^2$','','$(kg\ m^{-3})^2$','$(mol\ m^{-3})^2$']
	base_cov = CovCM4Global.load(depth_idx = 2)

	fig = plt.figure(figsize=(16,14))
	gs = GridSpec(6, 4, width_ratios=[0.1,0.1,1, 7], height_ratios = [0.1,1,1,1,1,1]) 
	ax_list = [fig.add_subplot(gs[ii]) for ii in range(24)]

	for ii in range(3):
		ax_list[ii].axes.xaxis.set_visible(False)
		ax_list[ii].axes.yaxis.set_visible(False)
		ax_list[ii].spines['top'].set_visible(False)
		ax_list[ii].spines['right'].set_visible(False)
		ax_list[ii].spines['bottom'].set_visible(False)
		ax_list[ii].spines['left'].set_visible(False)

	for i,var in enumerate(base_cov.trans_geo.variable_list):
		i +=1
		ax_list[4*i+1].axes.xaxis.set_visible(False)
		ax_list[4*i+1].axes.yaxis.set_visible(False)
		ax_list[4*i+1].spines['top'].set_visible(False)
		ax_list[4*i+1].spines['right'].set_visible(False)
		ax_list[4*i+1].spines['bottom'].set_visible(False)
		ax_list[4*i+1].spines['left'].set_visible(False)

		depth_len = len(initial_var_dict[var])
		XX,YY = np.meshgrid([1,2],depths[:depth_len])
		data = np.array([val for val in initial_var_dict[var] for _ in (0, 1)]).reshape(depth_len,2)
		pcm = ax_list[4*i+2].pcolormesh(XX,YY,data,snap=True)	
		fig.colorbar(pcm,cax=ax_list[4*i],label = colorbar_label[i-1])
		ax_list[4*i].yaxis.tick_left()
		ax_list[4*i].yaxis.set_label_position("left")
		ax_list[4*i+2].invert_yaxis()
		ax_list[4*i+2].axes.xaxis.set_visible(False)
		ax_list[4*i+2].axes.yaxis.set_visible(False)
		ax_list[4*i+2].set_yscale('log')

		XX,YY = np.meshgrid(time_list,depths[:depth_len])
		fill_array = np.zeros(XX.shape)
		for depth_idx,x_index,data in mapping_error_var_dict[var]:
			y_index = depth_idx_list.index(depth_idx)
			fill_array[y_index,x_index] = data
		pcm = ax_list[4*i+3].pcolormesh(XX,YY,fill_array,snap=True,cmap='YlOrBr',vmin=0,vmax=1)
		ax_list[4*i+3].invert_yaxis()
		ax_list[4*i+3].set_yscale('log')
		ax_list[4*i+3].axes.xaxis.set_visible(False)
		ax_list[4*i+3].yaxis.tick_right()
		ax_list[4*i+3].yaxis.set_label_position("right")
		ax_list[4*i+3].set_ylabel('Depth (m)')
	fig.colorbar(pcm,cax=ax_list[3],label = 'Mapping Error',orientation='horizontal')
	ax_list[3].xaxis.tick_top()
	ax_list[3].xaxis.set_label_position("top")
	ax_list[4*i+3].axes.xaxis.set_visible(True)
	ax_list[4*i+3].set_xlabel('Time in Future (months)')
	plt.subplots_adjust(hspace=0.05)
	plt.subplots_adjust(wspace=0)
	annotate_left = ['a','c','e','g','i']
	for k,ii in enumerate([4*i+2 for i in range(1,6)]):
		ax_list[ii].annotate(annotate_left[k], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	annotate_left = ['b','d','f','h','j']
	for k,ii in enumerate([4*i+3 for i in range(1,6)]):
		ax_list[ii].annotate(annotate_left[k], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	plt.savefig(plot_handler.out_file('Figure_22'))
	plt.close()