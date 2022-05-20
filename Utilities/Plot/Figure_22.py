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

def make_plot():
	plt.rcParams['font.size'] = '22'
	plt.rcParams['text.usetex'] = True
	data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgo/base')
	plot_handler = FilePathHandler(PLOT_DIR,'final_figures')
	depth = 2
	cov_holder = CovCM4Global.load(depth_idx = depth)
	depth_idx_list = [2,4,6,8,10,12,14,16,18,20,22,24]
	depths = cov_holder.get_depths().data[depth_idx_list, 0]
	time_list = [x for x in range(3,3*16+1,3)]
	annotate_list = ['a','b','c','d','e']
	colorbar_label = ['$(^\circ C)^2$','$(PSU)^2$','','$(kg\ m^{-3})^2$','$(mol\ m^{-3})^2$']
	fig = plt.figure(figsize=(14,14))
	for i,var in enumerate(cov_holder.trans_geo.variable_list):
		ax = fig.add_subplot(5,1,(i+1))
		array_list = []
		for depth_idx in depth_idx_list:
			if (depth_idx>=cov_holder.chl_depth_idx)&(var=='chl'):
				continue
			temp_list = []
			for k in range(16):
				filename = data_file_handler.tmp_file(str(depth_idx)+'_'+str(k)+'_base')
				data = load(filename)
				try:
					temp_list.append(data[i].mean())
				except IndexError:
					temp_list.append(data[i-1].mean())
			array_list.append(np.stack(temp_list))
		data_array = np.stack(array_list)
		XX,YY = np.meshgrid(time_list,depths[:data_array.shape[0]])
		pcm = ax.pcolor(XX,YY,data_array,snap=True)
		plt.gca().invert_yaxis()
		plt.colorbar(pcm,label = colorbar_label[i])
		ax.annotate(annotate_list[i], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
		ax.get_xaxis().set_visible(False)
		ax.set_ylabel('Depth (m)')
	ax.get_xaxis().set_visible(True)
	ax.set_xlabel('Time in Future (months)')
	plt.subplots_adjust(hspace=0.05)
	plt.savefig(plot_handler.out_file('Figure_22'))
	plt.close()