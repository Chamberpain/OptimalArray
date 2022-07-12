from OptimalArray.Data.__init__ import ROOT_DIR
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR as PLOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from GeneralUtilities.Data.Filepath.search import find_files
from GeneralUtilities.Compute.list import BaseList
from GeneralUtilities.Data.pickle_utilities import load
from OptimalArray.Utilities.CM4Mat import CovCM4Global
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os
import numpy as np


optimal_data_file_handler = FilePathHandler(ROOT_DIR,'OptimalArray')
random_data_file_handler = FilePathHandler(ROOT_DIR,'RandomArray')
plot_file_handler = FilePathHandler(PLOT_DIR,'final_figures')
plt.rcParams['font.size'] = '22'
plt.rcParams['text.usetex'] = True


def snr_decode_filename(filename):
	basename = os.path.basename(filename)
	dummy,snr,depth_idx,number = basename.split('_')
	snr = float(snr.replace('-','.'))
	return (snr, int(depth_idx),int(number))

def random_decode_filename(filename):
	basename = os.path.basename(filename)
	dummy,float_num,depth_idx,kk = basename.split('_')
	return (int(float_num), int(depth_idx),int(kk))

def optimal_decode_filename(filename):
	basename = os.path.basename(filename)
	dummy,depth_idx,number = basename.split('_')
	return (int(depth_idx),int(number))

def make_plot():
	filenames = find_files(optimal_data_file_handler.tmp_file(''),'snr*')
	snr_list = []
	depth_list = []
	kk_list = []
	for filename in filenames:
		snr,depth_idx,kk = snr_decode_filename(filename)
		snr_list.append(snr)
	snr_sorted = sorted(np.unique(snr_list))
	random_list = [[] for x in range(len(snr_sorted))]
	optimal_list = [[] for x in range(len(snr_sorted))]
	for filename in filenames:
		snr,depth_idx,kk = snr_decode_filename(filename)
		H_ideal_index,H_random_index,p_hat_ideal,p_hat_random = load(filename)
		random_list[snr_sorted.index(snr)].append(p_hat_random.sum())
		optimal_list[snr_sorted.index(snr)].append(p_hat_ideal.sum())
	random_snr_std = [np.std(x) for x in random_list]
	random_snr_mean = [np.mean(x) for x in random_list]
	ideal_snr_mean = [np.mean(x) for x in optimal_list]
	filenames = find_files(random_data_file_handler.tmp_file(''),'random*')
	random_floatnum_list = []
	for filename in filenames:
		float_num,depth_idx,kk = random_decode_filename(filename)
		if depth_idx!=0:
			continue
		random_floatnum_list.append(float_num)
	random_floatnum_sorted = sorted(np.unique(random_floatnum_list))
	random_list = [[] for x in range(len(random_floatnum_sorted))]
	for filename in filenames:
		float_num,depth_idx,kk = random_decode_filename(filename)
		if depth_idx!=0:
			continue
		if not load(filename):
			continue 
		H_array,p_hat = load(filename)
		random_list[random_floatnum_sorted.index(float_num)].append(p_hat.sum())


	filenames = find_files(optimal_data_file_handler.tmp_file(''),'optimal*')
	optimal_floatnum_list = []
	for filename in filenames:
		depth_idx,float_num = optimal_decode_filename(filename)
		if depth_idx!=0:
			continue
		optimal_floatnum_list.append(float_num)
	optimal_floatnum_sorted = sorted(np.unique(optimal_floatnum_list))
	optimal_list = [[] for x in range(len(optimal_floatnum_sorted))]
	for filename in filenames:
		depth_idx,float_num = optimal_decode_filename(filename)
		if depth_idx!=0:
			continue
		H_array,p_hat = load(filename)
		optimal_list[optimal_floatnum_sorted.index(float_num)]= p_hat.sum()

	random_mean = np.array([np.mean(x) for x in random_list])
	random_std = np.array([np.std(x) for x in random_list])



	fig = plt.figure(figsize = (14,14))
	ax1 = fig.add_subplot(2,1,1)
	mean = (np.array(ideal_snr_mean)+np.array(random_snr_mean))/2
	diff = (np.array(ideal_snr_mean)-np.array(random_snr_mean))/mean*100
	ax1.plot(snr_sorted,diff,'g',label='Random Array')
	ax1.fill_between(snr_sorted,np.array(diff)+np.array(random_snr_std)*100/mean,np.array(diff)-np.array(random_snr_std)/mean*100,color='g',alpha=0.2)
	# ax1.plot(snr_sorted,ideal_snr_mean,'b',label='Optimal Array')
	ax1.set_xscale('log')
	# ax1.set_yscale('log')
	ax1.set_xlabel("Signal to Noise Ratio")
	ax1.set_ylabel('Optimal Array \% Difference')
	ax1.annotate('a', xy = (0.17,0.9),zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	ax2 = fig.add_subplot(2,1,2)
	ax2.plot(random_floatnum_sorted,random_mean,'r',label='Random Array')
	ax2.fill_between(random_floatnum_sorted,random_mean+random_std,random_mean-random_std,color='r',alpha=0.2)
	ax2.plot(optimal_floatnum_sorted[:1000],optimal_list[:1000],'b',label='Optimal Array')

	ideal_equivalent_floats = optimal_floatnum_sorted[1000]
	last_p_hat = optimal_list[1000]
	random_equivalent_floats = random_floatnum_sorted[BaseList.find_nearest(random_mean.tolist(), last_p_hat,idx = True)]
	ax2.plot([ideal_equivalent_floats,random_equivalent_floats],[last_p_hat,last_p_hat],'k--')
	float_diff = random_equivalent_floats-ideal_equivalent_floats
	ax2.annotate(str(float_diff)+' Float Outperformance', xy = (np.mean([ideal_equivalent_floats,random_equivalent_floats]),last_p_hat),
		xytext = (ideal_equivalent_floats*0.7,last_p_hat*1.2),arrowprops=dict(facecolor='black', shrink=0.05),zorder=11,size=22)



	ax2.set_xlabel("Floats Deployed")
	ax2.set_ylabel('Total Unconstrained Variance')
	ax2.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.legend()
	plt.savefig(plot_file_handler.out_file('Figure_9'))
	plt.close()
