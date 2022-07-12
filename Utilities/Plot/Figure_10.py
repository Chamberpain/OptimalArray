from OptimalArray.Utilities.CM4Mat import CovCM4Global
from OptimalArray.Utilities.CorGeo import InverseGlobal
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
from GeneralUtilities.Data.pickle_utilities import load
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
import matplotlib
import numpy as np
from TransitionMatrix.Utilities.ArgoData import Core,BGC
from GeneralUtilities.Compute.list import GeoList
from OptimalArray.Utilities.Plot.Figure_9 import random_decode_filename
from GeneralUtilities.Data.Filepath.search import find_files
import copy
from matplotlib.gridspec import GridSpec
from OptimalArray.Utilities.MakeRandom import make_filename
from OptimalArray.Utilities.CorGeo import InverseGlobal


plt.rcParams['font.size'] = '16'
plt.rcParams["axes.axisbelow"] = False
plot_handler = FilePathHandler(PLOT_DIR,'final_figures')
data_file_handler = FilePathHandler(ROOT_DIR,'RandomArray')
base_cov = CovCM4Global.load(depth_idx = 2)
annotate_list = ['a','b','c','d','e','f','g','h','i','j','l','m']
colorbar_label = ['$(^\circ C)^2$','$(PSU)^2$','','$(kg\ m^{-3})^2$','$(mol\ m^{-3})^2$']
var_translate_dict = {'so':'Salinity','chl':'Chlorophyll','ph':'pH','o2':'Oxygen','thetao':'Temperature'}
filenames = find_files(data_file_handler.tmp_file(''),'random*')
data_dict = dict(zip(base_cov.trans_geo.variable_list,[{},{},{},{},{}]))


for k,filename in enumerate(filenames):
	print(k,' of ',len(filenames))
	temp_var_list = copy.deepcopy(base_cov.trans_geo.variable_list)
	float_num,depth_idx,kk = random_decode_filename(filename)
	if depth_idx >= base_cov.chl_depth_idx:
		temp_var_list.remove('chl')
	if float_num>1000:
		continue
	if depth_idx>24:
		print(filename)
		continue
	if depth_idx<2:
		print(filename)
		continue
	idx = (float_num,depth_idx)

	dummy = InverseGlobal(depth_idx = depth_idx)
	datascale = load(dummy.make_datascale_filename())
	data,var = zip(*datascale)
	datascale_dict = dict(zip(var,data))

	H_array,p_hat = load(filename)
	for var,data in zip(temp_var_list,np.split(p_hat,len(temp_var_list))):
		try:
			data_dict[var][idx].append(data.sum())
		except KeyError:
			data_dict[var][idx] = [data.sum()]
floatnum, depth_idx = zip(*list(data_dict['thetao'].keys()))
unique_depth_list = list(set(depth_idx))
depths = base_cov.get_depths().data[unique_depth_list, 0]


final_var_dict = dict(zip(base_cov.trans_geo.variable_list,[[],[],[],[],[]]))
for depth_idx in unique_depth_list:
	print('depth is ',depth_idx)
	temp_var_list = copy.deepcopy(base_cov.trans_geo.variable_list)
	
	if depth_idx>=10:
		temp_var_list.remove('chl')
	datascale = load(InverseGlobal(depth_idx=depth_idx).make_datascale_filename())
	data,var = zip(*datascale)
	datascale_dict = dict(zip(var,data))

	for i,var in enumerate(temp_var_list):
		temp = []
		for kk in range(50):
			make_filename('random',depth_idx,kk)
			temp.append((data[i]*datascale_dict[var]**2).mean())
		final_var_dict[var].append(np.mean(temp))

x_offset = 7
label_offset_list = [(x_offset+3,0.5),(x_offset,0.5),(x_offset,0.5),(x_offset,0.6),(x_offset+6,0.5)]

fig = plt.figure(figsize=(14,14))
gs = GridSpec(8, 2, width_ratios=[7, 2], height_ratios = [0.1,0.8,0.8,0.8,0.8,0.8,0.5,2]) 
ax_list = [fig.add_subplot(gs[ii]) for ii in [0]+list(range(2,12))]


var_dict = {}
for kk,temp_dict in enumerate(data_dict.values()):
	var = list(data_dict.keys())[kk]
	ax = ax_list[2*kk+1]
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
		dummy_array[d_idx,f_idx] = np.mean(data)/np.mean(temp_dict[(0,depth_idx)])
	var_dict[var] = dummy_array
	pcm1 = ax.pcolor(XX,YY,dummy_array,cmap='YlOrBr',vmin=0,vmax=1)
	ax.annotate(annotate_list[2*kk], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	ax.invert_yaxis()
	ax.set_yscale('log')
	ax.axes.xaxis.set_visible(False)
	ax.set_ylabel('Depth (m)')

	ax = ax_list[2*kk+2]
	depth_len = len(final_var_dict[var])
	XX,YY = np.meshgrid([1,2],depths[:depth_len])
	data = np.array([val for val in final_var_dict[var] for _ in (0, 1)]).reshape(depth_len,2)
	pcm2 = ax.pcolormesh(XX,YY,data,snap=True)	
	fig.colorbar(pcm2,ax=ax,label = colorbar_label[kk])
	ax.invert_yaxis()
	ax.set_yscale('log')
	ax.annotate(annotate_list[2*kk+1], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	ax.axes.xaxis.set_visible(False)
	ax.axes.yaxis.set_visible(False)
fig.colorbar(pcm1,cax=ax_list[0],label = 'Mapping Error',orientation='horizontal')
ax_list[0].xaxis.tick_top()
ax_list[0].xaxis.set_label_position("top")
ax_list[9].axes.xaxis.set_visible(True)
ax_list[9].set_xlabel('Deployed Floats',zorder=50)

deep_list = []
shallow_list = []

gs = GridSpec(8, 2, width_ratios=[1, 1], height_ratios = [0.1,0.8,0.8,0.8,0.8,0.8,0.3,2]) 
ax = fig.add_subplot(gs[14])
for var in var_dict:
	data = var_dict[var]
	ax.plot(unique_floatnum_list,data.mean(axis=0),label=var_translate_dict[var])
	shallow_list.append(data[:4,:].mean(axis=0))
	if var != 'chl':
		deep_list.append(data[4:,:].mean(axis=0))
ax.set_ylim([0.7,1.05])
ax.legend(ncol=3,bbox_to_anchor=(1, 1.2),fancybox=True)
ax.annotate(annotate_list[10], xy = (0.1,0.1),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
ax.set_xlabel('Deployed Floats',zorder=50)
ax.set_ylabel('Mapping Error',zorder=50)

shallow_stacked = np.vstack(shallow_list)
deep_stacked = np.vstack(deep_list)
ax = fig.add_subplot(gs[15])
ax.plot(unique_floatnum_list,shallow_stacked.mean(axis=0),label='Upper')
ax.fill_between(unique_floatnum_list,shallow_stacked.mean(axis=0)-shallow_stacked.std(axis=0),shallow_stacked.mean(axis=0)+shallow_stacked.std(axis=0),alpha=0.2)
ax.plot(unique_floatnum_list,deep_stacked.mean(axis=0),label='Lower')
ax.fill_between(unique_floatnum_list,deep_stacked.mean(axis=0)-deep_stacked.std(axis=0),deep_stacked.mean(axis=0)+deep_stacked.std(axis=0),alpha=0.2)

ax.legend(ncol=2,loc='upper right',fancybox=True)
ax.set_ylim([0.7,1.05])
ax.axes.yaxis.set_visible(False)
ax.annotate(annotate_list[11], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
ax.set_xlabel('Deployed Floats',zorder=50)

plt.subplots_adjust(wspace=0)
plt.savefig(plot_handler.out_file('Figure_10'))
plt.close()
