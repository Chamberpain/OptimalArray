from OptimalArray.Utilities.CM4Mat import CovCM4Global,CovCM4Indian,CovCM4SO,CovCM4NAtlantic,CovCM4TropicalAtlantic,CovCM4SAtlantic,CovCM4NPacific,CovCM4TropicalPacific,CovCM4SPacific,CovCM4GOM,CovCM4CCS
from OptimalArray.Utilities.H import HInstance
from GeneralUtilities.Data.Filepath.search import find_files
import matplotlib.pyplot as plt
import numpy as np 
import uuid
from GeneralUtilities.Compute.list import GeoList
import cartopy.crs as ccrs
from netCDF4 import Dataset
from matplotlib.colors import ListedColormap


plt.rcParams['font.size'] = '30'

title_dict = {'total_variance': 'Total','o2':'$O_2$','so':'Salinity','ph':'pH','chl':'Chlorophyll','thetao':'$T_0$','po4':'Nitrate','ccs':'California Coastal Current','global':'Global',
'gom':'Gulf of Mexico','indian':'Indian','north_atlantic':'North Atlantic','north_pacific':'North Pacific','south_atlantic':'South Atlantic','south_pacific':'South Pacific'
,'southern_ocean':'Southern Ocean','tropical_atlantic':'Tropical Atlantic','tropical_pacific':'Tropical Pacific'}

def get_data_from_dict_list(array_list,depth_idx,float_idx,variable):
	depth_truth_list = np.array([x.tolist()['depth']==depth_idx for x in array_list])
	floatnum_truth_list = np.array([x.tolist()['floatnum']==float_idx for x in array_list])
	variable_truth_list = np.array([variable in x.tolist() for x in array_list])
	mask = depth_truth_list&floatnum_truth_list&variable_truth_list
	data_list = np.array([x.tolist()[variable] for x in np.array(array_list)[mask]])
	return data_list.mean()

def plot_and_save(data_array,XX,YY,filename,title):

	colormap = plt.get_cmap('viridis')
	colormap_r = ListedColormap(colormap.colors[::-1])

	fig = plt.figure(figsize=(13,13))
	plt.pcolor(XX,YY,data_array*100,vmin=0,vmax=95,cmap=colormap_r)
	plt.colorbar(label = '% Constrained')
	plt.gca().invert_yaxis()
	plt.plot([1 / 9.0, 1 / 9.0], [15, 1760],'r',linewidth=8)
	plt.xticks([1 / x ** 2 for x in [6, 5, 4, 3]],["6$^\circ$","5$^\circ$", "4$^\circ$", "3$^\circ$"])
	plt.ylabel('Depth (m)')
	plt.xlabel('Core Argo Float Spacing')
	plt.title(title)
	print('saving ',filename)
	plt.savefig(filename)
	plt.close()


depth_list = [2,4,6,8,10,12,14,16,18,20,22,24]
depth_chl_list = [2,4,6,8]
dummy = CovCM4Indian()
full_variable_list = dummy.trans_geo.variable_list+['total_variance']
total_plot_dict = dict(zip(full_variable_list+['idx_list'], [[] for x in full_variable_list+['idx_list']]))
for covclass in [CovCM4Indian,CovCM4SO,CovCM4NAtlantic,CovCM4TropicalAtlantic,CovCM4SAtlantic,CovCM4NPacific,CovCM4TropicalPacific,CovCM4SPacific,CovCM4GOM,CovCM4CCS]:
	dummy = covclass()

	for variable in full_variable_list:
		if variable is 'chl':
			depth_holder = depth_chl_list
			total_plot_dict['idx_list'].append(dummy.trans_geo.total_list)
		else:
			depth_holder = depth_list
		depths = np.array([x[0] for x in covclass.get_depths()])[depth_holder]
		depths = depths.ravel().tolist()
		try:
			data_array = np.load(dummy.trans_geo.make_array_filename(variable)+'.npy')
			floatnum_list = np.arange(0.001, float(1/5), 0.04)
		except FileNotFoundError:
			array_list = []
			for depth in depth_holder:
				dummy = covclass(depth_idx = depth)
				file_list = find_files(dummy.trans_geo.file_handler.store_file(''),'*.npy')
				for file in file_list:
					array_list.append(np.load(file,allow_pickle=True))
			floatnum_list = np.unique(np.array([x.tolist()['floatnum']for x in array_list]))
			data_array = np.zeros([len(depth_holder),len(floatnum_list)])
			for j,floatnum in enumerate(floatnum_list.tolist()):
				for i,depth in enumerate(depth_holder):
					data_array[i,j] = get_data_from_dict_list(array_list,depth,floatnum,variable)
			np.save(dummy.trans_geo.make_array_filename(variable),data_array)
		total_plot_dict[variable].append(data_array)
		XX,YY = np.meshgrid(floatnum_list.tolist(),depths)
		plot_and_save(data_array,XX,YY,dummy.trans_geo.make_array_filename(variable),title_dict[dummy.trans_geo.region]+' '+title_dict[variable])

for variable in full_variable_list:
	if variable is 'chl':
		depth_holder = np.array([x[0] for x in covclass.get_depths()])[depth_chl_list]
	else:
		depth_holder = np.array([x[0] for x in covclass.get_depths()])[depth_list]	

	array_list = total_plot_dict[variable]
	mean_list = [x.mean() for x in array_list]
	print(variable + ' min idx = '+str(mean_list.index(min(mean_list))))
	print(variable + ' max idx = '+str(mean_list.index(max(mean_list))))

	idx_list = [len(x) for x in total_plot_dict['idx_list']]
	idx_list = [x/sum(idx_list) for x in idx_list]
	data_array = sum([x*y for x,y in zip(array_list,idx_list)])
	XX,YY = np.meshgrid(floatnum_list.tolist(),depth_holder)
	plot_and_save(data_array,XX,YY,dummy.trans_geo.make_array_filename('../../total_'+variable),'Global '+title_dict[variable]+' Variance Constrained')
