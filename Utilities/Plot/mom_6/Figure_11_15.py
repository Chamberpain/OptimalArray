from OptimalArray.Utilities.MOM6Mat import CovMOM6CCS,InverseCristina
from GeneralUtilities.Data.Lagrangian.Argo.array_class import ArgoArray
from GeneralUtilities.Compute.list import VariableList, GeoList
from OptimalArray.Utilities.H import Float,HInstance
from OptimalArray.Utilities.Utilities import make_P_hat
import datetime
import numpy as np 
import matplotlib.pyplot as plt
from GeneralUtilities.Plot.Cartopy.regional_plot import CCSCartopy
import cartopy.crs as ccrs
from TransitionMatrix.Utilities.TransGeo import get_cmap
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR as PLOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from OptimalArray.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.pickle_utilities import save,load
data_file_handler = FilePathHandler(ROOT_DIR,'MOM6CurrentArgo')
plt.rcParams['font.size'] = '16'
plot_handler = FilePathHandler(PLOT_DIR,'mom6_final_figures')
variable_translation_dict = {'TEMP':'thetao','PSAL':'so','CHLA':'chl','DOXY':'o2','NITRATE':'no3'}

def recent_floats(GeoClass, array_instance):
	id_list = array_instance.get_id_list()
	deployed_list = array_instance.get_deployment_date_list()
	date_list = array_instance.get_recent_date_list()
	date_mask = [max(date_list) - datetime.timedelta(days=180) < x for x in date_list]
	bin_list = array_instance.get_recent_bins(GeoClass.get_lat_bins(),GeoClass.get_lon_bins())
	bin_list_mask = [x in GeoClass.total_list for x in bin_list]
	sensor_list = array_instance.get_sensors()
	sensor_idx_list = []
	sensor_mask = []
	for sensors in sensor_list:
		sensors = [x for x in sensors if x in variable_translation_dict.keys()]
		sensor_idx = [variable_translation_dict[x] for x in sensors if variable_translation_dict[x] in GeoClass.variable_list]
		sensor_idx_list.append(sensor_idx)
		if sensor_idx:
			sensor_mask.append(True)
		else:
			sensor_mask.append(False)
	total_mask = np.array(bin_list_mask)&np.array(date_mask)&np.array(sensor_mask)
	pos_list = np.array(bin_list)[total_mask].tolist()
	sensor_list = np.array(sensor_idx_list)[total_mask].tolist()
	deployed_date_list = np.array(deployed_list)[total_mask].tolist()
	id_list = np.array(id_list)[total_mask].tolist()
	return (pos_list,sensor_list,deployed_date_list,id_list)

def make_recent_float_H(trans_geo):
	array = ArgoArray()
	lats = trans_geo.get_lat_bins()
	lons = trans_geo.get_lon_bins()
	recent_mask = array.get_recent_mask()
	pos_mask = [trans_geo.ocean_shape.contains(x) for x in GeoList(array.get_recent_pos()).to_shapely()]
	recent_bins = array.get_recent_bins(lats,lons)
	deployment_date_list = array.get_deployment_date_list()
	id_list = array.get_id_list()
	sensor_list = array.get_sensors()
	H_array = HInstance(trans_geo=trans_geo)
	for time_mask,p_mask,pos,sensor,date,id_ in zip(recent_mask,pos_mask,recent_bins,sensor_list,deployment_date_list,id_list):
		if (~time_mask)&(p_mask):
			print('I am an active float')
			if pos in trans_geo.total_list:
				sensor = [x for x in sensor if x in variable_translation_dict.keys()]
				sensor = [variable_translation_dict[x] for x in sensor if variable_translation_dict[x] in trans_geo.variable_list]
				sensor = VariableList(sensor)
				print(sensor)
				if sensor:
					print('I am adding a float')
					H_array.add_float(Float(pos,sensor,date,id_))
	return H_array

def make_plots():
	try:
		shallow_out,shallow_var = load(data_file_handler.tmp_file('shallow_p_hat'))
		print('loaded shallow data')
	except FileNotFoundError:
		print('could not load shallow data, calculating...')
		cov_holder = CovMOM6CCS.load(depth_idx = 2)
		H_array = make_recent_float_H(cov_holder.trans_geo)
		p_hat_array = make_P_hat(cov_holder.cov,H_array,noise_factor=4)
		surface_out = np.split(p_hat_array.diagonal() / cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list))
		surface_var = cov_holder.trans_geo.variable_list
		save(data_file_handler.tmp_file('shallow_p_hat'),(surface_out,surface_var))
		del cov_holder
		del p_hat_array

	try:
		mid_out,mid_var = load(data_file_handler.tmp_file('mid_p_hat'))
		print('loaded mid data')
	except FileNotFoundError:
		print('could not load mid data, calculating...')
		cov_holder = CovMOM6CCS.load(depth_idx = 8)
		H_array = make_recent_float_H(cov_holder.trans_geo)
		p_hat_array = make_P_hat(cov_holder.cov,H_array,noise_factor=4)
		mid_out = np.split(p_hat_array.diagonal() / cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list))
		mid_var = cov_holder.trans_geo.variable_list
		save(data_file_handler.tmp_file('mid_p_hat'),(mid_out,mid_var))
		# del cov_holder
		del p_hat_array

	try:
		deep_out,deep_var = load(data_file_handler.tmp_file('deep_p_hat'))
		print('loaded deep data')
	except FileNotFoundError:
		print('could not load deep data, calculating...')
		cov_holder = CovMOM6CCS.load(depth_idx = 18)
		H_array = make_recent_float_H(cov_holder.trans_geo)
		p_hat_array = make_P_hat(cov_holder.cov,H_array,noise_factor=4)
		deep_out = np.split(p_hat_array.diagonal() / cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list))
		deep_var = cov_holder.trans_geo.variable_list
		save(data_file_handler.tmp_file('deep_p_hat'),(deep_out,deep_var))
		del cov_holder
		del p_hat_array


	label_translation_dict = {'no3':'Nitrate Equipped','chl':'Chlorophyll Equipped','o2':'Oxygen Equipped'}
	cov_holder = CovMOM6CCS.load(depth_idx = 2)
	H_array = make_recent_float_H(cov_holder.trans_geo)
	for k,var in enumerate(cov_holder.trans_geo.variable_list):
		print(var)
		core_pos = H_array.return_pos_of_core()
		bgc_pos = H_array.return_pos_of_bgc()
		lat_core,lon_core = core_pos.lats_lons()
		lat_bgc,lon_bgc = bgc_pos.lats_lons()


		fig = plt.figure(figsize=(14,14))
		ax0 = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
		plot_holder = CCSCartopy(ax=ax0)
		XX,YY,ax0 = plot_holder.get_map()
		XX,YY = cov_holder.trans_geo.get_coords()
		data = shallow_out[k]
		data = cov_holder.trans_geo.transition_vector_to_plottable(data)
		ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
		ax0.pcolormesh(XX,YY,data,cmap='YlOrBr',vmin=0,vmax=1)
		ax0.scatter(lon_core,lat_core,c='green',s=20,zorder=11,label = 'Core')
		ax0.scatter(lon_bgc,lat_bgc,c='blue',s=30,zorder=11,label = 'BGC')
		if var in ['no3','chl','o2']:
			var_pos = H_array.return_pos_of_variable(var)
			lat_var,lon_var = var_pos.lats_lons()
			ax0.scatter(lon_var,lat_var,c='cyan',s=40,zorder=11,label = label_translation_dict[var])
		ax0.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

		ax1 = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
		plot_holder = CCSCartopy(ax=ax1)
		XX,YY,ax1 = plot_holder.get_map()
		XX,YY = cov_holder.trans_geo.get_coords()
		if var == 'chl':
			data = mid_out[k]
		else:
			idx = deep_var.index(var)
			data = deep_out[idx]
		data = cov_holder.trans_geo.transition_vector_to_plottable(data)
		ax1.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
		pcm = ax1.pcolormesh(XX,YY,data,cmap='YlOrBr',vmin=0,vmax=1)
		ax1.scatter(lon_core,lat_core,c='green',s=20,zorder=11,label = 'Core')
		ax1.scatter(lon_bgc,lat_bgc,c='blue',s=30,zorder=11,label = 'BGC')
		if var in ['no3','chl','o2']:
			var_pos = H_array.return_pos_of_variable(var)
			lat_var,lon_var = var_pos.lats_lons()
			ax1.scatter(lon_var,lat_var,c='cyan',s=40,zorder=11,label = label_translation_dict[var])
		ax1.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
		fig.colorbar(pcm,ax=[ax0,ax1],pad=.05,label='Mapping Error',location='bottom')
		plt.legend(ncol=3,bbox_to_anchor=(-.3, 1.02, -.3, 1.02), loc=3)
		fig_num = 11+k
		plt.savefig(plot_handler.out_file('Figure_'+str(fig_num)),bbox_inches='tight')
		plt.close()
