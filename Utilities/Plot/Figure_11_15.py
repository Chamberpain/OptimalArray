from TransitionMatrix.Utilities.ArgoData import Core, BGC
from OptimalArray.Utilities.CM4Mat import CovCM4Global
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader
from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import full_argo_list, aggregate_argo_list
from GeneralUtilities.Compute.list import VariableList
from OptimalArray.Utilities.H import Float,HInstance
from OptimalArray.Utilities.Utilities import make_P_hat
import datetime
import numpy as np 
import matplotlib.pyplot as plt
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
import cartopy.crs as ccrs
from TransitionMatrix.Utilities.TransGeo import get_cmap
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR as PLOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from OptimalArray.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.pickle_utilities import save,load
data_file_handler = FilePathHandler(ROOT_DIR,'CurrentArgo')
plt.rcParams['font.size'] = '16'
plot_handler = FilePathHandler(PLOT_DIR,'final_figures')
variable_translation_dict = {'TEMP':'thetao','PSAL':'so','PH_IN_SITU_TOTAL':'ph','CHLA':'chl','DOXY':'o2'}

def recent_floats(GeoClass, FloatClass):
	id_list = FloatClass.get_id_list()
	deployed_list = FloatClass.get_deployment_date_list()
	date_list = FloatClass.get_recent_date_list()
	date_mask = [max(date_list) - datetime.timedelta(days=180) < x for x in date_list]
	bin_list = FloatClass.get_recent_bins(GeoClass.get_lat_bins(),GeoClass.get_lon_bins())
	bin_list_mask = [x in GeoClass.total_list for x in bin_list]
	sensor_list = FloatClass.get_sensors()
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
	aggregate_argo_list(read_class=BGCReader)
	bgc_pos_list,bgc_sensor_list,bgc_deployed_date,bgc_id_list = recent_floats(trans_geo,BGCReader)
	aggregate_argo_list()
	core_pos_list,core_sensor_list,core_deployed_date,core_id_list = recent_floats(trans_geo,ArgoReader)

	for pos in bgc_pos_list:
		idx = core_pos_list.index(pos)
		core_deployed_date.remove(core_deployed_date[idx])
		core_pos_list.remove(pos)
		core_sensor_list.remove(core_sensor_list[0])
		core_id_list.remove(core_id_list[idx])
	pos_list = bgc_pos_list + core_pos_list
	sensor_list = [VariableList(x) for x in bgc_sensor_list + core_sensor_list]
	deployed_date_list =bgc_deployed_date+core_deployed_date
	id_list = bgc_id_list + core_id_list

	H_array = HInstance(trans_geo=trans_geo)
	for pos,sensor,date,id_ in zip(pos_list,sensor_list,deployed_date_list,id_list):
		H_array.add_float(Float(pos,sensor,date,id_))
	return H_array

def make_plots():
	try:
		shallow_out,shallow_var = load(data_file_handler.tmp_file('shallow_p_hat'))
		print('loaded shallow data')
	except FileNotFoundError:
		print('could not load shallow data, calculating...')
		cov_holder = CovCM4Global.load(depth_idx = 2)
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
		cov_holder = CovCM4Global.load(depth_idx = 8)
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
		cov_holder = CovCM4Global.load(depth_idx = 18)
		H_array = make_recent_float_H(cov_holder.trans_geo)
		p_hat_array = make_P_hat(cov_holder.cov,H_array,noise_factor=4)
		deep_out = np.split(p_hat_array.diagonal() / cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list))
		deep_var = cov_holder.trans_geo.variable_list
		save(data_file_handler.tmp_file('deep_p_hat'),(deep_out,deep_var))
		del cov_holder
		del p_hat_array


	label_translation_dict = {'ph':'pH Equipped','chl':'Chlorophyll Equipped','o2':'Oxygen Equipped'}
	cov_holder = CovCM4Global.load(depth_idx = 2)
	H_array = make_recent_float_H(cov_holder.trans_geo)
	for k,var in enumerate(cov_holder.trans_geo.variable_list):
		print(var)
		core_pos = H_array.return_pos_of_core()
		bgc_pos = H_array.return_pos_of_bgc()
		lat_core,lon_core = core_pos.lats_lons()
		lat_bgc,lon_bgc = bgc_pos.lats_lons()


		fig = plt.figure(figsize=(14,14))
		ax0 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
		plot_holder = GlobalCartopy(ax=ax0,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		XX,YY = cov_holder.trans_geo.get_coords()
		data = shallow_out[k]
		data = cov_holder.trans_geo.transition_vector_to_plottable(data)
		ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
		ax0.pcolormesh(XX,YY,data,cmap='YlOrBr',vmin=0,vmax=1)
		ax0.scatter(lon_core,lat_core,c='green',s=Core.marker_size,zorder=11,label = 'Core')
		ax0.scatter(lon_bgc,lat_bgc,c='blue',s=BGC.marker_size,zorder=11,label = 'BGC')
		if var in ['ph','chl','o2']:
			var_pos = H_array.return_pos_of_variable(var)
			lat_var,lon_var = var_pos.lats_lons()
			ax0.scatter(lon_var,lat_var,c='black',s=BGC.marker_size,zorder=11,label = label_translation_dict[var])
		ax0.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)



		ax1 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
		plot_holder = GlobalCartopy(ax=ax1,adjustable=True)
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
		ax1.scatter(lon_core,lat_core,c='green',s=Core.marker_size,zorder=11,label = 'Core')
		ax1.scatter(lon_bgc,lat_bgc,c='blue',s=BGC.marker_size,zorder=11,label = 'BGC')
		if var in ['ph','chl','o2']:
			var_pos = H_array.return_pos_of_variable(var)
			lat_var,lon_var = var_pos.lats_lons()
			ax1.scatter(lon_var,lat_var,c='black',s=BGC.marker_size,zorder=11,label = label_translation_dict[var])
		ax1.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
		fig.colorbar(pcm,ax=[ax0,ax1],pad=.05,label='Scaled Unconstrained Variance',location='right')
		plt.legend(ncol=3,bbox_to_anchor=(.3, 1.02, .3, 1.02), loc=3)
		fig_num = 11+k
		plt.savefig(plot_handler.out_file('Figure_'+str(fig_num)),bbox_inches='tight')
		plt.close()
