from OptimalArray.Utilities.CorMat import CovElement
from OptimalArray.Utilities.CM4Mat import CovLowCM4Global
from OptimalArray.Utilities.__init__ import ROOT_DIR
from GeneralUtilities.Data.Lagrangian.Argo.array_class import ArgoArray
from GeneralUtilities.Data.Lagrangian.Argo.argo_read import ArgoReader
from OptimalArray.Utilities.Plot.Figure_17_21 import FutureFloatTrans
from GeneralUtilities.Data.Filepath.instance import FilePathHandler, make_folder_if_does_not_exist
from GeneralUtilities.Data.pickle_utilities import save,load
from OptimalArray.Utilities.Utilities import make_P_hat
from TransitionMatrix.Utilities.TransGeo import get_cmap
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.ArgoData import Core,BGC
from OptimalArray.Utilities.H import HInstance, Float
from GeneralUtilities.Compute.list import VariableList,LatList,LonList
from GeneralUtilities.Data.Lagrangian.Argo.utilities import BaseReadClass,ProfDate
from GeneralUtilities.Data.Lagrangian.Argo.prof_class import BaseProfClass,BGCProfClass
import geopy
import numpy as np
import pandas as pd
import datetime
from copy import deepcopy
from GeneralUtilities.Data.Mapped.cm4 import CM4Sal,CM4ThetaO,CM4O2,CM4PO4,CM4PH,CM4CHL,CM4DIC
import gsw
import pickle
from OptimalArray.Utilities.CM4DIC import CovCM4LowCorrelation,CovCM4MediumCorrelation,CovCM4HighCorrelation
import os 

label_translation_dict = {'ph':'pH Equipped','chl':'Chlorophyll Equipped','o2':'Oxygen Equipped','po4':'Nitrate Equipped'}


class MultTrans(FutureFloatTrans):
	data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/')

	@classmethod
	def save_multiple(cls):
		trans_mat = FutureFloatTrans.load_from_type(lat_spacing=2,lon_spacing=2,time_step=90)
		trans_list = []
		for x in range(20):
			trans_holder = trans_mat.multiply(x,value=0.00001)
			trans_holder.setup(days = 90*(1+x))
			trans_holder.save(filename=cls.make_filename(x))

	@classmethod
	def make_filename(cls,x):
		return cls.data_file_handler.out_file('trans_mat_'+str(x))

	@classmethod
	def load_multiple(cls,x):
		trans_holder = cls.load(cls.make_filename(x))
		trans_holder.setup(days = 90*(1+x))
		return trans_holder

trans_list = []
date_list = []
for x in range(20):
	trans_list.append(MultTrans.load_multiple(x))
	date_list.append(datetime.datetime(2025,6,10)+datetime.timedelta(days = 90*(1+x)))



class Base20sArray():
	def __init__(self,array,core_only = False):
		self.df = pd.read_csv(self.csv_file,names=['ID','Date','Lat','Lon'],sep='\s+',infer_datetime_format=True,parse_dates=['Date'])
		self.dict = FutureArgoArray()
		base_key = list(array.keys())[-1]
		for k,(id_,date,lat,lon) in self.df.iterrows():
			id_ = str(id_)
			deployment_pos = geopy.Point(lat,lon)
			try:
				dict_item = deepcopy(array[id_])
				# if dict_item.prof.name == 'BGC':
				# 	break
				if (dict_item.prof.name == 'BGC')&(core_only):
					print('This is a BGC Profile that is being removed')
					continue
				self.dict['advance_'+id_] = dict_item
				self.dict['advance_'+id_].prof.pos = BaseReadClass.Position([deployment_pos])
				self.dict['advance_'+id_].prof.date = ProfDate([date.to_pydatetime()+datetime.timedelta(days=365*5)])

			except:
				print(id_,' Wasnt in the list. Thats really wierd')
				dict_item = deepcopy(array[base_key])
				if (dict_item.prof.name == 'BGC')&(core_only):
					print('This is a BGC Profile that is being removed')
					continue				
				self.dict['advance_'+id_] = dict_item
				self.dict['advance_'+id_].prof.pos = BaseReadClass.Position([deployment_pos])
				self.dict['advance_'+id_].prof.date = ProfDate([date.to_pydatetime()+datetime.timedelta(days=365*5)])
				self.dict['advance_'+id_].meta.id = id_
				continue

class USA20sArray(Base20sArray):
	csv_file = '/Users/paulchamberlain/Projects/GeneralUtilities/DataDir/OptimalArray/FutureArgoMixedLayer/US_2020s_deployments'

class Int20sArray(Base20sArray):
	csv_file = '/Users/paulchamberlain/Projects/GeneralUtilities/DataDir/OptimalArray/FutureArgoMixedLayer/Intl_2020s_deployments'

class FutureArgoArray(ArgoArray):
	label = 'base'
	lifetime=''

	data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/')
	def __init__(self,*args,num=99999,**kwargs):
		super().__init__(*args, **kwargs)

	@classmethod
	def make_labelname(cls):
		return cls.label+'_'+str(cls.lifetime)

	@classmethod
	def make_filename(cls):
		return cls.data_file_handler.out_file('array_'+cls.make_labelname())

	@classmethod
	def load_preprocessed(cls):
		with open(cls.make_filename(),'rb') as pickle_file:
			out_data = pickle.load(pickle_file)
		out_data.lifetime = cls.lifetime
		return out_data

	def save_preprocessed(self):
		with open(self.make_filename(), 'wb') as pickle_file:
			pickle.dump(self,pickle_file)
		pickle_file.close()

	def return_time_array(self,time):
		dict_holder = FutureArgoArray(zip(self.keys(),self.values()))
		date_mask = dict_holder.get_time_mask(time)
		recent_ids = [item for item, condition in zip(dict_holder.keys(),date_mask) if condition]
		old_ids = [item for item in dict_holder.keys() if item not in recent_ids]
		for id_ in old_ids:
			dict_holder.pop(id_)

		deployment_date_list = dict_holder.get_deployment_date_list()
		time_diff_mask = [abs((time-item).days)>(365*self.lifetime) for item in deployment_date_list]  #
		dead_float_ids = [item for item, condition in zip(dict_holder.keys(),time_diff_mask) if condition]
		for id_ in dead_float_ids:
			dict_holder.pop(id_)		
		return dict_holder

	def advance_floats(self,trans_list):
		lats = trans_list[0].trans_geo.get_lat_bins()
		lons = trans_list[0].trans_geo.get_lon_bins()
		future_dates = [datetime.timedelta(days=(k+1)*trans_list[0].days) for k,data in enumerate(trans_list)]
		for dummy_float_id in self.keys():
			print(dummy_float_id)
			dummy_float = self[dummy_float_id]
			dummy_float.prof.date += [dummy_float.prof.date[-1] + x for x in future_dates]
			dummy_pos = dummy_float.prof.pos[-1]
			dummy_lat = lats.find_nearest(dummy_pos.latitude)
			dummy_lon = lons.find_nearest(dummy_pos.longitude)
			try:
				start_idx = trans_list[0].trans_geo.total_list.index(geopy.Point(dummy_lat,dummy_lon))

			except ValueError:
				print('the float was outside the coordinates of the transition_matrix, so it will stay put')
				dummy_float.prof.pos += [geopy.Point(dummy_lat,dummy_lon) for x in future_dates]
				continue #loop to next float in the array

			for trans_mat in trans_list:
				new_lon = trans_list[0].trans_geo.total_list[start_idx].longitude + trans_mat.east_west[start_idx]
				if new_lon > 180:
					new_lon-=360
				if new_lon <-180:
					new_lon += 360
				new_lat = trans_list[0].trans_geo.total_list[start_idx].latitude + trans_mat.north_south[start_idx]
				new_lon = lons.find_nearest(new_lon)
				new_lat = lats.find_nearest(new_lat)
				new_point = geopy.Point(new_lat,new_lon)
				try:
					idx = trans_list[0].trans_geo.total_list.index(new_point)

				except ValueError:
					print('the projected float was outside the coordinates of the model, so it will go to closest point')
					dist_list = [geopy.distance.great_circle(new_point,x) for x in trans_list[0].trans_geo.total_list]
					idx = dist_list.index(min(dist_list))
					new_point = trans_list[0].trans_geo.total_list[idx]
				print(new_point)
				dummy_float.prof.pos.append(new_point)
				old_pos = dummy_float.prof.pos[-2]
				new_pos = dummy_float.prof.pos[-1]
				dist = geopy.distance.great_circle(new_pos,old_pos)
				if dist==0:
					print('the float didnt move')
				else:
					print('the float moved from ',old_pos)
					print('to ',new_pos)
				assert geopy.distance.great_circle(new_pos,old_pos).km<(trans_list[0].days*20)

	def return_date_H(self,trans_geo,date):
		variable_dict = {'TEMP':'thetao','NITRATE':'po4','PSAL':'so','CHLA':'chl','PH_IN_SITU_TOTAL':'ph','DOXY':'o2'}
		H_holder = HInstance(trans_geo = trans_geo)
		date_mask = self.get_time_mask(date)
		sensor_list = [item for item, condition in zip(self.get_sensors(),date_mask) if condition]
		bin_list = [item for item, condition in zip(self.get_time_bins(trans_geo.get_lat_bins(),trans_geo.get_lon_bins(),date),date_mask) if condition]
		deployment_date_list = [item for item, condition in zip(self.get_deployment_date_list(),date_mask) if condition]
		id_list = [item for item, condition in zip(self.keys(),date_mask) if condition]
		for sensor, pos, deployment_date,id_ in zip(sensor_list, bin_list, deployment_date_list,id_list):
			sensor = [variable_dict[x] for x in sensor if x in variable_dict.keys()]
			try: 
				trans_geo.total_list.index(pos)
			except ValueError:
				dist_list = [geopy.distance.great_circle(pos,x) for x in trans_geo.total_list]
				min_dist_idx = dist_list.index(min(dist_list))
				pos = trans_geo.total_list[min_dist_idx]
			if (date-deployment_date).days<(365*self.lifetime):
				H_holder.add_float(Float(pos,VariableList(sensor),date_deployed=deployment_date,ID=id_))
		return H_holder

	@classmethod
	def make_H_filename(cls,label):
		return cls.data_file_handler.out_file('H_'+cls.make_labelname()+'_'+label)

	@classmethod
	def save_H(cls,trans_geo,label):
		assert label in ['deep','shallow','int']
		full_array = cls.load_preprocessed()
		full_array.lifetime = cls.lifetime
		future_H_list = []
		for x in range(20):
			date = datetime.datetime(2025,6,10)+datetime.timedelta(days = 90*(1+x))
			print(date)
			future_H_list.append(full_array.return_date_H(trans_geo,date))
		with open(cls.make_H_filename(label), 'wb') as pickle_file:
			pickle.dump(future_H_list,pickle_file)
		pickle_file.close()

	@classmethod
	def load_H(cls,label):
		with open(cls.make_H_filename(label),'rb') as pickle_file:
			out_data = pickle.load(pickle_file)
		return out_data

	def return_average_time(self,date):
		return np.mean([(date-item.meta.launch_date).days/365. for name,item in self.items()])

	def return_position(self,date):
		core_pos_list = []
		bgc_pos_list = []
		for name,item in self.items():
			idx = item.prof.date.find_nearest(date,idx=True)
			pos = item.prof.pos[idx]
			if len(item.get_variables())>3:
				bgc_pos_list.append(pos)
			else:
				core_pos_list.append(pos)
		return (bgc_pos_list,core_pos_list)

	def return_time_len(self):
		len_list = []
		for name,item in self.items():
			len_list.append(len(item.prof.pos))

	def dist_between_profiles(self):
		dist_list = []
		for name,item in self.items():
			pos_list = item.prof.pos
			dummy = []
			for k in range(len(pos_list)-1):
				dummy.append(geopy.distance.great_circle(pos_list[k+1],pos_list[k]).km)
			dist_list.append(max(dummy))

	@classmethod
	def load_bgc_int_data(self):
		data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+self.label+'/'+str(self.lifetime))
		data_list = []
		for var_label in ['ph','chl','o2','no3']:
			data_list.append([load(data_file_handler.out_file(var_label+'_uncertainty_'+str(x))) for x in range(20)])
		return zip(['ph','chl','o2','no3'],data_list)

	@classmethod
	def load_bgc_surface_data(self):
		data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+self.label+'/'+str(self.lifetime))
		data_list = []
		for var_label in ['ph','chl','o2','no3']:
			data_list.append([load(data_file_handler.out_file(var_label+'_surfaceuncertainty_'+str(x))) for x in range(20)])
		return zip(['ph','chl','o2','no3'],data_list)

	@classmethod
	def load_steric_data(self):
		data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+self.label+'/'+str(self.lifetime))
		return [load(data_file_handler.out_file('sealevel_'+str(x))) for x in range(20)]

	@classmethod
	def load_heat_data(self):
		data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+self.label+'/'+str(self.lifetime))
		return [load(data_file_handler.out_file('heat_uncertainty_'+str(x))) for x in range(20)]

	@classmethod
	def load_ml_data(self):
		data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+self.label+'/'+str(self.lifetime))
		return [load(data_file_handler.out_file('depth_uncertainty_'+str(x))) for x in range(20)]

	@classmethod
	def load_dic_data(self):
		data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+self.label+'/'+str(self.lifetime))
		return [load(data_file_handler.out_file('dic_uncertainty_'+str(x))) for x in range(20)]

	@classmethod
	def load_float_locations(self):
		data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+self.label+'/'+str(self.lifetime))
		filename = data_file_handler.out_file('bgc_array')
		bgc_list = load(filename)
		filename = data_file_handler.out_file('core_array')
		core_list = load(filename)
		return (core_list,bgc_list)

class USCoreOnly(FutureArgoArray):
	label = 'US_core_only'

	@classmethod
	def save_preprocessed(cls,trans_list):
		array = FutureArgoArray.load_preprocessed()
		recent_array = array.return_time_array(datetime.datetime(2025,6,10))
		array_scenario = FutureArgoArray({**Int20sArray(array).dict,**USA20sArray(array,core_only = True).dict})
		full_array = FutureArgoArray({**recent_array,**array_scenario})
		full_array.advance_floats(trans_list)
		with open(cls.make_filename(), 'wb') as pickle_file:
			pickle.dump(full_array,pickle_file)
		pickle_file.close()

class USCoreOnly_5(USCoreOnly):
	lifetime = 5
	linetype = 'r:'
	long_name = 'US Core - 5 Yr'

class USCoreOnly_7(USCoreOnly):
	lifetime = 7
	linetype = 'r--'
	long_name = 'US Core - 7 Yr'

class IntOnly(FutureArgoArray):
	label = 'no_US'
	@classmethod
	def save_preprocessed(cls,trans_list):
		array = FutureArgoArray.load_preprocessed()
		recent_array = array.return_time_array(datetime.datetime(2025,6,10))
		array_scenario = FutureArgoArray(Int20sArray(array).dict)
		full_array = FutureArgoArray({**recent_array,**array_scenario})
		full_array.advance_floats(trans_list)
		with open(cls.make_filename(), 'wb') as pickle_file:
			pickle.dump(full_array,pickle_file)
		pickle_file.close()

class IntOnly_5(IntOnly):
	lifetime=5
	linetype = 'b:'
	long_name = 'No US - 5 Yr'

class IntOnly_7(IntOnly):
	lifetime=7
	linetype = 'b--'
	long_name = 'No US - 7 Yr'

class NothingAdded(FutureArgoArray):
	label = 'no_added'
	@classmethod
	def save_preprocessed(cls,trans_list):
		array = FutureArgoArray.load_preprocessed()
		full_array = array.return_time_array(datetime.datetime(2025,6,10))
		full_array.advance_floats(trans_list)
		with open(cls.make_filename(), 'wb') as pickle_file:
			pickle.dump(full_array,pickle_file)
		pickle_file.close()

class NothingAdded_5(NothingAdded):
	lifetime=5
	linetype = 'y:'
	long_name = 'No Added - 5 Yr'

class NothingAdded_7(NothingAdded):
	lifetime=7	
	linetype = 'y--'
	long_name = 'No Added - 7 Yr'


def load_or_calculate_phat(filename,cov_holder,future_H):
	try:
		out= load(filename)
		print('loaded data')
	except FileNotFoundError:
		print('could not load, calculating...')
		p_hat_array = make_P_hat(cov_holder.cov,future_H,noise_factor=2)
		out = np.split(p_hat_array.diagonal(),len(cov_holder.trans_geo.variable_list))
		save(filename,out)
		del p_hat_array
	lat_core,lon_core = future_H.return_pos_of_core().lats_lons()
	lat_bgc,lon_bgc = future_H.return_pos_of_bgc().lats_lons()
	return (out,lat_core,lon_core,lat_bgc,lon_bgc)

def mixed_layer_uncertainty_calc(array_scenario):
	data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+array_scenario.label+'/'+str(array_scenario.lifetime))
	plot_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+array_scenario.label+'/'+str(array_scenario.lifetime)+'/plots')
	cov_holder_surf = CovLowCM4Global.load(depth_idx = 0)
	temp_uncert_surf = np.sqrt(cov_holder_surf.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_var(depth_idx=0)[::2,::2]))
	temp_mean_surf = cov_holder_surf.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_mean(depth_idx=0)[::2,::2])
	sal_uncert_surf = np.sqrt(cov_holder_surf.trans_geo.plottable_to_transition_vector(CM4Sal().return_var(depth_idx=0)[::2,::2]))
	sal_mean_surf = cov_holder_surf.trans_geo.plottable_to_transition_vector(CM4Sal().return_mean(depth_idx=0)[::2,::2])
	dummy,drho_ds_surf,drho_dt_surf = zip(*[gsw.density.rho_alpha_beta(sal,temp,0) for sal,temp in zip(sal_mean_surf,temp_mean_surf)])

	cov_holder_depth = CovLowCM4Global.load(depth_idx = 4)
	temp_uncert_depth = np.sqrt(cov_holder_surf.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_var(depth_idx=4)[::2,::2]))
	temp_mean_depth = cov_holder_depth.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_mean(depth_idx=4)[::2,::2])
	sal_uncert_depth = np.sqrt(cov_holder_surf.trans_geo.plottable_to_transition_vector(CM4Sal().return_var(depth_idx=4)[::2,::2]))
	sal_mean_depth = cov_holder_depth.trans_geo.plottable_to_transition_vector(CM4Sal().return_mean(depth_idx=4)[::2,::2])
	dummy,drho_ds_depth,drho_dt_depth = zip(*[gsw.density.rho_alpha_beta(sal,temp,50) for sal,temp in zip(sal_mean_depth,temp_mean_depth)])


	future_H_list = array_scenario.load_H('shallow')

	surface_density_dict = {'surf':[],'deep':[]}
	BGC_dict = {'surf':[],'deep':[]}
	Core_dict = {'surf':[],'deep':[]}

	for label,cov_holder,temp_uncert,temp_mean,sal_uncert,sal_mean,drho_ds,drho_dt in [
	('surf',cov_holder_surf,temp_uncert_surf,temp_mean_surf,sal_uncert_surf,sal_mean_surf,drho_ds_surf,drho_dt_surf),
	('deep',cov_holder_depth,temp_uncert_depth,temp_mean_depth,sal_uncert_depth,sal_mean_depth,drho_ds_depth,drho_dt_depth)]:

		for k,future_H in enumerate(future_H_list):
			filename = data_file_handler.tmp_file(str(cov_holder.trans_geo.depth_idx)+'_'+str(k))
			out,lat_core,lon_core,lat_bgc,lon_bgc = load_or_calculate_phat(filename,cov_holder,future_H)


			fig = plt.figure(figsize=(14,11))
			ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
			plot_holder = GlobalCartopy(ax=ax,adjustable=True)
			XX,YY,ax0 = plot_holder.get_map()
			XX,YY = future_H.trans_geo.get_coords()	

			var_list = cov_holder.trans_geo.variable_list
			t_index = var_list.index('thetao')
			s_index = var_list.index('so')
			cov_list = np.split(cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list))

			t_mapping_error = out[t_index]/cov_list[t_index]
			sigma_t = t_mapping_error*temp_uncert
			s_mapping_error = out[s_index]/cov_list[s_index]
			sigma_s = s_mapping_error*sal_uncert

			rho_uncert = np.sqrt((drho_dt*sigma_t)**2+(drho_ds*sigma_s)**2)
			surface_density_dict[label].append(rho_uncert)
			BGC_dict[label].append((lat_core,lon_core))
			Core_dict[label].append((lat_bgc,lon_bgc))

			data = future_H.trans_geo.transition_vector_to_plottable(rho_uncert)
			ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
			pcm = ax0.pcolormesh(XX,YY,data,cmap='YlOrBr',vmin=0,vmax=0.0025,transform=ccrs.PlateCarree())
			ax0.scatter(lon_core,lat_core,c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
			ax0.scatter(lon_bgc,lat_bgc,c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())

			plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
		              ncol=3, mode="expand", borderaxespad=0.)
			fig.colorbar(pcm,pad=-0.05,label='Density Uncertainty $(kg/m^3)$',location='bottom')
			make_folder_if_does_not_exist(plot_handler.tmp_file('density'))
			plt.savefig(plot_handler.tmp_file('density'+'/'+label+'_'+str(k)),bbox_inches='tight')
			plt.close()

	dates = []
	for x in range(20):
		dates.append(datetime.datetime(2025,6,10)+datetime.timedelta(days = 90*(1+x)))
		surf = surface_density_dict['surf'][x]
		deep = surface_density_dict['deep'][x]
		drho_dz=0.02/50.
		depth_uncertainty = 1/drho_dz*np.sqrt(surf**2+deep**2)
		filename = data_file_handler.out_file('depth_uncertainty_'+str(x))
		save(filename,depth_uncertainty)
		data = future_H.trans_geo.transition_vector_to_plottable(depth_uncertainty.tolist())

		fig = plt.figure(figsize=(14,11))
		ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
		plot_holder = GlobalCartopy(ax=ax,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		XX,YY = future_H.trans_geo.get_coords()	
		ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
		pcm = ax0.pcolormesh(XX,YY,data,cmap='YlOrBr',vmin = 0, vmax=10,transform=ccrs.PlateCarree())
		ax0.scatter(BGC_dict['surf'][x][1],BGC_dict['surf'][x][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
		ax0.scatter(Core_dict['surf'][x][1],Core_dict['surf'][x][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
		fig.colorbar(pcm,pad=-0.05,label='Depth Uncertainty (m)',location='bottom')
		make_folder_if_does_not_exist(plot_handler.tmp_file('ml_depth'))
		plt.savefig(plot_handler.tmp_file('ml_depth'+'/'+str(x)),bbox_inches='tight')
		plt.close()

def heat_content_uncertainty_calc(array_scenario):

	data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+array_scenario.label+'/'+str(array_scenario.lifetime))
	plot_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+array_scenario.label+'/'+str(array_scenario.lifetime)+'/plots')

	cov_holder_surf = CovLowCM4Global.load(depth_idx = 0)
	temp_uncert_surf = np.sqrt(cov_holder_surf.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_var(depth_idx=0)[::2,::2]))
	temp_mean_surf = cov_holder_surf.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_mean(depth_idx=0)[::2,::2])
	sal_uncert_surf = np.sqrt(cov_holder_surf.trans_geo.plottable_to_transition_vector(CM4Sal().return_var(depth_idx=0)[::2,::2]))
	sal_mean_surf = cov_holder_surf.trans_geo.plottable_to_transition_vector(CM4Sal().return_mean(depth_idx=0)[::2,::2])
	surf_depth = 80 - cov_holder_surf.get_depths()[0][0]

	cov_holder_depth_4 = CovLowCM4Global.load(depth_idx = 4)
	temp_uncert_depth_4 = np.sqrt(cov_holder_depth_4.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_var(depth_idx=4)[::2,::2]))
	temp_mean_depth_4 = cov_holder_depth_4.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_mean(depth_idx=4)[::2,::2])
	sal_uncert_depth_4 = np.sqrt(cov_holder_depth_4.trans_geo.plottable_to_transition_vector(CM4Sal().return_var(depth_idx=4)[::2,::2]))
	sal_mean_depth_4 = cov_holder_depth_4.trans_geo.plottable_to_transition_vector(CM4Sal().return_mean(depth_idx=4)[::2,::2])
	depth_4 = 200 - 80

	cov_holder_depth_14 = CovLowCM4Global.load(depth_idx = 14)
	temp_uncert_depth_14 = np.sqrt(cov_holder_depth_14.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_var(depth_idx=14)[::2,::2]))
	temp_mean_depth_14 = cov_holder_depth_14.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_mean(depth_idx=14)[::2,::2])
	sal_uncert_depth_14 = np.sqrt(cov_holder_depth_14.trans_geo.plottable_to_transition_vector(CM4Sal().return_var(depth_idx=14)[::2,::2]))
	sal_mean_depth_14 = cov_holder_depth_14.trans_geo.plottable_to_transition_vector(CM4Sal().return_mean(depth_idx=14)[::2,::2])
	depth_14 = 700 - 200


	future_H_list_shallow = array_scenario.load_H('shallow')
	future_H_list_deep = array_scenario.load_H('deep')

	Q_dict = {'surf':[],'4':[],'14':[]}
	BGC_dict = {'surf':[],'4':[],'14':[]}
	Core_dict = {'surf':[],'4':[],'14':[]}

	XX,YY = GlobalCartopy().meshgrid_xx_yy()
	plt.close('all')
	lats = cov_holder_surf.trans_geo.plottable_to_transition_vector(YY)
	area_in_grid = 4*np.cos(np.pi/180.*lats)*111132**2 

	for label,depth,cov_holder,temp_uncert,temp_mean,sal_uncert,sal_mean,thickness,future_H_list in [
	('surf',0,cov_holder_surf,temp_uncert_surf,temp_mean_surf,sal_uncert_surf,sal_mean_surf,surf_depth,future_H_list_shallow),
	('4',40,cov_holder_depth_4,temp_uncert_depth_4,temp_mean_depth_4,sal_uncert_depth_4,sal_mean_depth_4,depth_4,future_H_list_shallow),
	('14',550,cov_holder_depth_14,temp_uncert_depth_14,temp_mean_depth_14,sal_uncert_depth_14,sal_mean_depth_14,depth_14,future_H_list_deep),	
	]:
		rho = gsw.density.rho(sal_mean,temp_mean,[depth]*len(sal_mean))

		for k,future_H in enumerate(future_H_list):
			filename = data_file_handler.tmp_file(str(cov_holder.trans_geo.depth_idx)+'_'+str(k))
			out,lat_core,lon_core,lat_bgc,lon_bgc = load_or_calculate_phat(filename,cov_holder,future_H)


			fig = plt.figure(figsize=(14,11))
			ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
			plot_holder = GlobalCartopy(ax=ax,adjustable=True)
			XX,YY,ax0 = plot_holder.get_map()
			XX,YY = future_H.trans_geo.get_coords()	

			var_list = cov_holder.trans_geo.variable_list
			t_index = var_list.index('thetao')
			s_index = var_list.index('so')
			cov_list = np.split(cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list))

			t_mapping_error = out[t_index]/cov_list[t_index]
			sigma_t = t_mapping_error*temp_uncert
			# s_mapping_error = out[s_index]/cov_list[s_index]
			# sigma_s = s_mapping_error*sal_uncert

			# dummy,drho_ds,drho_dt = zip(*[gsw.density.rho_alpha_beta(sal_mean,CT,[depth]*len(sal_mean))])

			# dh_dt = (gsw.energy.enthalpy(sal_mean,temp_mean+0.001,[depth]*len(sal_mean))-gsw.energy.enthalpy(sal_mean,temp_mean,[depth]*len(sal_mean)))/0.001
			# # dh_ds = (gsw.energy.enthalpy(sal_mean+0.001,temp_mean,[depth]*len(sal_mean))-gsw.energy.enthalpy(sal_mean,temp_mean,[depth]*len(sal_mean)))/0.001
			# h = gsw.energy.enthalpy(sal_mean,temp_mean,[depth]*len(sal_mean))

			# dQ_dt = rho*dh_dt + h*drho_dt
			# dQ_ds = rho*dh_ds + h*drho_ds

			# dQ = (dQ_dt*sigma_t)*thickness*area_in_grid
			dQ = (3850*rho*sigma_t)*thickness*area_in_grid
			# dQ = (sigma_t)*thickness*area_in_grid

			Q_dict[label].append(dQ)
			Core_dict[label].append((lat_core,lon_core))
			BGC_dict[label].append((lat_bgc,lon_bgc))
			data = future_H.trans_geo.transition_vector_to_plottable(dQ.ravel())
			ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
			pcm = ax0.pcolormesh(XX,YY,data,cmap='YlOrBr',transform=ccrs.PlateCarree())
			ax0.scatter(lon_core,lat_core,c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
			ax0.scatter(lon_bgc,lat_bgc,c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())

			plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
		              ncol=3, mode="expand", borderaxespad=0.)
			fig.colorbar(pcm,pad=-0.05,label='Heat Uncertainty $(W/m^2)$',location='bottom')
			make_folder_if_does_not_exist(plot_handler.tmp_file('heat'))
			plt.savefig(plot_handler.tmp_file('heat'+'/'+label+'_'+str(k)),bbox_inches='tight')
			plt.close('all')


	line_plot_data = []	
	for x in range(20):
		future_H = future_H_list[x]
		dates.append(datetime.datetime(2025,6,10)+datetime.timedelta(days = 90*(1+x)))
		data = np.sqrt(sum([Q_dict[label][x]**2 for label in ['surf','4','14']]))
		filename = data_file_handler.out_file('heat_uncertainty_'+str(x))
		save(filename,data)

		line_plot_data.append(np.sqrt(sum([x**2 for x in data.ravel()])))
		# data = sum([Q_dict[label][x] for label in ['surf','4','14']])
		data = future_H.trans_geo.transition_vector_to_plottable(data.ravel())

		fig = plt.figure(figsize=(14,11))
		ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
		plot_holder = GlobalCartopy(ax=ax,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		XX,YY = future_H.trans_geo.get_coords()	
		ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
		pcm = ax0.pcolormesh(XX,YY,data,cmap='YlOrBr',transform=ccrs.PlateCarree())
		ax0.scatter(Core_dict['surf'][x][1],Core_dict['surf'][x][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
		ax0.scatter(BGC_dict['surf'][x][1],BGC_dict['surf'][x][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())

		fig.colorbar(pcm,pad=-0.05,label='Heat Uncertainty $J$',location='bottom')
		make_folder_if_does_not_exist(plot_handler.tmp_file('heat'))
		plt.savefig(plot_handler.tmp_file('heat'+'/'+str(x)),bbox_inches='tight')
		plt.close('all')

	heat_anomaly = 6.*10**21 # / year ... approximately, from https://www.climate.gov/news-features/understanding-climate/climate-change-ocean-heat-content
	plt.figure(figsize=(14,11))
	plt.plot(date_list,3*np.array(line_plot_data)/heat_anomaly)
	plt.ylabel('Years to Detect Anomalous Warming at 99% Confidence')
	plt.xlabel('Date')
	plt.savefig(plot_handler.tmp_file('heat/years'),bbox_inches='tight')
	plt.close('all')

def BGC_uncertainty_calc_surface(array_scenario):
	data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+array_scenario.label+'/'+str(array_scenario.lifetime))
	plot_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+array_scenario.label+'/'+str(array_scenario.lifetime)+'/plots')
	cov_holder = CovLowCM4Global.load(depth_idx = 0)
	future_H_list = array_scenario.load_H('shallow')

	ph_uncert = np.sqrt(cov_holder.trans_geo.plottable_to_transition_vector(CM4PH().return_var(depth_idx=0)[::2,::2]))
	ph_mean = cov_holder.trans_geo.plottable_to_transition_vector(CM4PH().return_mean(depth_idx=0)[::2,::2])
	po4_uncert = np.sqrt(cov_holder.trans_geo.plottable_to_transition_vector(CM4PO4().return_var(depth_idx=0)[::2,::2]))
	po4_mean = cov_holder.trans_geo.plottable_to_transition_vector(CM4PO4().return_mean(depth_idx=0)[::2,::2])
	chl_uncert = np.sqrt(cov_holder.trans_geo.plottable_to_transition_vector(CM4CHL().return_var(depth_idx=0)[::2,::2]))
	chl_mean = cov_holder.trans_geo.plottable_to_transition_vector(CM4CHL().return_mean(depth_idx=0)[::2,::2])
	o2_uncert = np.sqrt(cov_holder.trans_geo.plottable_to_transition_vector(CM4O2().return_var(depth_idx=0)[::2,::2]))
	o2_mean = cov_holder.trans_geo.plottable_to_transition_vector(CM4O2().return_mean(depth_idx=0)[::2,::2])

	XX,YY = GlobalCartopy().meshgrid_xx_yy()
	plt.close('all')
	lats = cov_holder.trans_geo.plottable_to_transition_vector(YY)

	mapping_error_list = []
	sigma_list = []
	BGCArgo_list = []
	Core_list = []

	data_dict = {'ph':[],'chl':[],'o2':[],'no3':[]}
	mapping_dict = {'ph':[],'chl':[],'o2':[],'no3':[]}


	for k,future_H in enumerate(future_H_list):
		filename = data_file_handler.tmp_file('0_'+str(k))
		out,lat_core,lon_core,lat_bgc,lon_bgc = load_or_calculate_phat(filename,cov_holder,future_H)

		Core_list.append((lat_core,lon_core))
		BGCArgo_list.append((lat_bgc,lon_bgc))

		var_list = cov_holder.trans_geo.variable_list
		ph_index = var_list.index('ph')
		o2_index = var_list.index('o2')
		po4_index = var_list.index('po4')
		chl_index = var_list.index('chl')
		
		cov_list = np.split(cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list))

		ph_mapping_error = out[ph_index]/cov_list[ph_index]
		sigma_ph = ph_mapping_error*ph_uncert
		data_dict['ph'].append(sigma_ph)
		mapping_dict['ph'].append(ph_mapping_error)

		o2_mapping_error = out[o2_index]/cov_list[o2_index]
		sigma_o2 = o2_mapping_error*o2_uncert
		data_dict['o2'].append(sigma_o2)
		mapping_dict['o2'].append(ph_mapping_error)

		po4_mapping_error = out[po4_index]/cov_list[po4_index]
		sigma_N = 16*po4_mapping_error*po4_uncert
		data_dict['no3'].append(sigma_N)
		mapping_dict['no3'].append(po4_mapping_error)

		chl_mapping_error = out[chl_index]/cov_list[chl_index]
		sigma_chl = chl_mapping_error*chl_uncert
		data_dict['chl'].append(sigma_chl)
		mapping_dict['chl'].append(chl_mapping_error)


	label_uncertainty_dict = {'ph':'pH Uncertainty Increase','chl':'Chlorophyll Uncertainty Increase ($kg~m^{-3}$)','o2':'Oxygen Uncertainty Increase ($mol~m^{-3}$)','no3':'Nitrate Uncertainty Increase ($mol~m^{-3}$)'}
	colorbar_dict = {'ph':0.04,'chl':4*10**-7,'o2':0.0175,'no3':0.003}
	dates = []
	base_dict = {}	
	plot_data = {'ph':[],'chl':[],'o2':[],'no3':[]}
	for x in range(20):
		dates.append(datetime.datetime(2025,6,10)+datetime.timedelta(days = 90*(1+x)))
		for var in ['ph','chl','o2','no3']:
			data = data_dict[var][x]
			if x==0:
				base_dict[var] = data
			filename = data_file_handler.out_file(var+'_surfaceuncertainty_'+str(x))
			save(filename,data)
			plot_data[var].append(np.mean(data-base_dict[var]))
			data = future_H.trans_geo.transition_vector_to_plottable((data-base_dict[var]).tolist())

			fig = plt.figure(figsize=(14,11))
			ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
			plot_holder = GlobalCartopy(ax=ax,adjustable=True)
			XX,YY,ax0 = plot_holder.get_map()
			XX,YY = future_H.trans_geo.get_coords()	


			ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
			pcm = ax0.pcolormesh(XX,YY,data,vmin=0,cmap='YlOrBr',transform=ccrs.PlateCarree())
			ax0.scatter(Core_list[x][1],Core_list[x][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
			ax0.scatter(BGCArgo_list[x][1],BGCArgo_list[x][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())

			fig.colorbar(pcm,pad=-0.05,label=label_uncertainty_dict[var],location='bottom')
			make_folder_if_does_not_exist(plot_handler.tmp_file(var))
			plt.savefig(plot_handler.tmp_file(var+'/'+'surface'+var+'_'+str(x)),bbox_inches='tight')
			plt.close('all')

	for var in ['ph','o2','no3','chl']:
		data = data_dict[var]
		plt.figure(figsize=(14,11))
		if var == 'ph':
			rate = 0.06/36
			plt.plot(date_list,3*np.array(data)/rate)
			plt.ylabel('Additional Years of Observation')
		elif var == 'o2':
			rate = o2_mean.mean()*0.02/60
			plt.plot(date_list,3*np.array(data)/rate)
			plt.ylabel('Additional Years of Observation')

		else:
			plt.plot(date_list,data)
			plt.ylabel('Mean '+label_uncertainty_dict[var])
		plt.xlabel('Date')
		plt.savefig(plot_handler.tmp_file(var+'/surface_years'),bbox_inches='tight')
		plt.close('all')

def BGC_uncertainty_int_calc(array_scenario):
	data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+array_scenario.label+'/'+str(array_scenario.lifetime))
	plot_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+array_scenario.label+'/'+str(array_scenario.lifetime)+'/plots')
	cov_holder = CovCM4HighCorrelation.load()
	future_H_list = array_scenario.load_H('int')

	ph_uncert = np.sqrt(cov_holder.trans_geo.plottable_to_transition_vector(CM4PH().return_int_var()[::2,::2]))
	ph_mean = cov_holder.trans_geo.plottable_to_transition_vector(CM4PH().return_int_mean()[::2,::2])
	po4_uncert = np.sqrt(cov_holder.trans_geo.plottable_to_transition_vector(CM4PO4().return_int_var()[::2,::2]))
	po4_mean = cov_holder.trans_geo.plottable_to_transition_vector(CM4PO4().return_int_mean()[::2,::2])
	chl_uncert = np.sqrt(cov_holder.trans_geo.plottable_to_transition_vector(CM4CHL().return_int_var()[::2,::2]))
	chl_mean = cov_holder.trans_geo.plottable_to_transition_vector(CM4CHL().return_int_mean()[::2,::2])
	o2_uncert = np.sqrt(cov_holder.trans_geo.plottable_to_transition_vector(CM4O2().return_int_var()[::2,::2]))
	o2_mean = cov_holder.trans_geo.plottable_to_transition_vector(CM4O2().return_int_mean()[::2,::2])

	XX,YY = GlobalCartopy().meshgrid_xx_yy()
	plt.close('all')
	lats = cov_holder.trans_geo.plottable_to_transition_vector(YY)

	mapping_error_list = []
	sigma_list = []
	BGCArgo_list = []
	Core_list = []

	data_dict = {'ph':[],'chl':[],'o2':[],'no3':[]}
	mapping_dict = {'ph':[],'chl':[],'o2':[],'no3':[]}


	for k,future_H in enumerate(future_H_list):
		filename = data_file_handler.tmp_file('int_'+str(k))
		out,lat_core,lon_core,lat_bgc,lon_bgc = load_or_calculate_phat(filename,cov_holder,future_H)

		Core_list.append((lat_core,lon_core))
		BGCArgo_list.append((lat_bgc,lon_bgc))

		var_list = cov_holder.trans_geo.variable_list
		ph_index = var_list.index('ph')
		o2_index = var_list.index('o2')
		po4_index = var_list.index('po4')
		chl_index = var_list.index('chl')
		
		cov_list = np.split(cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list))

		ph_mapping_error = out[ph_index]/cov_list[ph_index]
		sigma_ph = ph_mapping_error*ph_uncert
		data_dict['ph'].append(sigma_ph)
		mapping_dict['ph'].append(ph_mapping_error)

		o2_mapping_error = out[o2_index]/cov_list[o2_index]
		sigma_o2 = o2_mapping_error*o2_uncert
		data_dict['o2'].append(sigma_o2)
		mapping_dict['o2'].append(ph_mapping_error)

		po4_mapping_error = out[po4_index]/cov_list[po4_index]
		sigma_N = 16*po4_mapping_error*po4_uncert
		data_dict['no3'].append(sigma_N)
		mapping_dict['no3'].append(po4_mapping_error)

		chl_mapping_error = out[chl_index]/cov_list[chl_index]
		sigma_chl = chl_mapping_error*chl_uncert
		data_dict['chl'].append(sigma_chl)
		mapping_dict['chl'].append(chl_mapping_error)


	label_uncertainty_dict = {'ph':'pH Uncertainty Increase','chl':'Chlorophyll Uncertainty Increase ($kg~m^{-3}$)','o2':'Oxygen Uncertainty Increase ($mol~m^{-3}$)','no3':'Nitrate Uncertainty Increase ($mol~m^{-3}$)'}
	colorbar_dict = {'ph':0.04,'chl':4*10**-7,'o2':0.0175,'no3':0.003}
	dates = []
	base_dict = {}	
	plot_data = {'ph':[],'chl':[],'o2':[],'no3':[]}
	for x in range(20):
		dates.append(datetime.datetime(2025,6,10)+datetime.timedelta(days = 90*(1+x)))
		for var in ['ph','chl','o2','no3']:
			data = data_dict[var][x]
			if x==0:
				base_dict[var] = data
			filename = data_file_handler.out_file(var+'_uncertainty_'+str(x))
			save(filename,data)
			plot_data[var].append(np.mean(data-base_dict[var]))
			data = future_H.trans_geo.transition_vector_to_plottable((data-base_dict[var]).tolist())

			fig = plt.figure(figsize=(14,11))
			ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
			plot_holder = GlobalCartopy(ax=ax,adjustable=True)
			XX,YY,ax0 = plot_holder.get_map()
			XX,YY = future_H.trans_geo.get_coords()	


			ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
			pcm = ax0.pcolormesh(XX,YY,data,vmin=0,cmap='YlOrBr',transform=ccrs.PlateCarree())
			ax0.scatter(Core_list[x][1],Core_list[x][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
			ax0.scatter(BGCArgo_list[x][1],BGCArgo_list[x][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())

			fig.colorbar(pcm,pad=-0.05,label=label_uncertainty_dict[var],location='bottom')
			make_folder_if_does_not_exist(plot_handler.tmp_file(var))
			plt.savefig(plot_handler.tmp_file(var+'/'+var+'_'+str(x)),bbox_inches='tight')
			plt.close('all')

	for var in ['ph','o2','no3','chl']:
		data = data_dict[var]
		plt.figure(figsize=(14,11))
		if var == 'ph':
			rate = 0.06/36
			plt.plot(date_list,3*np.array(data)/rate)
			plt.ylabel('Additional Years of Observation')
		elif var == 'o2':
			rate = o2_mean.mean()*0.02/60
			plt.plot(date_list,3*np.array(data)/rate)
			plt.ylabel('Additional Years of Observation')

		else:
			plt.plot(date_list,data)
			plt.ylabel('Mean '+label_uncertainty_dict[var])
		plt.xlabel('Date')
		plt.savefig(plot_handler.tmp_file(var+'/years'),bbox_inches='tight')
		plt.close('all')

def steric_uncertainty_calc(array_scenario):
	data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+array_scenario.label+'/'+str(array_scenario.lifetime))
	plot_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+array_scenario.label+'/'+str(array_scenario.lifetime)+'/plots')

	cov_holder_surf = CovLowCM4Global.load(depth_idx = 0)
	temp_uncert_surf = np.sqrt(cov_holder_surf.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_var(depth_idx=0)[::2,::2]))
	temp_mean_surf = cov_holder_surf.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_mean(depth_idx=0)[::2,::2])
	sal_uncert_surf = np.sqrt(cov_holder_surf.trans_geo.plottable_to_transition_vector(CM4Sal().return_var(depth_idx=0)[::2,::2]))
	sal_mean_surf = cov_holder_surf.trans_geo.plottable_to_transition_vector(CM4Sal().return_mean(depth_idx=0)[::2,::2])
	surf_depth = 80 - cov_holder_surf.get_depths()[0][0]
	drho_dt_surf = [gsw.density.alpha(sal,temp,0) for sal,temp in zip(sal_mean_surf,temp_mean_surf)]


	cov_holder_depth_4 = CovLowCM4Global.load(depth_idx = 4)
	temp_uncert_depth_4 = np.sqrt(cov_holder_depth_4.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_var(depth_idx=4)[::2,::2]))
	temp_mean_depth_4 = cov_holder_depth_4.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_mean(depth_idx=4)[::2,::2])
	sal_uncert_depth_4 = np.sqrt(cov_holder_depth_4.trans_geo.plottable_to_transition_vector(CM4Sal().return_var(depth_idx=4)[::2,::2]))
	sal_mean_depth_4 = cov_holder_depth_4.trans_geo.plottable_to_transition_vector(CM4Sal().return_mean(depth_idx=4)[::2,::2])
	depth_4 = 200 - 80
	drho_dt_depth_4 = [gsw.density.alpha(sal,temp,80) for sal,temp in zip(sal_mean_depth_4,temp_mean_depth_4)]


	cov_holder_depth_14 = CovLowCM4Global.load(depth_idx = 14)
	temp_uncert_depth_14 = np.sqrt(cov_holder_depth_14.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_var(depth_idx=14)[::2,::2]))
	temp_mean_depth_14 = cov_holder_depth_14.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_mean(depth_idx=14)[::2,::2])
	sal_uncert_depth_14 = np.sqrt(cov_holder_depth_14.trans_geo.plottable_to_transition_vector(CM4Sal().return_var(depth_idx=14)[::2,::2]))
	sal_mean_depth_14 = cov_holder_depth_14.trans_geo.plottable_to_transition_vector(CM4Sal().return_mean(depth_idx=14)[::2,::2])
	depth_14 = 700 - 200
	drho_dt_depth_14 = [gsw.density.alpha(sal,temp,550) for sal,temp in zip(sal_mean_depth_14,temp_mean_depth_14)]

	cov_holder_depth_18 = CovLowCM4Global.load(depth_idx = 18)
	temp_uncert_depth_18 = np.sqrt(cov_holder_depth_18.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_var(depth_idx=18)[::2,::2]))
	temp_mean_depth_18 = cov_holder_depth_18.trans_geo.plottable_to_transition_vector(CM4ThetaO().return_mean(depth_idx=18)[::2,::2])
	sal_uncert_depth_18 = np.sqrt(cov_holder_depth_18.trans_geo.plottable_to_transition_vector(CM4Sal().return_var(depth_idx=18)[::2,::2]))
	sal_mean_depth_18 = cov_holder_depth_18.trans_geo.plottable_to_transition_vector(CM4Sal().return_mean(depth_idx=18)[::2,::2])
	depth_18 = 2000 - 700
	drho_dt_depth_18 = [gsw.density.alpha(sal,temp,950) for sal,temp in zip(sal_mean_depth_18,temp_mean_depth_18)]

	future_H_list_shallow = array_scenario.load_H('shallow')
	future_H_list_deep = array_scenario.load_H('deep')


	BGC_dict = {'surf':[],'4':[],'14':[],'18':[]}
	Core_dict = {'surf':[],'4':[],'14':[],'18':[]}
	volume_dict = {'surf':[],'4':[],'14':[],'18':[]}

	for label,depth,cov_holder,temp_uncert,temp_mean,sal_uncert,sal_mean,thickness,future_H_list,alpha in [
	('surf',0,cov_holder_surf,temp_uncert_surf,temp_mean_surf,sal_uncert_surf,sal_mean_surf,surf_depth,future_H_list_shallow,drho_dt_surf),
	('4',40,cov_holder_depth_4,temp_uncert_depth_4,temp_mean_depth_4,sal_uncert_depth_4,sal_mean_depth_4,depth_4,future_H_list_shallow,drho_dt_depth_4),
	('14',550,cov_holder_depth_14,temp_uncert_depth_14,temp_mean_depth_14,sal_uncert_depth_14,sal_mean_depth_14,depth_14,future_H_list_deep,drho_dt_depth_14),
	('18',950,cov_holder_depth_18,temp_uncert_depth_18,temp_mean_depth_18,sal_uncert_depth_18,sal_mean_depth_18,depth_18,future_H_list_deep,drho_dt_depth_18),	
	]:
		for k,future_H in enumerate(future_H_list):
			filename = data_file_handler.tmp_file(str(cov_holder.trans_geo.depth_idx)+'_'+str(k))
			out,lat_core,lon_core,lat_bgc,lon_bgc = load_or_calculate_phat(filename,cov_holder,future_H)

			fig = plt.figure(figsize=(14,11))
			ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
			plot_holder = GlobalCartopy(ax=ax,adjustable=True)
			XX,YY,ax0 = plot_holder.get_map()
			XX,YY = future_H.trans_geo.get_coords()	

			var_list = cov_holder.trans_geo.variable_list
			t_index = var_list.index('thetao')
			s_index = var_list.index('so')
			cov_list = np.split(cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list))

			t_mapping_error = out[t_index]/cov_list[t_index]
			sigma_t = t_mapping_error*temp_uncert

			thermal_coefficient = alpha*sigma_t
			volume_dict[label].append(thermal_coefficient*thickness)
			BGC_dict[label].append((lat_bgc,lon_bgc))
			Core_dict[label].append((lat_core,lon_core))

			data = future_H.trans_geo.transition_vector_to_plottable(1/rho_uncert)
			ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
			pcm = ax0.pcolormesh(XX,YY,data,cmap='YlOrBr',vmin=0,vmax=0.0025,transform=ccrs.PlateCarree())
			ax0.scatter(lon_core,lat_core,c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
			ax0.scatter(lon_bgc,lat_bgc,c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())

			plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
		              ncol=3, mode="expand", borderaxespad=0.)
			fig.colorbar(pcm,pad=-0.05,label='Density Uncertainty $(kg/m^3)$',location='bottom')
			make_folder_if_does_not_exist(plot_handler.tmp_file('sea_level'))
			plt.savefig(plot_handler.tmp_file('sea_level'+'/'+label+'_'+str(k)),bbox_inches='tight')
			plt.close()

	dates = []		
	data_list = []
	for x in range(20):
		data = []
		dates.append(datetime.datetime(2025,6,10)+datetime.timedelta(days = 90*(1+x)))
		for elements in zip(*[volume_dict[label][x] for label in ['surf','4','14','18']]):
			data.append(np.sqrt(sum([element**2 for element in elements])))
		data_list.append(np.mean(data))
		filename = data_file_handler.out_file('sealevel_'+str(x))
		save(filename,data)
		filename = data_file_handler.out_file('bgc_array')
		save(filename,BGC_dict['surf'])
		filename = data_file_handler.out_file('core_array')
		save(filename,Core_dict['surf'])

		data = future_H.trans_geo.transition_vector_to_plottable(data)
		fig = plt.figure(figsize=(14,11))
		ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
		plot_holder = GlobalCartopy(ax=ax,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		XX,YY = future_H.trans_geo.get_coords()	


		ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
		pcm = ax0.pcolormesh(XX,YY,np.array(data)*100,cmap='YlOrBr',vmin=0,vmax=10,transform=ccrs.PlateCarree())
		ax0.scatter(Core_dict['surf'][x][1],Core_dict['surf'][x][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
		ax0.scatter(BGC_dict['surf'][x][1],BGC_dict['surf'][x][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())

		fig.colorbar(pcm,pad=-0.05,label='Steric Uncertainty (cm)',location='bottom')
		make_folder_if_does_not_exist(plot_handler.tmp_file('sea_level'))
		plt.savefig(plot_handler.tmp_file('sea_level/'+str(x)),bbox_inches='tight')
		plt.close('all')

	plt.figure(figsize=(14,11))
	plt.plot(date_list,np.array(data_list)*100)
	plt.ylabel('Average Steric Uncertainty (cm)')
	plt.xlabel('Date')
	plt.savefig(plot_handler.tmp_file('sea_level_years'),bbox_inches='tight')
	plt.close('all')

def DIC_uncertainty_calc(array_scenario,cov_class=CovCM4HighCorrelation):
	data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+array_scenario.label+'/'+str(array_scenario.lifetime))
	plot_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/'+array_scenario.label+'/'+str(array_scenario.lifetime)+'/plots')

	cov_holder = cov_class.load()
	future_H_list = array_scenario.load_H('int')

	dic_uncert_int = np.sqrt(cov_holder.trans_geo.plottable_to_transition_vector(CM4DIC().return_int_var()[::2,::2]))
	dic_mean_int = cov_holder.trans_geo.plottable_to_transition_vector(CM4DIC().return_int_mean()[::2,::2])

	mapping_error_list = []
	sigma_list = []
	BGCArgo_list = []
	Core_list = []



	for k,future_H in enumerate(future_H_list):
		filename = data_file_handler.tmp_file('int_'+str(k))
		out,lat_core,lon_core,lat_bgc,lon_bgc = load_or_calculate_phat(filename,cov_holder,future_H)

		var_list = cov_holder.trans_geo.variable_list
		dic_index = var_list.index('dissic')

		
		cov_list = np.split(cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list))

		dic_mapping_error = out[dic_index]/cov_list[dic_index]
		mapping_error_list.append(dic_mapping_error)
		sigma_list.append(dic_mapping_error*dic_uncert_int)

		Core_list.append((lat_core,lon_core))
		BGCArgo_list.append((lat_bgc,lon_bgc))


	dates = []		
	data_list = []
	base_dict = {}
	for x in range(20):
		dates.append(datetime.datetime(2025,6,10)+datetime.timedelta(days = 90*(1+x)))
		data = sigma_list[x]
		data_list.append(np.linalg.norm(data))
		if x==0:
			base_data = data
		filename = data_file_handler.out_file('dic_uncertainty_'+str(x))
		save(filename,data)
		data = future_H.trans_geo.transition_vector_to_plottable((data-base_data).tolist())
		map_data = future_H.trans_geo.transition_vector_to_plottable(mapping_error_list[x].tolist())
		fig = plt.figure(figsize=(18,14))
		ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
		ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
		plot_holder = GlobalCartopy(ax=ax1,adjustable=True)
		XX,YY,ax1 = plot_holder.get_map()
		XX,YY = future_H.trans_geo.get_coords()	

		ax1.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
		cov_colorbar = ax1.pcolormesh(XX,YY,data,vmin=0,vmax = base_data.max()/40, cmap='YlOrBr',transform=ccrs.PlateCarree())
		ax1.scatter(Core_list[x][1],Core_list[x][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
		ax1.scatter(BGCArgo_list[x][1],BGCArgo_list[x][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())

		cbar1 = fig.colorbar(cov_colorbar, ax=ax1, orientation='vertical', pad=0.05, shrink=0.8)
		cbar1.set_label('$\Delta$ DIC Uncertainty (mol $m^{-2}$)')

		plot_holder = GlobalCartopy(ax=ax2,adjustable=True)
		XX,YY,ax2 = plot_holder.get_map()
		XX,YY = future_H.trans_geo.get_coords()	

		ax2.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
		mapping_cb = ax2.pcolormesh(XX,YY,map_data,vmin=0.5,vmax=1,transform=ccrs.PlateCarree())
		ax2.scatter(Core_list[x][1],Core_list[x][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
		ax2.scatter(BGCArgo_list[x][1],BGCArgo_list[x][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())

		cbar2 = fig.colorbar(mapping_cb, ax=ax2, orientation='vertical', pad=0.05, shrink=0.8)
		cbar2.set_label('Mapping Error')
		fig.suptitle(dates[-1])
		make_folder_if_does_not_exist(plot_handler.tmp_file('dic'))
		plt.savefig(plot_handler.tmp_file('dic/dic_'+str(x)),bbox_inches='tight')
		plt.close('all')


	plt.plot(date_list,data_list)
	plt.ylabel('Total DIC Uncertainty (mol $m^{-2}$)')
	plt.xlabel('Date')
	plt.savefig(plot_handler.tmp_file('dic/years'),bbox_inches='tight')
	plt.close('all')

	os.chdir(plot_handler.tmp_file('dic/'))
	os.system("ffmpeg -r 5 -i dic_%01d.png -vcodec mpeg4 -y movie.mp4")

def bgc_surface_plots():
	plot_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/plots')

	cov_holder = CovLowCM4Global.load(depth_idx = 0)
	o2_mean = cov_holder.trans_geo.plottable_to_transition_vector(CM4O2().return_mean(depth_idx=0)[::2,::2])
	dates = [datetime.datetime(2025,6,10)+datetime.timedelta(days = 90*(1+x)) for x in range(20)]	
	label_uncertainty_dict = {'ph':'$m~pH$','chl':'$mg~l^{-1}$','o2':'$m~mol~m^{-3}$','no3':'$m~mol~m^{-3}$'}
	unit_modifier_dict = {'ph':10**3,'chl':10**9,'o2':10**3,'no3':10**3}
	vmax_uncertainty_dict = {'ph':20,'chl':60,'o2':8,'no3':0.6}
	annotate_dict = {'ph':'a','chl':'b','o2':'c','no3':'d'}

	future_H = future_H_list[0]

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex=True,figsize=(16,10))
	axs_dict = {'ph':ax1,'o2':ax2,'no3':ax3,'chl':ax4}
	for scenario in [USCoreOnly_5,IntOnly_5,NothingAdded_5,USCoreOnly_7,IntOnly_7,NothingAdded_7]:
		for k,(var,data_list) in enumerate(scenario.load_bgc_surface_data()):
			print(var)
			print(data_list)
			data = [np.mean(dummy-data_list[0]) for dummy in data_list]
			# all of these plots, need to divide by 2000 for the integral

			if var == 'ph':
				rate = 0.07/36
				# axs_dict[var].plot(date_list,3*np.array(data)/2000/rate,scenario.linetype,lw=5,label=scenario.long_name,alpha=0.5)
				axs_dict[var].plot(date_list,3*np.array(data)/rate,scenario.linetype,lw=5,label=scenario.long_name,alpha=0.5)
				axs_dict[var].set_ylabel('years')
				axs_dict[var].annotate(annotate_dict[var], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

			elif var == 'o2':
				#base units = 'mol m-3'
				rate = o2_mean.mean()*0.02/60
				# axs_dict[var].plot(date_list,3*np.array(data)/rate,scenario.linetype,lw=5,label=scenario.long_name,alpha=0.5)
				axs_dict[var].plot(date_list,3*np.array(data)/rate,scenario.linetype,lw=5,label=scenario.long_name,alpha=0.5)
				axs_dict[var].set_ylabel('years')
				axs_dict[var].annotate(annotate_dict[var], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

			elif var == 'no3':
				#base units = 'mol m-3'
				axs_dict[var].plot(date_list,np.array(data)*10**3,scenario.linetype,lw=5,label=scenario.long_name,alpha=0.5)
				axs_dict[var].set_ylabel(label_uncertainty_dict[var])
				axs_dict[var].annotate(annotate_dict[var], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

			else:
				#base units = 'kg m-3'
				axs_dict[var].plot(date_list,np.array(data)*10**9,scenario.linetype,lw=5,label=scenario.long_name,alpha=0.5)
				axs_dict[var].set_ylabel(label_uncertainty_dict[var])
				axs_dict[var].annotate(annotate_dict[var], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	plt.legend(ncol=3,bbox_to_anchor=(-.9, 2.2, -.9, 2.2), loc=3)
	# plt.tight_layout(rect=[0, 0, 1, 0.7])
	plt.savefig(plot_handler.out_file('bgc_surface_plots_years'))
	plt.close('all')

	for var in ['ph','o2','no3','chl']:
		fig = plt.figure(figsize=(16,16))
		ax1 = fig.add_subplot(3,1,1, projection=ccrs.PlateCarree())
		ax2 = fig.add_subplot(3,1,2, projection=ccrs.PlateCarree())
		ax3 = fig.add_subplot(3,1,3, projection=ccrs.PlateCarree())
		for ax,scenario,annotate_label in [(ax1,USCoreOnly_5,'a'),(ax2,IntOnly_5,'b'),(ax3,NothingAdded_5,'c')]:
			core_list,bgc_list = scenario.load_float_locations()

			data_list = dict(scenario.load_bgc_surface_data())[var]
			data = ((data_list[19]-data_list[0])*unit_modifier_dict[var]).tolist()

			data = future_H.trans_geo.transition_vector_to_plottable(data)

			plot_holder = GlobalCartopy(ax=ax,adjustable=True)
			XX,YY,ax0 = plot_holder.get_map()
			XX,YY = future_H.trans_geo.get_coords()	
			ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
			pcm = ax0.pcolormesh(XX,YY,data,vmin=0,vmax=vmax_uncertainty_dict[var],cmap='YlOrBr',transform=ccrs.PlateCarree())
			ax0.scatter(core_list[19][1],core_list[19][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
			ax0.scatter(bgc_list[19][1],bgc_list[19][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
			ax0.annotate(annotate_label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
		fig.colorbar(pcm,ax=[ax1,ax2,ax3],label=label_uncertainty_dict[var],location='right',shrink=0.8)
		plt.savefig(plot_handler.out_file('bgc_plots_5_'+var))
		plt.close('all')

	for var in ['ph','o2','no3','chl']:
		fig = plt.figure(figsize=(16,16))
		ax1 = fig.add_subplot(3,1,1, projection=ccrs.PlateCarree())
		ax2 = fig.add_subplot(3,1,2, projection=ccrs.PlateCarree())
		ax3 = fig.add_subplot(3,1,3, projection=ccrs.PlateCarree())
		for ax,scenario,annotate_label in [(ax1,USCoreOnly_7,'a'),(ax2,IntOnly_7,'b'),(ax3,NothingAdded_7,'c')]:
			core_list,bgc_list = scenario.load_float_locations()

			data_list = dict(scenario.load_bgc_surface_data())[var]
			data = ((data_list[19]-data_list[0])*unit_modifier_dict[var]).tolist()
			data = future_H.trans_geo.transition_vector_to_plottable(data)

			plot_holder = GlobalCartopy(ax=ax,adjustable=True)
			XX,YY,ax0 = plot_holder.get_map()
			XX,YY = future_H.trans_geo.get_coords()	
			ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
			pcm = ax0.pcolormesh(XX,YY,data,vmin=0,vmax=vmax_uncertainty_dict[var],cmap='YlOrBr',transform=ccrs.PlateCarree())
			ax0.scatter(core_list[19][1],core_list[19][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
			ax0.scatter(bgc_list[19][1],bgc_list[19][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
			ax0.annotate(annotate_label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

		fig.colorbar(pcm,ax=[ax1,ax2,ax3],label=label_uncertainty_dict[var],location='right',shrink=0.8)
		plt.savefig(plot_handler.out_file('bgc_plots_7_'+var))
		plt.close('all')

def bgc_int_plots():
	plot_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/plots')

	dates = [datetime.datetime(2025,6,10)+datetime.timedelta(days = 90*(1+x)) for x in range(20)]	
	label_uncertainty_dict = {'ph':'$m~pH$','chl':'$mg~l^{-1}$','o2':'$m~mol~m^{-3}$','no3':'$\mu~mol~m^{-3}$'}
	vmax_uncertainty_dict = {'ph':1,'chl':30,'o2':30,'no3':500}
	annotate_dict = {'ph':'a','chl':'b','o2':'c','no3':'d'}

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex=True,figsize=(16,10))
	axs_dict = {'ph':ax1,'o2':ax2,'no3':ax3,'chl':ax4}
	for scenario in [USCoreOnly_5,IntOnly_5,NothingAdded_5,USCoreOnly_7,IntOnly_7,NothingAdded_7]:
		for k,(var,data_list) in enumerate(scenario.load_bgc_int_data()):
			data = [np.mean(dummy-data_list[0]) for dummy in data_list]
			# all of these plots, need to divide by 2000 for the integral

			if var == 'ph':
				# rate = 0.07/36
				# axs_dict[var].plot(date_list,3*np.array(data)/2000/rate,scenario.linetype,lw=5,label=scenario.long_name,alpha=0.5)
				axs_dict[var].plot(date_list,np.array(data)*1000/2000,scenario.linetype,lw=5,label=scenario.long_name,alpha=0.5)
				axs_dict[var].set_ylabel(label_uncertainty_dict[var])
				axs_dict[var].annotate(annotate_dict[var], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

			elif var == 'o2':
				#base units = 'mol m-3'
				# rate = o2_mean.mean()*0.02/60
				# axs_dict[var].plot(date_list,3*np.array(data)/rate,scenario.linetype,lw=5,label=scenario.long_name,alpha=0.5)
				axs_dict[var].plot(date_list,np.array(data)*1000/2000,scenario.linetype,lw=5,label=scenario.long_name,alpha=0.5)
				axs_dict[var].set_ylabel(label_uncertainty_dict[var])
				axs_dict[var].annotate(annotate_dict[var], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

			elif var == 'no3':
				#base units = 'mol m-3'
				axs_dict[var].plot(date_list,np.array(data)*10**6/2000,scenario.linetype,lw=5,label=scenario.long_name,alpha=0.5)
				axs_dict[var].set_ylabel(label_uncertainty_dict[var])
				axs_dict[var].annotate(annotate_dict[var], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

			else:
				#base units = 'kg m-3'
				axs_dict[var].plot(date_list,np.array(data)*10**9/200,scenario.linetype,lw=5,label=scenario.long_name,alpha=0.5)
				axs_dict[var].set_ylabel(label_uncertainty_dict[var])
				axs_dict[var].annotate(annotate_dict[var], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	plt.legend(ncol=3,bbox_to_anchor=(-.9, 2.2, -.9, 2.2), loc=3)
	# plt.tight_layout(rect=[0, 0, 1, 0.7])
	plt.savefig(plot_handler.out_file('bgc_plots_years'))
	plt.close('all')

	for var in ['ph','o2','no3','chl']:
		fig = plt.figure(figsize=(16,16))
		ax1 = fig.add_subplot(3,1,1, projection=ccrs.PlateCarree())
		ax2 = fig.add_subplot(3,1,2, projection=ccrs.PlateCarree())
		ax3 = fig.add_subplot(3,1,3, projection=ccrs.PlateCarree())
		for ax,scenario,annotate_label in [(ax1,USCoreOnly_5,'a'),(ax2,IntOnly_5,'b'),(ax3,NothingAdded_5,'c')]:
			core_list,bgc_list = scenario.load_float_locations()

			data_list = dict(scenario.load_bgc_int_data())[var]
			if var == 'chl':
				data = ((data_list[19]-data_list[0])*10**6).tolist()
			if var == 'ph':
				data = (data_list[19]-data_list[0]).tolist()
			if var == 'o2':
				data = ((data_list[19]-data_list[0])).tolist()
			if var == 'no3':
				data = ((data_list[19]-data_list[0])*1000).tolist()
			data = future_H.trans_geo.transition_vector_to_plottable(data)

			plot_holder = GlobalCartopy(ax=ax,adjustable=True)
			XX,YY,ax0 = plot_holder.get_map()
			XX,YY = future_H.trans_geo.get_coords()	
			ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
			pcm = ax0.pcolormesh(XX,YY,data,vmin=0,cmap='YlOrBr',transform=ccrs.PlateCarree())
			ax0.scatter(core_list[19][1],core_list[19][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
			ax0.scatter(bgc_list[19][1],bgc_list[19][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
			ax0.annotate(annotate_label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
		fig.colorbar(pcm,ax=[ax1,ax2,ax3],label=label_uncertainty_dict[var],location='right',shrink=0.8)
		plt.savefig(plot_handler.out_file('bgc_plots_5_'+var))
		plt.close('all')

	for var in ['ph','o2','no3','chl']:
		fig = plt.figure(figsize=(16,16))
		ax1 = fig.add_subplot(3,1,1, projection=ccrs.PlateCarree())
		ax2 = fig.add_subplot(3,1,2, projection=ccrs.PlateCarree())
		ax3 = fig.add_subplot(3,1,3, projection=ccrs.PlateCarree())
		for ax,scenario,annotate_label in [(ax1,USCoreOnly_7,'a'),(ax2,IntOnly_7,'b'),(ax3,NothingAdded_7,'c')]:
			core_list,bgc_list = scenario.load_float_locations()

			data_list = dict(scenario.load_bgc_int_data())[var]
			if var == 'chl':
				data = ((data_list[19]-data_list[0])*10**6).tolist()
			if var == 'ph':
				data = (data_list[19]-data_list[0]).tolist()
			if var == 'o2':
				data = ((data_list[19]-data_list[0])).tolist()
			if var == 'no3':
				data = ((data_list[19]-data_list[0])*1000).tolist()
			data = future_H.trans_geo.transition_vector_to_plottable(data)

			plot_holder = GlobalCartopy(ax=ax,adjustable=True)
			XX,YY,ax0 = plot_holder.get_map()
			XX,YY = future_H.trans_geo.get_coords()	
			ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
			pcm = ax0.pcolormesh(XX,YY,data,vmin=0,cmap='YlOrBr',transform=ccrs.PlateCarree())
			ax0.scatter(core_list[19][1],core_list[19][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
			ax0.scatter(bgc_list[19][1],bgc_list[19][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
			ax0.annotate(annotate_label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

		fig.colorbar(pcm,ax=[ax1,ax2,ax3],label=label_uncertainty_dict[var],location='right',shrink=0.8)
		plt.savefig(plot_handler.out_file('bgc_plots_7_'+var))
		plt.close('all')

def sealevel_plots():
	plot_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/plots')
	dates = [datetime.datetime(2025,6,10)+datetime.timedelta(days = 90*(1+x)) for x in range(20)]	


	fig = plt.figure(figsize=(16,16))
	ax1 = fig.add_subplot(3,1,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(3,1,2, projection=ccrs.PlateCarree())
	ax3 = fig.add_subplot(3,1,3, projection=ccrs.PlateCarree())
	for ax,scenario,annotate_label in [(ax1,USCoreOnly_5,'a'),(ax2,IntOnly_5,'b'),(ax3,NothingAdded_5,'c')]:
		data_list = scenario.load_steric_data()
		data_list = [(np.array(dummy)-np.array(data_list[0]))*100 for dummy in data_list]
		core_list,bgc_list = scenario.load_float_locations()

		data = future_H.trans_geo.transition_vector_to_plottable(data_list[19])
		plot_holder = GlobalCartopy(ax=ax,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		XX,YY = future_H.trans_geo.get_coords()	
		ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
		pcm = ax0.pcolormesh(XX,YY,data,vmin=-1,vmax=8,cmap='YlOrBr',transform=ccrs.PlateCarree())
		ax0.scatter(core_list[19][1],core_list[19][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
		ax0.scatter(bgc_list[19][1],bgc_list[19][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
		ax0.annotate(annotate_label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	fig.colorbar(pcm,ax=[ax1,ax2,ax3],label='cm',location='right',shrink=0.8)
	plt.savefig(plot_handler.out_file('sealevel_5'))
	plt.close('all')

	fig = plt.figure(figsize=(16,16))
	ax1 = fig.add_subplot(3,1,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(3,1,2, projection=ccrs.PlateCarree())
	ax3 = fig.add_subplot(3,1,3, projection=ccrs.PlateCarree())
	for ax,scenario,annotate_label in [(ax1,USCoreOnly_7,'a'),(ax2,IntOnly_7,'b'),(ax3,NothingAdded_7,'c')]:
		data_list = scenario.load_steric_data()
		data_list = [(np.array(dummy)-np.array(data_list[0]))*100 for dummy in data_list]

		core_list,bgc_list = scenario.load_float_locations()

		data = future_H.trans_geo.transition_vector_to_plottable(data_list[19])
		plot_holder = GlobalCartopy(ax=ax,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		XX,YY = future_H.trans_geo.get_coords()	
		ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
		pcm = ax0.pcolormesh(XX,YY,data,vmin=-1,vmax=8,cmap='YlOrBr',transform=ccrs.PlateCarree())
		ax0.scatter(core_list[19][1],core_list[19][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
		ax0.scatter(bgc_list[19][1],bgc_list[19][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
		ax0.annotate(annotate_label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	fig.colorbar(pcm,ax=[ax1,ax2,ax3],label='cm',location='right',shrink=0.8)
	plt.savefig(plot_handler.out_file('sealevel_7'))
	plt.close('all')

	plt.figure(figsize=(14,10))
	for scenario in [USCoreOnly_5,IntOnly_5,NothingAdded_5,USCoreOnly_7,IntOnly_7,NothingAdded_7]:
		data_list = scenario.load_steric_data()
		data_list = [3*np.mean(np.array(dummy)-np.array(data_list[0]))/(1.3*10**-3) for dummy in data_list]
		plt.plot(date_list,data_list,scenario.linetype,lw=5,label=scenario.long_name)
		plt.ylabel('Time to Detect Anomaly (Yr)')
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=3)

	plt.savefig(plot_handler.out_file('sealevel_years'))
	plt.close('all')

def heat_plots():
	plot_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/plots')
	dates = [datetime.datetime(2025,6,10)+datetime.timedelta(days = 90*(1+x)) for x in range(20)]	

	fig = plt.figure(figsize=(16,16))
	ax1 = fig.add_subplot(3,1,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(3,1,2, projection=ccrs.PlateCarree())
	ax3 = fig.add_subplot(3,1,3, projection=ccrs.PlateCarree())
	for ax,scenario,annotate_label in [(ax1,USCoreOnly_5,'a'),(ax2,IntOnly_5,'b'),(ax3,NothingAdded_5,'c')]:
		data_list = scenario.load_heat_data()
		data_list = [(np.array(dummy)-np.array(data_list[0])) for dummy in data_list]
		core_list,bgc_list = scenario.load_float_locations()

		data = future_H.trans_geo.transition_vector_to_plottable(data_list[19])
		plot_holder = GlobalCartopy(ax=ax,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		XX,YY = future_H.trans_geo.get_coords()	
		ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
		pcm = ax0.pcolormesh(XX,YY,data,vmin=-10**19,vmax=6*10**19,cmap='YlOrBr',transform=ccrs.PlateCarree())
		ax0.scatter(core_list[19][1],core_list[19][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
		ax0.scatter(bgc_list[19][1],bgc_list[19][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
		ax0.annotate(annotate_label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	fig.colorbar(pcm,ax=[ax1,ax2,ax3],label='J',location='right',shrink=0.8)
	plt.savefig(plot_handler.out_file('heat_5'))
	plt.close('all')

	fig = plt.figure(figsize=(16,16))
	ax1 = fig.add_subplot(3,1,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(3,1,2, projection=ccrs.PlateCarree())
	ax3 = fig.add_subplot(3,1,3, projection=ccrs.PlateCarree())
	for ax,scenario,annotate_label in [(ax1,USCoreOnly_7,'a'),(ax2,IntOnly_7,'b'),(ax3,NothingAdded_7,'c')]:
		data_list = scenario.load_heat_data()
		data_list = [(np.array(dummy)-np.array(data_list[0])) for dummy in data_list]

		core_list,bgc_list = scenario.load_float_locations()

		data = future_H.trans_geo.transition_vector_to_plottable(data_list[19])
		plot_holder = GlobalCartopy(ax=ax,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		XX,YY = future_H.trans_geo.get_coords()	
		ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
		pcm = ax0.pcolormesh(XX,YY,data,vmin=-10**19,vmax=6*10**19,cmap='YlOrBr',transform=ccrs.PlateCarree())
		ax0.scatter(core_list[19][1],core_list[19][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
		ax0.scatter(bgc_list[19][1],bgc_list[19][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
		ax0.annotate(annotate_label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	fig.colorbar(pcm,ax=[ax1,ax2,ax3],label='J',location='right',shrink=0.8)
	plt.savefig(plot_handler.out_file('heat_7'))
	plt.close('all')

	heat_anomaly = 6.*10**21 # / year ... approximately, from https://www.climate.gov/news-features/understanding-climate/climate-change-ocean-heat-content
	plt.figure(figsize=(14,10))
	for scenario in [USCoreOnly_5,IntOnly_5,NothingAdded_5,USCoreOnly_7,IntOnly_7,NothingAdded_7]:
		data_list = scenario.load_heat_data()
		data_list = [np.sqrt(sum([x**2 for x in data])) for data in data_list]
		data_list = [np.array(dummy)-np.array(data_list[0]) for dummy in data_list]

		plt.plot(date_list,3*np.array(data_list)/heat_anomaly,scenario.linetype,lw=5,label=scenario.long_name)
		plt.ylabel('Time to Detect Anomaly (Yr)')
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=3)

	plt.savefig(plot_handler.out_file('heat_years'))
	plt.close('all')

def mld_plots():
	plot_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/plots')
	dates = [datetime.datetime(2025,6,10)+datetime.timedelta(days = 90*(1+x)) for x in range(20)]	

	fig = plt.figure(figsize=(16,16))
	ax1 = fig.add_subplot(3,1,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(3,1,2, projection=ccrs.PlateCarree())
	ax3 = fig.add_subplot(3,1,3, projection=ccrs.PlateCarree())
	for ax,scenario,annotate_label in [(ax1,USCoreOnly_5,'a'),(ax2,IntOnly_5,'b'),(ax3,NothingAdded_5,'c')]:
		data_list = scenario.load_ml_data()
		data_list = [(np.array(dummy)-np.array(data_list[0])) for dummy in data_list]
		core_list,bgc_list = scenario.load_float_locations()

		data = future_H.trans_geo.transition_vector_to_plottable(data_list[19])
		plot_holder = GlobalCartopy(ax=ax,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		XX,YY = future_H.trans_geo.get_coords()	
		ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
		pcm = ax0.pcolormesh(XX,YY,data,vmin=0,vmax=8,cmap='YlOrBr',transform=ccrs.PlateCarree())
		ax0.scatter(core_list[19][1],core_list[19][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
		ax0.scatter(bgc_list[19][1],bgc_list[19][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
		ax0.annotate(annotate_label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	fig.colorbar(pcm,ax=[ax1,ax2,ax3],label='Depth Uncertainty (m)',location='right',shrink=0.8)
	plt.savefig(plot_handler.out_file('ml_5'))
	plt.close('all')

	fig = plt.figure(figsize=(16,16))
	ax1 = fig.add_subplot(3,1,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(3,1,2, projection=ccrs.PlateCarree())
	ax3 = fig.add_subplot(3,1,3, projection=ccrs.PlateCarree())
	for ax,scenario,annotate_label in [(ax1,USCoreOnly_7,'a'),(ax2,IntOnly_7,'b'),(ax3,NothingAdded_7,'c')]:
		data_list = scenario.load_ml_data()
		data_list = [(np.array(dummy)-np.array(data_list[0])) for dummy in data_list]

		core_list,bgc_list = scenario.load_float_locations()

		data = future_H.trans_geo.transition_vector_to_plottable(data_list[19])
		plot_holder = GlobalCartopy(ax=ax,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		XX,YY = future_H.trans_geo.get_coords()	
		ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
		pcm = ax0.pcolormesh(XX,YY,data,vmin=0,vmax=8,cmap='YlOrBr',transform=ccrs.PlateCarree())
		ax0.scatter(core_list[19][1],core_list[19][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
		ax0.scatter(bgc_list[19][1],bgc_list[19][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
		ax0.annotate(annotate_label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	fig.colorbar(pcm,ax=[ax1,ax2,ax3],label='Depth Uncertainty (m)',location='right',shrink=0.8)
	plt.savefig(plot_handler.out_file('ml_7'))
	plt.close('all')

	plt.figure(figsize=(14,10))
	for scenario in [USCoreOnly_5,IntOnly_5,NothingAdded_5,USCoreOnly_7,IntOnly_7,NothingAdded_7]:
		data_list = scenario.load_heat_data()
		data_list = [np.mean([x**2 for x in data]) for data in data_list]
		data_list = [np.array(dummy)-np.array(data_list[0]) for dummy in data_list]

		plt.plot(date_list,np.array(data_list),scenario.linetype,lw=5,label=scenario.long_name)
		plt.ylabel('Mean Depth Uncertainty (m)')
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=3)

	plt.savefig(plot_handler.out_file('ml_years'))
	plt.close('all')

def DIC_plots():
	plot_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/plots')

	dates = [datetime.datetime(2025,6,10)+datetime.timedelta(days = 90*(1+x)) for x in range(20)]	
	label_uncertainty = '(mol $m^{-2}$)'

	cov_class=CovCM4HighCorrelation
	cov_holder = cov_class.load()

	lats = cov_holder.trans_geo.plottable_to_transition_vector(YY)
	area_in_grid = 4*np.cos(np.pi/180.*lats)*111132**2 

	fig, ax = plt.subplots(1,1,figsize=(13,10))
	for scenario in [USCoreOnly_5,IntOnly_5,NothingAdded_5]:
		data_list = scenario.load_dic_data()
		mol_per_m_squared_to_petagrams = area_in_grid*12.01/10**15
		data = [np.sum((dummy-data_list[0])*mol_per_m_squared_to_petagrams) for dummy in data_list]
		# data = [np.sqrt(np.sum(((dummy-data_list[0])*mol_per_m_squared_to_petagrams)**2)) for dummy in data_list]
		ax.plot(date_list,np.array(data),scenario.linetype,lw=5,label=scenario.long_name,alpha=0.5)
	for scenario in [USCoreOnly_7,IntOnly_7,NothingAdded_7]:
		data_list = scenario.load_dic_data()
		data = [np.sum((dummy-data_list[0])*mol_per_m_squared_to_petagrams) for dummy in data_list]
		# data = [np.sqrt(np.sum(((dummy-data_list[0])*mol_per_m_squared_to_petagrams)**2)) for dummy in data_list]
		ax.plot(date_list,np.array(data),scenario.linetype,lw=5,label=scenario.long_name,alpha=0.5)
	ax.set_ylabel('Increase in DIC Uncertainty (Pg)')
	plt.legend(ncol=3,bbox_to_anchor=(0, 1.02), loc=3)

	plt.savefig(plot_handler.out_file('dic_years'))
	plt.close('all')

	future_H = scenario.load_H('int')[0]

	fig = plt.figure(figsize=(16,16))
	ax1 = fig.add_subplot(3,1,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(3,1,2, projection=ccrs.PlateCarree())
	ax3 = fig.add_subplot(3,1,3, projection=ccrs.PlateCarree())
	for ax,scenario,annotate_label in [(ax1,USCoreOnly_7,'a'),(ax2,IntOnly_7,'b'),(ax3,NothingAdded_7,'c')]:
		core_list,bgc_list = scenario.load_float_locations()

		data_list = scenario.load_dic_data()
		data = (data_list[19]-data_list[0])*area_in_grid*12.01/10**12
		data = future_H.trans_geo.transition_vector_to_plottable(data)

		plot_holder = GlobalCartopy(ax=ax,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		XX,YY = future_H.trans_geo.get_coords()	
		ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
		pcm = ax0.pcolormesh(XX,YY,data,vmin=0,vmax=3,cmap='YlOrBr',transform=ccrs.PlateCarree())
		ax0.scatter(core_list[19][1],core_list[19][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
		ax0.scatter(bgc_list[19][1],bgc_list[19][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
		ax0.annotate(annotate_label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	fig.colorbar(pcm,ax=[ax1,ax2,ax3],label='Increase in DIC Uncertainty (Tg)',location='right',shrink=0.8)
	plt.savefig(plot_handler.out_file('dic_plots_7'))
	plt.close('all')

	fig = plt.figure(figsize=(16,16))
	ax1 = fig.add_subplot(3,1,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(3,1,2, projection=ccrs.PlateCarree())
	ax3 = fig.add_subplot(3,1,3, projection=ccrs.PlateCarree())
	for ax,scenario,annotate_label in [(ax1,USCoreOnly_5,'a'),(ax2,IntOnly_5,'b'),(ax3,NothingAdded_5,'c')]:
		core_list,bgc_list = scenario.load_float_locations()

		data_list = scenario.load_dic_data()
		data = (data_list[19]-data_list[0])*area_in_grid*12.01/10**12
		data = future_H.trans_geo.transition_vector_to_plottable(data)

		plot_holder = GlobalCartopy(ax=ax,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		XX,YY = future_H.trans_geo.get_coords()	
		ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
		pcm = ax0.pcolormesh(XX,YY,data,vmin=0,vmax=3,cmap='YlOrBr',transform=ccrs.PlateCarree())
		ax0.scatter(core_list[19][1],core_list[19][0],c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
		ax0.scatter(bgc_list[19][1],bgc_list[19][0],c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
		ax0.annotate(annotate_label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	fig.colorbar(pcm,ax=[ax1,ax2,ax3],label='Increase in DIC Uncertainty (Tg)',location='right',shrink=0.8)
	plt.savefig(plot_handler.out_file('dic_plots_5'))
	plt.close('all')

future_H_list = USCoreOnly_5.load_H('shallow')

# for scenario in [USCoreOnly_5,USCoreOnly_7,IntOnly_5,IntOnly_7,NothingAdded_5,NothingAdded_7]:
# 	BGC_uncertainty_calc_surface(scenario)
# 	BGC_uncertainty_int_calc(scenario)