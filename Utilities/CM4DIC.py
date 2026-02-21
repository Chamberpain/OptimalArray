import os
os.environ['PROJ_LIB'] = '/home/pachamberlain/miniconda3/envs/optimal_array/share/proj'

from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from OptimalArray.Utilities.CorMat import CovArray,InverseInstance
from GeneralUtilities.Compute.list import GeoList, VariableList
from netCDF4 import Dataset
from GeneralUtilities.Data.pickle_utilities import save,load
import os
import numpy as np
import geopy
import gsw
from GeneralUtilities.Data.Filepath.instance import get_data_folder
from OptimalArray.Utilities.CorGeo import InverseGlobal,InverseGlobalSubsample,InverseIndian,InverseSO,InverseNAtlantic,InverseTropicalAtlantic,InverseSAtlantic,InverseNPacific,InverseTropicalPacific,InverseSPacific,InverseGOM,InverseCCS
from OptimalArray.Utilities.CM4Mat import CovCM4
import gc
import matplotlib.pyplot as plt
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR as PLOT_DIR
plt.rcParams['font.size'] = '16'
plot_handler = FilePathHandler(PLOT_DIR,'DIC_INT')
import cartopy.crs as ccrs

class CovCM4DIC(CovArray):
	data_directory = os.path.join(get_data_folder(),'Processed/CM4DIC/')
	trans_geo_class = InverseGlobal
	max_depth_lev = 25  #this corresponds to 2062.5 meters 

	def __init__(self,*args,**kwargs):
		variable_list = VariableList(['spco2','dissic','po4','thetao','so','ph','chl','o2'])
		super().__init__(*args,depth_idx='int',variable_list = variable_list,**kwargs)

	def stack_data(self):
		master_list = self.get_filenames()
		depth_mask = self.get_depth_mask()
		array_variable_list = []
		data_scale_list = []
		for variable,files in master_list:
			time_list = []
			holder_list = []
			for file in sorted(files):
				print(file)
				dh = Dataset(file)
				time_list.append(dh['time'][0])
				if variable=='spco2':
					var_temp = dh[variable][:,:,:]
				else:
					var_temp = dh[variable][:,:self.max_depth_lev,:,:]
					print('The data mask sum is ',var_temp.mask.sum())
					delta_z = np.diff(self.get_depths().data)
					delta_z = delta_z.flatten().tolist()[:self.max_depth_lev]
					reshape_dims = np.ones(var_temp.shape, dtype=int)
					for k,delta in enumerate(delta_z):
						reshape_dims[:,k,:,:] = reshape_dims[:,k,:,:]*delta
					if variable=='chl':
						assert (var_temp>0).all()
						var_temp = np.log(var_temp)
					var_temp = var_temp*reshape_dims
					var_temp = var_temp.sum(axis=1) #now we have computed the vertical integral
				holder_list.append(var_temp[:,depth_mask].data)
				print('holder list is ',holder_list[-1].shape)
			holder_total_list = np.vstack([x for _,x in sorted(zip(time_list,holder_list))])
			print(holder_total_list.shape)
			mean_removed,holder_total_list,data_scale = self.normalize_data(holder_total_list)				
			print(holder_total_list.var().max())

			array_variable_list.append((holder_total_list[:,self.trans_geo.truth_array[depth_mask]],variable))
			print(array_variable_list[-1][0].shape)
			data_scale_list.append((data_scale[self.trans_geo.truth_array[depth_mask]],variable))
		del holder_total_list
		del holder_list
		del var_temp
		save(self.trans_geo.make_datascale_filename(),data_scale_list)
		return array_variable_list

	def get_depth_mask(self):
		var,files = self.get_filenames()[0]
		file = files[0]
		dh = Dataset(str(file))
		temp = dh[var][:,self.max_depth_lev,:,:]
		depth_mask = ~temp.mask[0] # no need to grab values deepeer than 2000 meters
		return depth_mask

	def dimensions_and_masks(self):
		var,files = self.get_filenames()[0]
		file = files[0]
		dh = Dataset(str(file))
		temp = dh[var][:,self.max_depth_lev,:,:]
		depth_mask = self.get_depth_mask()
		X,Y = np.meshgrid(np.floor(dh['lon'][:].data),np.floor(dh['lat'][:]).data)
		X[X>=180] = X[X>=180]-360
		lat_grid = self.trans_geo.get_lat_bins()
		lon_grid = self.trans_geo.get_lon_bins()

		subsample_mask = ((np.isin(X, lon_grid))&(np.isin(Y, lat_grid)))
		geolist = GeoList([geopy.Point(x) for x in list(zip(Y.ravel(),X.ravel()))],lat_sep=self.trans_geo.lat_sep,lon_sep=self.trans_geo.lon_sep)
		oceans_list = []
		for k,dummy in enumerate(geolist.to_shapely()):
			oceans_list.append(self.trans_geo.ocean_shape.contains(dummy))	# only choose coordinates within the ocean basin of interest

		total_mask = (depth_mask)&(subsample_mask)&(np.array(oceans_list).reshape(X.shape))
		X = X[total_mask]
		Y = Y[total_mask]
		geolist = GeoList([geopy.Point(x) for x in list(zip(Y,X))],lat_sep=self.trans_geo.lat_sep,lon_sep=self.trans_geo.lon_sep)

		return (total_mask,geolist)

	@staticmethod
	def get_depths():
		var,files = CovCM4.get_filenames()[0]
		file = files[0]
		dh = Dataset(file)
		return dh["lev_bnds"][:]

	def get_units(self):
		var_list = []
		var_dict = dict(self.get_filenames())
		for var in self.trans_geo.variable_list:
			filename = var_dict[var][0]
			dh = Dataset(filename)
			var_list.append(dh[var].units)
		return var_list
		
	@staticmethod
	def get_filenames():
		master_dict = {}
		for file in os.listdir(CovCM4.data_directory):
			if '.DS_Store' in file:
				continue
			if 'mean.pkl' in file:
				continue
			if 'var.pkl' in file:
				continue
			filename = os.path.join(CovCM4.data_directory,file)
			filename = filename
			var = file.split('_')[0]
			try:
				master_dict[var].append(filename)
			except KeyError:
				master_dict[var] = [filename]
		return list(master_dict.items())

	@classmethod
	def load(cls):
		holder = cls()
		trans_geo = holder.trans_geo.set_l_mult(1)
		submeso_cov = InverseInstance.load(trans_geo = trans_geo)
		trans_geo = holder.trans_geo.set_l_mult(2)
		global_cov = InverseInstance.load(trans_geo = trans_geo)
		holder.cov = global_cov+submeso_cov
		return holder


class CovCM4LowCorrelation(CovCM4DIC):
	covariance_scale = 0.3
	label = 'cm4dic_low'

class CovCM4MediumCorrelation(CovCM4DIC):
	covariance_scale = 0.5
	label = 'cm4dic_med'

class CovCM4HighCorrelation(CovCM4DIC):
	covariance_scale = 0.7
	label = 'cm4dic_high'

class CovLowCM4Indian(CovCM4LowCorrelation):
	trans_geo_class = InverseIndian
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovLowCM4SO(CovCM4LowCorrelation):
	trans_geo_class = InverseSO
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovLowCM4NAtlantic(CovCM4LowCorrelation):
	trans_geo_class = InverseNAtlantic
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovLowCM4TropicalAtlantic(CovCM4LowCorrelation):
	trans_geo_class = InverseTropicalAtlantic
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovLowCM4SAtlantic(CovCM4LowCorrelation):
	trans_geo_class = InverseSAtlantic
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovLowCM4NPacific(CovCM4LowCorrelation):
	trans_geo_class = InverseNPacific
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovLowCM4TropicalPacific(CovCM4LowCorrelation):
	trans_geo_class = InverseTropicalPacific
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovLowCM4SPacific(CovCM4LowCorrelation):
	trans_geo_class = InverseSPacific
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovLowCM4GOM(CovCM4LowCorrelation):
	trans_geo_class = InverseGOM
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovLowCM4CCS(CovCM4LowCorrelation):
	trans_geo_class = InverseCCS
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)



def calculate_cov():
	for covclass in [CovCM4LowCorrelation,CovCM4MediumCorrelation]:

		print('covariance class is '+str(covclass))
		dummy = covclass()
		if os.path.isfile(dummy.trans_geo.make_inverse_filename()):
			continue
		try:
			# dummy.stack_data()
			dummy.calculate_cov()
			dummy.scale_cov()
		except FileNotFoundError:
			# dummy.stack_data()
			dummy.calculate_cov()
			dummy.scale_cov()
		dummy.save()
		del dummy
		gc.collect(generation=2)

def calculate_regional_cov():
	for covclass in [CovLowCM4Indian,CovLowCM4SO,CovLowCM4NAtlantic,CovLowCM4TropicalAtlantic,CovLowCM4SAtlantic,CovLowCM4NPacific,CovLowCM4TropicalPacific,CovLowCM4SPacific,CovLowCM4GOM,CovLowCM4CCS]:

		print('covariance class is '+str(covclass))
		dummy = covclass()
		if os.path.isfile(dummy.trans_geo.make_inverse_filename()):
			continue
		try:
			# dummy.stack_data()
			dummy.calculate_cov()
			dummy.scale_cov()
		except FileNotFoundError:
			# dummy.stack_data()
			dummy.calculate_cov()
			dummy.scale_cov()
		dummy.save()
		del dummy
		gc.collect(generation=2)


def plot_covariances():
	name_dict = {'spco2':'Surface PCO2','po4':'Nitrate Inventory','thetao':'Temperature Inventory'
	,'so':'Salinity Inventory','ph':'pH Inventory','chl':'Chlorophyll Inventory','o2':'Oxygen Inventory','dissic':'DIC Inventory'}
	covclass = CovCM4HighCorrelation
	print('covariance class is '+str(covclass))
	dummy = covclass.load()
	cov1 = dummy.get_cov('dissic','dissic').diagonal()

	for variable in ['spco2','po4','thetao','so','ph','chl','o2']:
		cov2 = dummy.get_cov(variable,variable).diagonal()
		try:
			cov = dummy.get_cov(variable,'dissic')
		except FileNotFoundError:
			cov = dummy.get_cov('dissic',variable)
		cor = cov/(np.sqrt(cov1)*np.sqrt(cov2))
		data_map = dummy.trans_geo.transition_vector_to_plottable(cor.diagonal())
		fig = plt.figure(figsize=(14,10))
		ax0 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
		plot_holder = dummy.trans_geo.plot_class(ax=ax0,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		pcm = ax0.pcolormesh(XX[::2,::2],YY[::2,::2],data_map,cmap='PiYG',vmin=-1,vmax=1)
		cbar = fig.colorbar(pcm, orientation='vertical', shrink=0.7)
		cbar.set_label('Correlation')

		plt.title(name_dict[variable]+' to DIC Inventory Correlation')
		plt.savefig(plot_handler.out_file(variable+'_dissic_cor'))
		plt.close()

	cov1 = dummy.get_cov('o2','o2').diagonal()

	for variable in ['spco2','po4','thetao','so','ph','chl','dissic']:
		cov2 = dummy.get_cov(variable,variable).diagonal()
		try:
			cov = dummy.get_cov(variable,'o2')
		except FileNotFoundError:
			cov = dummy.get_cov('o2',variable)
		cor = cov/(np.sqrt(cov1)*np.sqrt(cov2))
		data_map = dummy.trans_geo.transition_vector_to_plottable(cor.diagonal())
		fig = plt.figure(figsize=(14,10))
		ax0 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
		plot_holder = dummy.trans_geo.plot_class(ax=ax0,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		pcm = ax0.pcolormesh(XX[::2,::2],YY[::2,::2],data_map,cmap='PiYG',vmin=-1,vmax=1)
		cbar = fig.colorbar(pcm, orientation='vertical', shrink=0.7)
		cbar.set_label('Correlation')

		plt.title(name_dict[variable]+' to Oxygen Inventory Correlation')
		plt.savefig(plot_handler.out_file(variable+'_o2_cor'))
		plt.close()

def plot_timeseries():
	from GeneralUtilities.Data.Mapped.cm4 import CM4Sal,CM4ThetaO,CM4O2,CM4PO4,CM4PH,CM4CHL,CM4DIC,CM4PC02
	name_dict = {'spco2':'Surface PCO2','po4':'Nitrate Inventory','thetao':'Temperature Inventory'
	,'so':'Salinity Inventory','ph':'pH Inventory','chl':'Chlorophyll Inventory','o2':'Oxygen Inventory'}
	data_list = []
	for data_class in [CM4Sal,CM4ThetaO,CM4O2,CM4PO4,CM4PH,CM4CHL]:
		data = data_class().return_int()
		data_list.append((data_class.variable,data))
	data_list.append((CM4PC02.variable,CM4PC02().return_dataset()))
	lons, lats = data_class().return_dimensions()
	lon_idx = lons.find_nearest(-90,idx=True)
	lat_idx_list = []
	for lat in [-60,-50,-40,-30,-20,-10]:
		lat_idx_list.append(lats.find_nearest(lat,idx=True))

	for lat_idx in lat_idx_list:
		fig = plt.figure(figsize=(10,20))
		dic_time_series = data_list[-2][1][0][:,lat_idx,lon_idx]
		dic_time_series = dic_time_series-dic_time_series.min()
		dic_time_series = dic_time_series/dic_time_series.max()

		for k,(var,data) in enumerate(data_list[:-2]+[data_list[-1]]):
			ax = fig.add_subplot(len(data_list[:-1]),1,(k+1)) 
			time_series = data[0][:,lat_idx,lon_idx]
			time_series = time_series-time_series.min()
			time_series = time_series/time_series.max()
			ax.plot(time_series)
			ax.plot(dic_time_series)
			ax.set_title(var)
			print('Latitude = ',lats[lat_idx])
			print('Longitude = ',lons[lon_idx])
			print('Variable is ',var)
			print('Corrlation is ',np.corrcoef(dic_time_series,time_series))
		plt.tight_layout(h_pad=1.0)
		plt.suptitle('('+str(round(lats[lat_idx]))+','+str(round(lons[lon_idx]))+')')
		plt.savefig(plot_handler.tmp_file('('+str(round(lats[lat_idx]))+','+str(round(lons[lon_idx]))+')'))
		plt.close()

def plot_spatial_scales():
	covclass = CovCM4HighCorrelation
	cov_holder = covclass.load()
	unique_dist_list = np.unique(cov_holder.dist)
	unique_dist_list = unique_dist_list[unique_dist_list<20]
	cov1 = dummy.get_cov('dissic','dissic').diagonal()
	cor_dict = {'spco2':[],'po4':[],'thetao':[],'so':[],'ph':[],'chl':[],'o2':[]}
	for dist in np.sort(unique_dist_list):
		print('distance calculating is ',dist)
		mask = cov_holder.dist==dist
		for variable in ['spco2','po4','thetao','so','ph','chl','o2']:
			cov2 = dummy.get_cov(variable,variable).diagonal()
			try:
				cov = dummy.get_cov(variable,'dissic')
			except FileNotFoundError:
				cov = dummy.get_cov('dissic',variable)
			cor = cov/(np.sqrt(cov1)*np.sqrt(cov2))
			cor_dict[variable].append(abs(cor[mask]).mean().tolist())
	plt.figure(figsize=(14,10))
	for variable in ['spco2','po4','thetao','so','ph','chl','o2']:
		plt.plot(unique_dist_list,cor_dict[variable],label=variable)
		plt.legend(loc='upper right',ncol=3)
	plt.ylabel('Mean Absolute Correlation')
	plt.xlabel('Distance (degrees)')
	plt.savefig(plot_handler.out_file('bgc_to_DIC_cor_scales'))
