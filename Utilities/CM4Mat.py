from OptimalArray.Utilities.CorGeo import InverseGlobal,InverseGlobalSubsample,InverseIndian,InverseSO,InverseNAtlantic,InverseTropicalAtlantic,InverseSAtlantic,InverseNPacific,InverseTropicalPacific,InverseSPacific,InverseGOM,InverseCCS
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

class CovCM4(CovArray):
	data_directory = os.path.join(get_data_folder(),'Processed/CM4/')
	chl_depth_idx = 10
	from OptimalArray.__init__ import ROOT_DIR
	label = 'cm4'
	max_depth_lev = 25  #this corresponds to 2062.5 meters 

	def __init__(self,*args,depth_idx=0,**kwargs):
		if depth_idx<self.chl_depth_idx:
			variable_list = VariableList(['po4','thetao','so','ph','chl','o2'])
		else:
			variable_list = VariableList(['po4','thetao','so','ph','o2'])
		super().__init__(*args,depth_idx=depth_idx,variable_list = variable_list,**kwargs)


	def stack_data(self):
		master_list = self.get_filenames()
		depth_mask = self.get_depth_mask()
		array_variable_list = []
		data_scale_list = []
		for variable,files in master_list:
			time_list = []
			holder_list = []
			if self.trans_geo.depth_idx>=self.chl_depth_idx:
				if variable=='chl':
					continue
			for file in files:
				print(file)
				dh = Dataset(file)
				time_list.append(dh['time'][0])
				var_temp = dh[variable][:,self.trans_geo.depth_idx,:,:]
				holder_list.append(var_temp[:,depth_mask].data)
			holder_total_list = np.vstack([x for _,x in sorted(zip(time_list,holder_list))])
			if variable=='chl':
				assert (holder_total_list>0).all()
				holder_total_list = np.log(holder_total_list)
				mean_removed,holder_total_list,data_scale = self.normalize_data(holder_total_list)
				print(holder_total_list.var().max())
			else:
				mean_removed,holder_total_list,data_scale = self.normalize_data(holder_total_list)				
				print(holder_total_list.var().max())

			array_variable_list.append((holder_total_list[:,self.trans_geo.truth_array[depth_mask]],variable))
			data_scale_list.append((data_scale[self.trans_geo.truth_array[depth_mask]],variable))
		del holder_total_list
		del holder_list
		del var_temp
		save(self.trans_geo.make_datascale_filename(),data_scale_list)
		return array_variable_list

	def make_density_plot(self):
		from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
		from matplotlib.colors import LogNorm
		from OptimalArray.Utilities.Plot.__init__ import PLOT_DIR
		file_handler = FilePathHandler(PLOT_DIR,'final_figures')

		master_list = self.get_filenames()
		filenames,variables = zip(*master_list)
		temp_filenames = filenames[list(variables).index('thetao')]
		sal_filenames = filenames[list(variables).index('so')]
		sal_holder = []
		temp_holder = []
		for sal_filename,temp_filename in zip(sal_filenames,temp_filenames):
			time_list = []
			dh_sal = Dataset(sal_filename)
			dh_temp = Dataset(temp_filename)
			time_list.append(dh_sal['time'][0])
			sal_temp = dh_sal['so'][:,self.trans_geo.depth_idx,:,:]
			sal_holder.append(sal_temp[:,self.trans_geo.truth_array].data)
			temp_temp = dh_temp['thetao'][:,self.trans_geo.depth_idx,:,:]
			temp_holder.append(temp_temp[:,self.trans_geo.truth_array].data)
		z = dh_sal["lev"][:][self.trans_geo.depth_idx]
		temp_total_list = np.vstack([x for _,x in sorted(zip(time_list,temp_holder))])
		sal_total_list = np.vstack([x for _,x in sorted(zip(time_list,sal_holder))])
		density_total_list = np.zeros(sal_total_list.shape)
		for kk in range(temp_total_list.shape[1]):
			print(kk)
			lat = self.trans_geo.total_list[kk].latitude
			lon = self.trans_geo.total_list[kk].longitude
			p = gsw.p_from_z(-z,lat)
			for ii in range(temp_total_list.shape[0]):
				SA = gsw.SA_from_SP(sal_total_list[ii,kk],p,lon,lat)
				CT = gsw.CT_from_t(SA, temp_total_list[ii,kk], p)
				rho = gsw.density.sigma0(SA,CT)
				density_total_list[ii,kk] = rho		
		plot_holder = GlobalCartopy(adjustable=True)
		XX,YY,ax = plot_holder.get_map()
		XX,YY = self.trans_geo.get_coords()
		plt.pcolormesh(XX,YY,self.trans_geo.transition_vector_to_plottable(density_total_list.var(axis=0)),norm=LogNorm())
		plt.colorbar(location='bottom',label='$(kg\ m^{-3})^2$')
		plt.savefig(file_handler.out_file('density_'+str(self.trans_geo.depth_idx)))

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
			filename = os.path.join(CovCM4.data_directory,file)
			filename = filename
			var = file.split('_')[0]
			try:
				master_dict[var].append(filename)
			except KeyError:
				master_dict[var] = [filename]
		return list(master_dict.items())

	@classmethod
	def load(cls,depth_idx):
		holder = cls(depth_idx = depth_idx)
		trans_geo = holder.trans_geo.set_l_mult(1)
		submeso_cov = InverseInstance.load(trans_geo = trans_geo)
		trans_geo = holder.trans_geo.set_l_mult(2)
		global_cov = InverseInstance.load(trans_geo = trans_geo)
		holder.cov = global_cov+submeso_cov
		return holder

class CovCM4Global(CovCM4):
	trans_geo_class = InverseGlobal
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4GlobalSubsample(CovCM4):
	trans_geo_class = InverseGlobalSubsample
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4Indian(CovCM4):
	trans_geo_class = InverseIndian
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4SO(CovCM4):
	trans_geo_class = InverseSO
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4NAtlantic(CovCM4):
	trans_geo_class = InverseNAtlantic
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4TropicalAtlantic(CovCM4):
	trans_geo_class = InverseTropicalAtlantic
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4SAtlantic(CovCM4):
	trans_geo_class = InverseSAtlantic
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4NPacific(CovCM4):
	trans_geo_class = InverseNPacific
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4TropicalPacific(CovCM4):
	trans_geo_class = InverseTropicalPacific
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4SPacific(CovCM4):
	trans_geo_class = InverseSPacific
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4GOM(CovCM4):
	trans_geo_class = InverseGOM
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4CCS(CovCM4):
	trans_geo_class = InverseCCS
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)