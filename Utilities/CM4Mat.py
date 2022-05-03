from OptimalArray.Utilities.CorGeo import InverseGlobal,InverseIndian,InverseSO,InverseNAtlantic,InverseTropicalAtlantic,InverseSAtlantic,InverseNPacific,InverseTropicalPacific,InverseSPacific,InverseGOM,InverseCCS
from GeneralUtilities.Filepath.instance import FilePathHandler
from OptimalArray.Utilities.CorMat import CovArray,InverseInstance
from GeneralUtilities.Compute.list import GeoList, VariableList
from netCDF4 import Dataset
import os
import numpy as np
import geopy


class CovCM4(CovArray):
	from OptimalArray.Data.__init__ import ROOT_DIR as DATA_DIR
	data_directory = DATA_DIR + '/cm4'
	chl_depth_idx = 10
	from OptimalArray.__init__ import ROOT_DIR
	label = 'cm4'
	max_depth_lev = 25  #this corresponds to 2062.5 meters 

	def __init__(self,*args,depth_idx=0,**kwargs):
		if depth_idx<self.chl_depth_idx:
			variable_list = VariableList(['thetao','so','ph','chl','o2'])
		else:
			variable_list = VariableList(['thetao','so','ph','o2'])
		super().__init__(*args,depth_idx=depth_idx,variable_list = variable_list,**kwargs)


	def stack_data(self):
		master_list = self.get_filenames()
		array_variable_list = []
		for files,variable in master_list:
			time_list = []
			holder_list = []
			if self.trans_geo.depth_idx>self.chl_depth_idx:
				if variable=='chl':
					continue
			for file in files:
				print(file)
				dh = Dataset(file)
				time_list.append(dh['time'][0])
				var_temp = dh[variable][:,self.trans_geo.depth_idx,:,:]
				holder_list.append(var_temp[:,self.trans_geo.truth_array].data)
			holder_total_list = np.vstack([x for _,x in sorted(zip(time_list,holder_list))])
			if variable=='chl':
				holder_total_list = self.normalize_data(holder_total_list,lower_percent=0.8,upper_percent = 0.8)
				print(holder_total_list.var().max())
			else:
				holder_total_list = self.normalize_data(holder_total_list,lower_percent=0.9,upper_percent = 0.9)				
				print(holder_total_list.var().max())
			array_variable_list.append((holder_total_list,variable))
		del holder_total_list
		del holder_list
		del var_temp
		return array_variable_list

	def dimensions_and_masks(self):
		files,var = self.get_filenames()[0]
		file = files[0]
		dh = Dataset(file)
		temp = dh[var][:,self.max_depth_lev,:,:]
		depth_mask = ~temp.mask[0] # no need to grab values deepeer than 2000 meters
		X,Y = np.meshgrid(np.floor(dh['lon'][:].data),np.floor(dh['lat'][:]).data)
		X[X>180] = X[X>180]-360
		subsample_mask = ((X%self.trans_geo.lon_sep==0)&(Y%self.trans_geo.lat_sep==0))
		geolist = GeoList([geopy.Point(x) for x in list(zip(Y.ravel(),X.ravel()))],lat_sep=self.trans_geo.lat_sep,lon_sep=self.trans_geo.lon_sep)
		oceans_list = []
		for k,dummy in enumerate(geolist.to_shapely()):
			print(k)
			oceans_list.append(self.trans_geo.ocean_shape.contains(dummy))	# only choose coordinates within the ocean basin of interest

		total_mask = (depth_mask)&(subsample_mask)&(np.array(oceans_list).reshape(X.shape))
		X = X[total_mask]
		Y = Y[total_mask]
		geolist = GeoList([geopy.Point(x) for x in list(zip(Y,X))],lat_sep=self.trans_geo.lat_sep,lon_sep=self.trans_geo.lon_sep)

		return (total_mask,geolist)

	@staticmethod
	def get_depths():
		files,var = CovCM4.get_filenames()[0]
		file = files[0]
		dh = Dataset(file)
		return dh["lev_bnds"][:]

	@staticmethod
	def get_filenames():
		master_list = []
		for holder in os.walk(CovCM4.data_directory):
			folder,dummy,files = holder
			folder = folder[1:]
			variable = os.path.basename(folder)
			print(variable)
			files = [os.path.join(folder,file) for file in files if variable in file]
			if not files:
				continue
			master_list.append((files,variable))
		return master_list

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