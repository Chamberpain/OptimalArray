from OptimalArray.Utilities.CorGeo import InverseGlobal,InverseIndian,InverseSO,InverseNAtlantic,InverseTropicalAtlantic,InverseSAtlantic,InverseNPacific,InverseTropicalPacific,InverseSPacific,InverseGOM,InverseCCS
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from OptimalArray.Utilities.CorMat import CovArray,InverseInstance
from GeneralUtilities.Compute.list import GeoList, VariableList
from netCDF4 import Dataset
import os
import numpy as np
import geopy


class CovMOM6(CovArray):
	from OptimalArray.Data.__init__ import ROOT_DIR as DATA_DIR
	data_directory = DATA_DIR + '/mom6'
	chl_depth_idx = 10
	from OptimalArray.__init__ import ROOT_DIR
	label = 'mom6'
	max_depth_lev = 25  #this corresponds to 2062.5 meters 

	def __init__(self,*args,depth_idx=0,**kwargs):
		if depth_idx<self.chl_depth_idx:
			variable_list = VariableList(['thetao','so','no3','chl','o2'])
		else:
			variable_list = VariableList(['thetao','so','no3','o2'])
		super().__init__(*args,depth_idx=depth_idx,variable_list = variable_list,**kwargs)


	def stack_data(self):
		master_list = self.get_filenames()
		dh = Dataset(master_list[0])
		array_variable_list = []
		for var in self.variable_list:
			if (self.trans_geo.depth_idx>self.chl_depth_idx)&(var=='chl'):
				continue
			var_temp = dh[var][:,self.trans_geo.depth_idx,:,:]
			holder_total_list = var_temp[:,self.trans_geo.truth_array].data
			holder_total_list = self.normalize_data(holder_total_list,self.label+'_'+var,plot=False)
			array_variable_list.append((holder_total_list,var))
		del holder_total_list
		del var_temp
		return array_variable_list

	def dimensions_and_masks(self):
		file = self.get_filenames()[0]
		dh = Dataset(file)
		temp = dh['so'][:,self.max_depth_lev,:,:]
		temp = np.ma.masked_greater(temp,40)
		depth_mask = ~temp.mask[0] # no need to grab values deepeer than 2000 meters
		X = dh['lon'][:]
		Y = dh['lat'][:]
		X_subsample_mask = (abs((X%self.trans_geo.lon_sep))<=0.0625)|(abs(X%self.trans_geo.lon_sep)>(self.trans_geo.lon_sep-0.0625))
		Y_subsample_mask = (abs((Y%self.trans_geo.lat_sep))<=0.0625)|(abs(Y%self.trans_geo.lat_sep)>(self.trans_geo.lat_sep-0.0625))
		subsample_mask = X_subsample_mask&Y_subsample_mask
		geolist = GeoList([geopy.Point(x) for x in list(zip(Y.ravel(),X.ravel()))],lat_sep=self.trans_geo.lat_sep,lon_sep=self.trans_geo.lon_sep)
		oceans_list = []
		for k,dummy in enumerate(geolist.to_shapely()):
			print(k)
			oceans_list.append(self.trans_geo.ocean_shape.contains(dummy))	# only choose coordinates within the ocean basin of interest
		total_mask = (depth_mask)&(subsample_mask)&(np.array(oceans_list).reshape(X.shape))

		lat_list = self.trans_geo.get_lat_bins()
		lon_list = self.trans_geo.get_lon_bins()
		X = [lon_list.find_nearest(x) for x in X[total_mask]]
		Y = [lat_list.find_nearest(x) for x in Y[total_mask]]
		geolist = GeoList([geopy.Point(x) for x in list(zip(Y,X))],lat_sep=self.trans_geo.lat_sep,lon_sep=self.trans_geo.lon_sep)
		print(geolist)
		return (total_mask,geolist)

	@staticmethod
	def get_depths():
		files,var = CovMOM6.get_filenames()[0]
		file = files[0]
		dh = Dataset(file)
		return dh["lev_bnds"][:]

	@staticmethod
	def get_filenames():
		return [os.path.join(CovMOM6.data_directory, "MOM6_CCS_cov_20_25N_135-123W.nc")[1:]]

	@classmethod
	def load(cls,depth_idx):
		holder = cls(depth_idx = depth_idx)
		trans_geo = holder.trans_geo.set_l_mult(1)
		submeso_cov = InverseInstance.load(trans_geo = holder.trans_geo)
		trans_geo = holder.trans_geo.set_l_mult(5)
		global_cov = InverseInstance.load(trans_geo = holder.trans_geo)
		holder.cov = global_cov+submeso_cov
		return holder

class CovMOM6Global(CovMOM6):
	trans_geo_class = InverseGlobal
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovMOM6Indian(CovMOM6):
	trans_geo_class = InverseIndian
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovMOM6SO(CovMOM6):
	trans_geo_class = InverseSO
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovMOM6NAtlantic(CovMOM6):
	trans_geo_class = InverseNAtlantic
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovMOM6TropicalAtlantic(CovMOM6):
	trans_geo_class = InverseTropicalAtlantic
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovMOM6SAtlantic(CovMOM6):
	trans_geo_class = InverseSAtlantic
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovMOM6NPacific(CovMOM6):
	trans_geo_class = InverseNPacific
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovMOM6TropicalPacific(CovMOM6):
	trans_geo_class = InverseTropicalPacific
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovMOM6SPacific(CovMOM6):
	trans_geo_class = InverseSPacific
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovMOM6GOM(CovMOM6):
	trans_geo_class = InverseGOM
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovMOM6CCS(CovMOM6):
	trans_geo_class = InverseCCS
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)