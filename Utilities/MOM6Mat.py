from OptimalArray.Utilities.CorGeo import InverseGeo
from GeneralUtilities.Plot.Cartopy.regional_plot import CCSCartopy, GOMCartopy
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from OptimalArray.Utilities.CorMat import CovArray,InverseInstance
from GeneralUtilities.Compute.list import GeoList, VariableList
from netCDF4 import Dataset
import os
import numpy as np
import geopy
from GeneralUtilities.Data.Filepath.instance import get_data_folder
from GeneralUtilities.Compute.list import find_nearest,flat_list,LonList,LatList
import datetime
import gc

class InverseCristina(InverseGeo):
	facecolor = 'salmon'
	facename = 'California Current'
	plot_class = CCSCartopy
	region = 'ccs'
	lat_sep=.5
	lon_sep=.5
	l=1
	coord_list = [(-130.4035261964233,55),(-135.07,55),(-135.07,19.90),
	(-104.6431889409656,19.90),(-105.4266560754428,23.05901404803846),(-113.2985172073168,31.65136326179817),
	(-117.3894585799435,32.49679570904591),(-121.8138182188833,35.72586240471471),(-123.4646493631059,38.58314108287027),
	(-123.9821609070654,42.58507262179968),(-123.5031152865783,47.69525998053675),(-127.0812674058722,49.62755804643379),
	(-130.4035261964233,55)]

	def get_lat_bins(self):
		nc_fid = Dataset(self.get_dummy_file())
		return LatList(nc_fid['lat'][:][::2,0])

	def get_lon_bins(self):
		nc_fid = Dataset(self.get_dummy_file())
		return LonList(nc_fid['lon'][:][0,::2])

	def get_dummy_file(self):
		return os.path.join(get_data_folder(),'Processed/ca_mom6/MOM6_CCS_bgc_phys_2008_01.nc')

class InverseGOM(InverseGeo):
	facecolor = 'aquamarine'
	facename = 'Gulf of Mexico'
	plot_class = GOMCartopy
	region = 'gom'
	lat_sep=.5
	lon_sep=.5
	l=1
	coord_list = [(-130.4035261964233,55),(-135.07,55),(-135.07,19.90),
	(-104.6431889409656,19.90),(-105.4266560754428,23.05901404803846),(-113.2985172073168,31.65136326179817),
	(-117.3894585799435,32.49679570904591),(-121.8138182188833,35.72586240471471),(-123.4646493631059,38.58314108287027),
	(-123.9821609070654,42.58507262179968),(-123.5031152865783,47.69525998053675),(-127.0812674058722,49.62755804643379),
	(-130.4035261964233,55)]

	def get_lat_bins(self):
		nc_fid = Dataset(self.get_dummy_file())
		return LatList(nc_fid['lat'][:][::2,0])

	def get_lon_bins(self):
		nc_fid = Dataset(self.get_dummy_file())
		return LonList(nc_fid['lon'][:][0,::2])

	def get_dummy_file(self):
		return os.path.join(get_data_folder(),'Processed/gom_mom6/MOM6_CCS_bgc_phys_2008_01.nc')



class CovMOM6(CovArray):
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
		array_variable_list = []
		data_scale_list = []
		for var in self.trans_geo.variable_list:
			time_list = []
			holder_list = []
			if (self.trans_geo.depth_idx>self.chl_depth_idx)&(var=='chl'):
				continue
			for file_ in self.get_filenames():
				dh = Dataset(file_)
				time_holder = [datetime.date(int(x),int(y),int(z)) for x,y,z in zip(dh['year'][:].tolist(),dh['month'][:].tolist(),dh['day'][:].tolist())]
				time_list.append(time_holder)
				var_temp = dh[var][:,self.trans_geo.depth_idx,::2,::2]
				holder_list.append(var_temp[:,self.trans_geo.truth_array].data)
			holder_total_list = np.vstack([x for _,x in sorted(zip(time_list,holder_list))])
			if var=='chl':
				assert (holder_total_list>0).all()
				holder_total_list = np.log(holder_total_list)
				mean_removed,holder_total_list,data_scale = self.normalize_data(holder_total_list)
				print(holder_total_list.var().max())
			else:
				mean_removed,holder_total_list,data_scale = self.normalize_data(holder_total_list)				
				print(holder_total_list.var().max())
			array_variable_list.append((holder_total_list,var))
			data_scale_list.append((data_scale,var))
		del holder_total_list
		del holder_list
		del var_temp
		return array_variable_list

	def dimensions_and_masks(self):
		file = self.get_filenames()[0]
		dh = Dataset(file)
		temp = dh['so'][:,self.max_depth_lev,::2,::2]
		depth_mask = ~temp.mask[0] # no need to grab values deeper than 2000 meters
		X = dh['lon'][:][::2,::2]
		Y = dh['lat'][:][::2,::2]
		geolist = GeoList([geopy.Point(x) for x in list(zip(Y.ravel(),X.ravel()))],lat_sep=self.trans_geo.lat_sep,lon_sep=self.trans_geo.lon_sep)
		oceans_list = []
		for k,dummy in enumerate(geolist.to_shapely()):
			print(k)
			oceans_list.append(self.trans_geo.ocean_shape.contains(dummy))	# only choose coordinates within the ocean basin of interest
		total_mask = (depth_mask)&(np.array(oceans_list).reshape(X.shape))

		lat_list = self.trans_geo.get_lat_bins()
		lon_list = self.trans_geo.get_lon_bins()
		X = [lon_list.find_nearest(x) for x in X[total_mask]]
		Y = [lat_list.find_nearest(x) for x in Y[total_mask]]
		geolist = GeoList([geopy.Point(x) for x in list(zip(Y,X))],lat_sep=self.trans_geo.lat_sep,lon_sep=self.trans_geo.lon_sep)
		print(geolist)
		return (total_mask,geolist)

	@staticmethod
	def get_depths():
		file = CovMOM6.get_filenames()[0]
		dh = Dataset(file)
		return dh["depth"][:]

	@staticmethod
	def get_filenames():
		return [os.path.join(CovMOM6.data_directory,x) for x in os.listdir(CovMOM6.data_directory)]

	@classmethod
	def load(cls,depth_idx):
		holder = cls(depth_idx = depth_idx)
		trans_geo = holder.trans_geo.set_l_mult(1)
		submeso_cov = InverseInstance.load(trans_geo = trans_geo)
		trans_geo = holder.trans_geo.set_l_mult(2)
		global_cov = InverseInstance.load(trans_geo = trans_geo)
		holder.cov = global_cov+submeso_cov
		return holder

class CovMOM6CCS(CovMOM6):
	data_directory = os.path.join(get_data_folder(),'Processed/ca_mom6/')
	trans_geo_class = InverseCristina
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovMOM6GOM(CovMOM6):
	data_directory = os.path.join(get_data_folder(),'Processed/gom_mom6/')
	trans_geo_class = InverseGOM
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)


def calculate_cov():
	for covclass in [CovMOM6CCS]:
		# for depth in [8,26]:
		for depth in [4,6,10,12,14,16,18,20,22]:
			print('depth idx is '+str(depth))
			dummy = covclass(depth_idx = depth)
			if os.path.isfile(dummy.trans_geo.make_inverse_filename()):
				continue
			try:
				# dummy.stack_data()
				dummy.calculate_cov()
				dummy.scale_cov()
			except FileNotFoundError:
				# dummy.staclk_data()
				dummy.calculate_cov()
				dummy.scale_cov()
			dummy.save()
			del dummy
			gc.collect(generation=2)

