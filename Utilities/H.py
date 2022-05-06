import numpy as np
from TransitionMatrix.Utilities.ArgoData import Float,Core,BGC
from OptimalArray.Utilities.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import datetime
from GeneralUtilities.Compute.list import VariableList
import scipy.sparse
from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import full_argo_list
from TransitionMatrix.Utilities.TransGeo import GeoBase
import geopy

variable_translation_dict = {'TEMP':'thetao','PSAL':'so','PH_IN_SITU_TOTAL':'ph','CHLA':'chl','DOXY':'o2'}


class Float():
	def __init__(self,position,sensors):
		assert isinstance(position,geopy.Point)
		assert isinstance(sensors,VariableList)
		self.pos = position
		self.sensors = sensors

	def return_position_index(self,trans_geo):
		assert issubclass(trans_geo.__class__,GeoBase)
		return trans_geo.total_list.index(self.pos)

	def return_sensor_index(self,trans_geo):
		assert issubclass(trans_geo.__class__,GeoBase)
		return [trans_geo.variable_list.index(x) for x in self.sensors]


class HInstance():
	def __init__(self,trans_geo=None):
		self.trans_geo = trans_geo
		self._list_of_floats = []
		self._index_of_pos = []
		self._index_of_sensors = []

	def add_float(self,float_):
		self._list_of_floats.append(float_)
		self._index_of_pos.append(float_.return_position_index(self.trans_geo))
		self._index_of_sensors.append(float_.return_sensor_index(self.trans_geo))

	@staticmethod
	def recent_floats(GeoClass, FloatClass):
		date_list = FloatClass.get_recent_date_list()
		date_mask = [max(date_list) - datetime.timedelta(days=180) < x for x in date_list]
		bin_list = FloatClass.get_recent_bins(GeoClass.get_lat_bins(),GeoClass.get_lon_bins())
		bin_list_mask = [x in GeoClass.total_list for x in bin_list]
		sensor_list = FloatClass.get_sensors()
		sensor_idx_list = []
		sensor_mask = []
		for sensors in sensor_list:
			sensors = [x for x in sensors if x in variable_translation_dict.keys()]
			sensor_idx = [GeoClass.variable_list.index(variable_translation_dict[x]) for x in sensors if variable_translation_dict[x] in GeoClass.variable_list]
			sensor_idx_list.append(sensor_idx)
			if sensor_idx:
				sensor_mask.append(True)
			else:
				sensor_mask.append(False)

		total_mask = np.array(bin_list_mask)&np.array(date_mask)&np.array(sensor_mask)
		bin_list = np.array(bin_list)[total_mask].tolist()
		bin_index_list = [GeoClass.total_list.index(x) for x in bin_list if x in GeoClass.total_list]
		sensor_idx_list = np.array(sensor_idx_list)[total_mask].tolist()
		return HInstance.assemble_output(GeoClass,sensor_idx_list,bin_index_list)

	@staticmethod
	def random_floats(GeoClass,float_num,percent_list):
		assert len(GeoClass.variable_list)==len(percent_list)
		from random import sample
		bin_index_list = sample(range(len(GeoClass.total_list)), float_num)
		sensor_idx_list = [[] for x in range(len(bin_index_list))]
		for k,percent in enumerate(percent_list):
			num = int(np.ceil(percent*float_num))
			idxs = sample(range(len(bin_index_list)),num)
			for idx in idxs:
				sensor_idx_list[idx].append(k)
		return HInstance.assemble_output(GeoClass,sensor_idx_list,bin_index_list)

	def get_sensor_idx_list(self):
		return self._index_of_sensors

	def get_bin_idx_list(self):
		return self._index_of_pos

	def base_return(self):
		bin_idx_list = self.get_bin_idx_list()
		unique_bin_idx_list = np.unique(bin_idx_list).tolist()
		sensor_idx_list = self.get_sensor_idx_list()
		block_mat = np.zeros([len(self.trans_geo.variable_list),1]).tolist()
		for dummy in range(len(self.trans_geo.variable_list)):
			block_mat[dummy] = np.zeros([len(self.trans_geo.total_list),len(unique_bin_idx_list)])
		for variable_list,bin_idx in zip(sensor_idx_list,bin_idx_list):
			k = unique_bin_idx_list.index(bin_idx)
			for variable_idx in variable_list:
				block_mat[variable_idx][bin_idx,k]+=1
		out = np.vstack(block_mat)
		idx,dummy,data = scipy.sparse.find(scipy.sparse.csc_matrix(out))
		return (idx,data)

	def return_H(self):
		idx,data = self.base_return()
		col_idx = np.sort(np.unique(idx))
		row_idx = range(len(col_idx))
		data = [1]*len(col_idx)
		out = scipy.sparse.csc_matrix((data,(row_idx,col_idx)),shape = (max(row_idx)+1,len(self.trans_geo.variable_list)*len(self.trans_geo.total_list)))
		return out

	def return_noise_divider(self):
		idx,data = self.base_return()
		data = [x for _, x in sorted(zip(idx, data))]
		col_idx = range(len(idx))
		return data

