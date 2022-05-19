import numpy as np
from TransitionMatrix.Utilities.ArgoData import Float,Core,BGC
from OptimalArray.Utilities.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
import datetime
from GeneralUtilities.Compute.list import VariableList,GeoList
import scipy.sparse
from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader, full_argo_list
from TransitionMatrix.Utilities.TransGeo import GeoBase
import geopy
from random import sample

variable_translation_dict = {'TEMP':'thetao','PSAL':'so','PH_IN_SITU_TOTAL':'ph','CHLA':'chl','DOXY':'o2'}


class Float():
	def __init__(self,position,sensors,date_deployed=None,ID=None):
		assert isinstance(position,geopy.Point)
		assert isinstance(sensors,VariableList)
		self.pos = position
		self.sensors = sensors
		self.ID = ID
		self.date_deployed = date_deployed
		self.date_death = self.date_deployed+datetime.timedelta(days=(365*5))

	def return_position_index(self,trans_geo):
		assert issubclass(trans_geo.__class__,GeoBase)
		return trans_geo.total_list.index(self.pos)

	def return_sensor_index(self,trans_geo):
		assert issubclass(trans_geo.__class__,GeoBase)
		return [trans_geo.variable_list.index(x) for x in self.sensors if x in trans_geo.variable_list]


class HInstance():
	def __init__(self,trans_geo=None):
		assert issubclass(trans_geo.__class__,GeoBase)
		self.trans_geo = trans_geo
		self._list_of_floats = []
		self._index_of_pos = []
		self._index_of_sensors = []

	def add_float(self,float_):
		self._list_of_floats.append(float_)
		self._index_of_pos.append(float_.return_position_index(self.trans_geo))
		self._index_of_sensors.append(float_.return_sensor_index(self.trans_geo))

	def return_pos_of_variable(self,var):
		var_index = self.trans_geo.variable_list.index(var)
		mask = [var_index in x for x in self._index_of_sensors]
		pos_list = [x.pos for x,y in zip(self._list_of_floats,mask) if y]
		return GeoList(pos_list)

	def return_pos_of_bgc(self):
		mask = [x for x in self._index_of_sensors if len(x)>=3]
		pos_list = [x.pos for x,y in zip(self._list_of_floats,mask) if y]
		return GeoList(pos_list)

	def return_pos_of_core(self):
		temp_index = self.trans_geo.variable_list.index('thetao')
		sal_index = self.trans_geo.variable_list.index('so')
		core_mask = [(x==[temp_index,sal_index])|(x==[sal_index,temp_index]) for x in self._index_of_sensors]
		pos_list = [x.pos for x,y in zip(self._list_of_floats,core_mask) if y]
		return GeoList(pos_list)

	@staticmethod
	def random_floats(GeoClass,float_num,percent_list):
		assert len(GeoClass.variable_list)==len(percent_list)
		H_holder = HInstance(trans_geo = GeoClass)
		bin_index_list = sample(range(len(GeoClass.total_list)), float_num)
		sensor_idx_list = [[] for x in range(len(bin_index_list))]
		for k,percent in enumerate(percent_list):
			num = int(np.ceil(percent*float_num))
			idxs = sample(range(len(bin_index_list)),num)
			for idx in idxs:
				sensor_idx_list[idx].append(GeoClass.variable_list[k])
		sensor_idx_list = [VariableList(x) for x in sensor_idx_list]
		pos_list = [GeoClass.total_list[x] for x in bin_index_list]
		for pos,sensor in zip(pos_list,sensor_idx_list):
			dummy_float = Float(pos,sensor)
			H_holder.add_float(dummy_float)
		return H_holder

	def remove_by_index(self,idx):
		del self._list_of_floats[idx]
		del self._index_of_pos[idx]
		del self._index_of_sensors[idx]

	def get_sensor_idx_list(self):
		return self._index_of_sensors

	def get_bin_idx_list(self):
		return self._index_of_pos

	def get_id_list(self):
		return [x.ID for x in self._list_of_floats]

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

