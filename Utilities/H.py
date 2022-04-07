import numpy as np
from TransitionMatrix.Utilities.ArgoData import Float,Core,BGC
from OptimalArray.Utilities.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import datetime
import scipy.sparse
from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import full_argo_list

variable_translation_dict = {'TEMP':'thetao','PSAL':'so','PH_IN_SITU_TOTAL':'ph','CHLA':'chl','DOXY':'o2'}


class HInstance(scipy.sparse.csc_matrix):
	def __init__(self,*args,trans_geo=None,**kwargs):
		self.trans_geo = trans_geo
		super().__init__(*args,**kwargs)

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

	@staticmethod 
	def assemble_output(GeoClass,sensor_idx_list,bin_index_list):
		block_mat = np.zeros([len(GeoClass.variable_list),1]).tolist()
		for dummy in range(len(GeoClass.variable_list)):
			block_mat[dummy] = np.zeros([len(GeoClass.total_list),len(sensor_idx_list)])
		for k,(variable_list,idx) in enumerate(zip(sensor_idx_list,bin_index_list)):
			for variable_idx in variable_list:
				block_mat[variable_idx][idx,k]+=1
		out = np.vstack(block_mat)
		return (HInstance(out,trans_geo=GeoClass).T,bin_index_list)


