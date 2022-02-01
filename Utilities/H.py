import numpy as np
import scipy 
from OptimalArray.Utilities.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler

class HInstance(scipy.sparse.csc_matrix):
	def __init__(self,*args,trans_geo=None,**kwargs):
		self.trans_geo = trans_geo
		super().__init__(*args,**kwargs)

	@classmethod
	def recent_floats(cls,GeoClass, FloatClass):
		block_mat = np.zeros([len(GeoClass.variable_list),len(GeoClass.variable_list)]).tolist()
		for k,variable in enumerate(GeoClass.variable_list):
			float_var = GeoClass.variable_translation_dict[variable]
			var_grid = FloatClass.recent_bins_by_sensor(float_var,GeoClass.get_lat_bins(),GeoClass.get_lon_bins())
			idx_list = [GeoClass.total_list.index(x) for x in var_grid if x in GeoClass.total_list]
			holder_array = np.zeros([len(GeoClass.total_list),len(idx_list)])
			for ii,idx in enumerate(idx_list):
				holder_array[idx,ii]+=1
			block_mat[k][k] = holder_array

		for k in range(len(block_mat)):
			holder = np.zeros(block_mat[k][k].shape)
			for j in range(len(block_mat)):
				if k==j:
					continue
				else:
					block_mat[j][k]=holder
		out = np.block(block_mat)
		return cls(out,trans_geo=GeoClass)

	@classmethod
	def random_floats(cls,GeoClass,float_num):
		holder_array = np.zeros([len(GeoClass.total_list),float_num])
		idxs = list(range(len(GeoClass.total_list)))
		subsampled_idx = random.sample(idxs,float_num)
		for col,row in zip(range(float_num),subsampled_idx):
			holder_array[row,col]=1
		block_mat = np.zeros([len(GeoClass.variable_list),len(GeoClass.variable_list)]).tolist()
		for k,variable in enumerate(GeoClass.variable_list):
			block_mat[k][k] = holder_array
		for k in range(len(block_mat)):
			holder = np.zeros(block_mat[k][k].shape)
			for j in range(len(block_mat)):
				if k==j:
					continue
				else:
					block_mat[j][k]=holder
		out = np.block(block_mat)
		return cls(out,trans_geo=GeoClass)
