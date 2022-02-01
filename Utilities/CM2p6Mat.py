from OptimalArray.Utilities.CorMat import CovArray
import numpy as np
from GeneralUtilities.Compute.list import VariableList
from GeneralUtilities.Filepath.search import find_files

class CovCM2p6(CovArray):
	from OptimalArray.__init__ import ROOT_DIR as DATA_DIR
	data_directory = DATA_DIR + '/data/cm2p6'
	variable_list = VariableList['o2','dic','temp','salt']
	label = 'cm2p6'
	from OptimalArray.Data.__init__ import ROOT_DIR
	file_handler = FilePathHandler(ROOT_DIR,'CorCalc/'+label)

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

		def combine_data(file_format,data_directory):
			matches = find_files(data_directory)
			array_list = [np.load(match)[:,self.truth_array] for match in sorted(matches)]
			return np.vstack(array_list)

		salt = combine_data('*100m_subsampled_salt.npy',self.data_directory)
		dic = combine_data('*100m_subsampled_dic.npy',self.data_directory)
		o2 = combine_data('*100m_subsampled_o2.npy',self.data_directory)
		temp = combine_data('*100m_subsampled_temp.npy',self.data_directory)

		file_format = '*_time.npy'
		matches = self.get_filenames()
		
		self.time = flat_list([np.load(match).tolist() for match in sorted(matches)])

		salt = normalize_data(salt,'salt')
		dic = normalize_data(dic,'dic')
		o2 = normalize_data(o2,'o2')
		temp = normalize_data(temp,'temp')
		self.data = np.hstack([o2,dic,temp,salt])

		del salt
		del dic
		del o2
		del temp

	@staticmethod
	def dimensions_and_masks(lat_sep=None,lon_sep=None):
		lat_grid = np.arange(-90,91,lat_sep)
		lats = np.load(cov_array.data_directory+'lat_list.npy')
		lat_translate = [find_nearest(lats,x) for x in lat_grid] 
		lat_truth_list = np.array([x in lat_translate for x in lats])

		lon_grid = np.arange(-180,180,lon_sep)
		lons = np.load(cov_array.data_directory+'lon_list.npy')
		lons[lons<-180]=lons[lons<-180]+360
		lon_translate = [find_nearest(lons,x) for x in lon_grid] 
		lon_truth_list = np.array([x in lon_translate for x in lons])

		truth_list = lat_truth_list&lon_truth_list
		lats = lats[truth_list]
		lons = lons[truth_list]
		idx_lons = [find_nearest(lon_grid,x) for x in lons] 
		idx_lats = [find_nearest(lat_grid,x) for x in lats]
		index_list = [list(x) for x in list(zip(idx_lats,idx_lons))]		
		return (lats,lons,truth_list,lat_grid,lon_grid,index_list)

	def get_filenames(self):
		return find_files(self.data_directory)
