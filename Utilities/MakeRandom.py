from OptimalArray.Utilities.CorMat import InverseInstance
from OptimalArray.Utilities.CM4Mat import CovCM4Global
from OptimalArray.Utilities.H import HInstance,Float
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR
from OptimalArray.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from GeneralUtilities.Data.pickle_utilities import save,load
import numpy as np
import scipy
from GeneralUtilities.Compute.list import VariableList
from OptimalArray.Utilities.Utilities import make_P_hat,get_index_of_first_eigen_vector
import os 
from numpy.random import normal

# plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')
data_file_handler = FilePathHandler(ROOT_DIR,'RandomArray')

def make_filename(label,depth_idx,kk):
	return data_file_handler.tmp_file(label+'_'+str(depth_idx)+'_'+str(kk))

def load_array(cov_holder,kk,label):
	filepath = make_filename(label,cov_holder.trans_geo.depth_idx,kk)
	idx_list,p_hat_diagonal =  load(filepath)
	H_out = HInstance(trans_geo=cov_holder.trans_geo)
	for idx in idx_list:
		pos = cov_holder.trans_geo.total_list[idx]
		new_float = Float(pos,cov_holder.trans_geo.variable_list)
		H_out.add_float(new_float)
	print('loaded H instance '+str(kk))
	print('last float location was '+str(cov_holder.trans_geo.total_list[idx_list[-1]]))
	return (H_out,p_hat_diagonal)

def save_array(cov_holder,H_out,p_hat_out,kk,label):
	filepath = make_filename(label,cov_holder.trans_geo.depth_idx,kk)
	data = (H_out._index_of_pos,p_hat_out.diagonal())
	print('saved H instance '+str(kk))
	save(filepath,data)

def make_random():
	for depth_idx in [2,4,6,8,10,12,14,16,18,20,22,24]:
		cov_holder = CovCM4Global.load(depth_idx = depth_idx)
		for float_num in range(0,501,50):
			for kk in range(10):
				print ('depth = '+str(depth_idx)+', float_num = '+str(float_num)+', kk = '+str(kk))
				label = 'random_'+str(float_num)
				filepath = make_filename(label,cov_holder.trans_geo.depth_idx,kk)
				if os.path.exists(filepath):
					continue
				save(filepath,[]) # create holder so when this is run in parallel work isnt repeated
				H_random = HInstance.random_floats(cov_holder.trans_geo, float_num, [1]*len(cov_holder.trans_geo.variable_list))
				if float_num ==0: 
					save_array(cov_holder,H_random,cov_holder.cov,kk,label)					
				else:
					p_hat_random = make_P_hat(cov_holder.cov,H_random,noise_factor=4)
					save_array(cov_holder,H_random,p_hat_random,kk,label)
					
make_random()