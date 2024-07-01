from OptimalArray.Utilities.CorMat import InverseInstance
from OptimalArray.Utilities.CM4Mat import CovCM4Global,CovCM4Indian,CovCM4SO,CovCM4NAtlantic,CovCM4TropicalAtlantic,CovCM4SAtlantic,CovCM4NPacific,CovCM4TropicalPacific,CovCM4SPacific,CovCM4GOM,CovCM4CCS
from OptimalArray.Utilities.MOM6Mat import CovMOM6CCS, CovMOM6GOM, InverseGOM
from OptimalArray.Utilities.H import HInstance,Float
from OptimalArray.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from GeneralUtilities.Data.pickle_utilities import save,load
import numpy as np
import scipy
from GeneralUtilities.Compute.list import VariableList
from OptimalArray.Utilities.Utilities import make_P_hat,get_index_of_first_eigen_vector
import os 
from numpy.random import normal
from OptimalArray.Utilities.ComputeOptimal import CovCM4Optimal
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

def make_random(cov=CovMOM6GOM):
	for depth_idx in [2,4,6,8,10,12,14,16]:
		cov_holder = cov.load(depth_idx = depth_idx)
		for float_num in range(0,70,10):
			for kk in range(10):
				print ('depth = '+str(depth_idx)+', float_num = '+str(float_num)+', kk = '+str(kk))
				label = cov.label+'_'+CovMOM6GOM.trans_geo_class.region+'_'+'random_'+str(float_num)
				filepath = make_filename(label,cov_holder.trans_geo.depth_idx,kk)
				print(filepath)
				if os.path.exists(filepath):
					continue
				save(filepath,[]) # create holder so when this is run in parallel work isnt repeated
				H_random = HInstance.random_floats(cov_holder.trans_geo, float_num, [1]*len(cov_holder.trans_geo.variable_list))
				if float_num ==0: 
					save_array(cov_holder,H_random,cov_holder.cov,kk,label)
				else:
					p_hat_random = make_P_hat(cov_holder.cov,H_random,noise_factor=4)
					save_array(cov_holder,H_random,p_hat_random,kk,label)


def make_different_variable_optimization():
	percent_list = [0,0.2,0.4,0.6,0.8,1]
	for cov in [CovCM4Indian,CovCM4SO,CovCM4NAtlantic,CovCM4TropicalAtlantic,CovCM4SAtlantic,CovCM4NPacific,CovCM4TropicalPacific,CovCM4SPacific,CovCM4GOM,CovCM4CCS]:
		for depth_idx in [2,4,6,8,10,12,14,16,18,20,22,24,26]:
			cov_holder = cov.load(depth_idx = depth_idx)
			for float_num in range(10,50,10):
				for kk in range(5):
					for ph_percent in percent_list:
						for o2_percent in percent_list:
							if depth_idx > 8:
								H_random = HInstance.random_floats(cov_holder.trans_geo, float_num, [1,1,ph_percent,o2_percent])
								label = cov.label+'_'+cov.trans_geo_class.region+'_'+'instrument_ph'+str(ph_percent)+'_o2'+str(o2_percent)+'_num'+str(float_num)
								filepath = make_filename(label,cov_holder.trans_geo.depth_idx,kk)
								print(filepath)
								if os.path.exists(filepath):
									continue
								save(filepath,[]) # create holder so when this is run in parallel work isnt repeated
								if float_num ==0: 
									save_array(cov_holder,H_random,cov_holder.cov,kk,label)
								else:
									p_hat_random = make_P_hat(cov_holder.cov,H_random,noise_factor=4)
									save_array(cov_holder,H_random,p_hat_random,kk,label)
							else:
								for chl_percent in percent_list:
									H_random = HInstance.random_floats(cov_holder.trans_geo, float_num, [1,1,ph_percent,chl_percent,o2_percent])
									label = cov.label+'_'+cov.trans_geo_class.region+'_'+'instrument_ph'+str(ph_percent)+'_o2'+str(o2_percent)+'_chl'+str(chl_percent)+'_num'+str(float_num)
									filepath = make_filename(label,cov_holder.trans_geo.depth_idx,kk)
									print(filepath)
									if os.path.exists(filepath):
										continue
									save(filepath,[]) # create holder so when this is run in parallel work isnt repeated
									if float_num ==0: 
										save_array(cov_holder,H_random,cov_holder.cov,kk,label)
									else:
										p_hat_random = make_P_hat(cov_holder.cov,H_random,noise_factor=4)
										save_array(cov_holder,H_random,p_hat_random,kk,label)

def make_random_optimal():
	depth_idx = 0
	cov_holder = CovCM4Optimal.load()
	for float_num in range(0,1301,50):
		for kk in range(50):
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
				
make_different_variable_optimization()