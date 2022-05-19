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

# plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')
data_file_handler = FilePathHandler(ROOT_DIR,'OptimalArray')

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

def make_optimal():
	depth_idx = 2
	cov_holder = CovCM4Global.load(depth_idx = depth_idx)
	p_hat_ideal = cov_holder.cov[:]
	H_ideal = HInstance(trans_geo=cov_holder.trans_geo)
	kk = 0 
	save_array(cov_holder,H_ideal,p_hat_ideal,kk,'optimal')

	for kk in range(1,1501):
		try:
			H_ideal,p_hat_diagonal = load_array(cov_holder,kk,'optimal')
			p_hat_ideal = None
		except FileNotFoundError:
			print('could not load H instance '+str(kk)+' calculating...')
			if not isinstance(p_hat_ideal, InverseInstance):
				p_hat_ideal = make_P_hat(cov_holder.cov,H_ideal,noise_factor=4)
			idx,e_vec = get_index_of_first_eigen_vector(p_hat_ideal,cov_holder.trans_geo)
			pos = cov_holder.trans_geo.total_list[idx]
			new_float = Float(pos,cov_holder.trans_geo.variable_list)
			print(new_float.pos)
			H_ideal.add_float(new_float)
			p_hat_ideal = make_P_hat(cov_holder.cov,H_ideal,noise_factor=4)
			save_array(cov_holder,H_ideal,p_hat_ideal,kk,'optimal')

def make_random():
	for depth_idx in [2,4,6,8,10,12,14,16,18,20,22,24,26]:
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

def make_noise_matrix(cov_holder,SNR):
	l=cov_holder.trans_geo.l*2
	scale = cov_holder.calculate_scaling(l=l)
	holder = cov_holder.dist
	gaussian = np.exp(-holder**2/(2*l)) # 2l because we will square this value
	full_gaussian = cov_holder.make_scaling(gaussian)
	scaling_list = (np.sqrt(cov_holder.cov.diagonal()/SNR)).tolist()
	row_idxs,col_idxs,data = scipy.sparse.find(full_gaussian)
	for k in range(len(scaling_list)):
		data[k] = data[k]*scaling_list[col_idxs[k]]
	noise_matrix = scipy.sparse.csc_matrix((data,(row_idxs,col_idxs)),shape = cov_holder.cov.shape)
	noise_matrix = noise_matrix.dot(noise_matrix.T)
	# evals,evecs = scipy.sparse.linalg.eigs(noise_matrix, 1, sigma=-1)
	# assert evals[0]>=-10**-10
	return noise_matrix

def make_SNR():
	depth_idx = 2
	cov_holder = CovCM4Global.load(depth_idx = depth_idx)
	for SNR in [10000,5000,1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01]:
		for kk in range(10):
			print ('SNR = '+str(SNR)+', kk = '+str(kk))
			SNR_string = str(SNR).replace(".", "-")
			label = 'snr_'+SNR_string
			filepath = make_filename(label,cov_holder.trans_geo.depth_idx,kk)
			if os.path.exists(filepath):
				continue
			print('SNR is ',SNR)
			H_ideal,p_hat_diagonal = load_array(cov_holder,100,'optimal')
			H_random = HInstance.random_floats(cov_holder.trans_geo, 100, [1]*len(cov_holder.trans_geo.variable_list))
			cov_snr = make_noise_matrix(cov_holder,SNR)
			cov_holder.cov + cov_snr
			p_hat_ideal = make_P_hat(cov_snr,H_ideal,noise_factor=4)
			p_hat_random = make_P_hat(cov_snr,H_random,noise_factor=4)
			data = (H_ideal._index_of_pos,H_random._index_of_pos,p_hat_ideal.diagonal(),p_hat_random.diagonal())
			print('saved H instance '+str(kk))
			save(filepath,data)

make_random()