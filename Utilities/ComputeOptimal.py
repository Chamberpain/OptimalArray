from OptimalArray.Utilities.CorMat import InverseInstance
from OptimalArray.Utilities.CM4Mat import CovCM4GlobalSubsample, CovCM4
from OptimalArray.Utilities.CorMat import CovArray 
from OptimalArray.Utilities.H import HInstance,Float
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR
from OptimalArray.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from GeneralUtilities.Data.pickle_utilities import save,load
import numpy as np
import scipy
from GeneralUtilities.Compute.list import VariableList
from OptimalArray.Utilities.Utilities import make_P_hat,get_index_of_first_eigen_vector
from OptimalArray.Utilities.CorGeo import InverseGeo
import os 
from numpy.random import normal
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
import copy


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

class OptimalGlobal(InverseGeo):
	plot_class = GlobalCartopy
	region = 'optimal_global'
	coord_list = [[-180, 90], [180, 90], [180, -90], [-180, -90], [-180, 90]]
	lat_sep=4
	lon_sep=4
	l=2

class CovCM4Optimal(CovArray):
	trans_geo_class = OptimalGlobal
	data_directory = CovCM4.data_directory
	from OptimalArray.__init__ import ROOT_DIR
	label = 'cm4'
	max_depth_lev = 25  #this corresponds to 2062.5 meters 

	def __init__(self,*args,**kwargs):
		variable_list = VariableList(['thetao_2','so_2','ph_2','chl_2','o2_2','thetao_6','so_6','ph_6','chl_6','o2_6','thetao_14','so_14','ph_14','o2_14','thetao_18','so_18','ph_18','o2_18'])
		super().__init__(*args,depth_idx=0,variable_list = variable_list,**kwargs)

	@classmethod
	def load(cls):
		cov_2 = CovCM4GlobalSubsample.load(2)
		cov_6 = CovCM4GlobalSubsample.load(6)
		cov_14 = CovCM4GlobalSubsample.load(14)
		cov_18 = CovCM4GlobalSubsample.load(18)
		holder = cls()
		holder.cov = InverseInstance(scipy.sparse.block_diag((cov_2.cov,cov_6.cov,cov_14.cov,cov_18.cov)))
		return holder

	@staticmethod
	def get_filenames():
		master_dict = {}
		for file in os.listdir(CovCM4.data_directory):
			if file == '.DS_Store':
				continue
			filename = os.path.join(CovCM4.data_directory,file)
			filename = filename
			var = file.split('_')[0]
			try:
				master_dict[var].append(filename)
			except KeyError:
				master_dict[var] = [filename]
		return list(master_dict.items())

CovCM4Optimal.dimensions_and_masks = CovCM4.dimensions_and_masks

def make_optimal():
	cov_holder = CovCM4Optimal.load()
	p_hat_ideal = cov_holder.cov[:]
	H_ideal = HInstance(trans_geo=cov_holder.trans_geo)
	kk = 0 
	save_array(cov_holder,H_ideal,p_hat_ideal,kk,'optimal')

	for kk in range(1,1101):
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



def make_total_noise_matrix(SNR):


	def make_noise_matrix(cov_holder,SNR):
		print('makeing the SNR matrix ...')
		print('SNR = ',SNR)
		l=cov_holder.trans_geo.l*2
		gaussian = np.exp(-cov_holder.dist**2/(4*l)) # 4l because we will square this value and it will become 2l
		full_gaussian = cov_holder.make_scaling(gaussian)
		data_list = full_gaussian[cov_holder.cov!=0].tolist()[0]
		row_idx,column_idx,dummy = scipy.sparse.find(cov_holder.cov)
		scaling_list = (np.sqrt(cov_holder.cov.diagonal()/SNR)).tolist()

		data = full_gaussian[cov_holder.cov!=0].tolist()[0]
		row_idx,col_idx,dummy = scipy.sparse.find(cov_holder.cov)
		scale_list = []
		for k,col in enumerate(col_idx):
			data[k] = normal(scale=abs(data[k])*scaling_list[col])
		noise_matrix = scipy.sparse.csc_matrix((data,(row_idx,col_idx)),shape = cov_holder.cov.shape)
		noise_matrix = noise_matrix*(noise_matrix.T)
		# evals,evecs = scipy.sparse.linalg.eigs(noise_matrix, 1, sigma=-1)
		# assert evals[0]>=-10**-10
		return noise_matrix

	cov_2 = CovCM4GlobalSubsample.load(2)
	cov_2 = make_noise_matrix(cov_2,SNR)
	cov_6 = CovCM4GlobalSubsample.load(6)
	cov_6 = make_noise_matrix(cov_6,SNR)
	cov_14 = CovCM4GlobalSubsample.load(14)
	cov_14 = make_noise_matrix(cov_14,SNR)
	cov_18 = CovCM4GlobalSubsample.load(18)
	cov_18 = make_noise_matrix(cov_18,SNR)

	return InverseInstance(scipy.sparse.block_diag((cov_2,cov_6,cov_14,cov_18)))



def make_SNR():
	cov_holder = CovCM4Optimal.load()
	for SNR in [10000,5000,1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01]:
		for kk in range(10):
			print ('SNR = '+str(SNR)+', kk = '+str(kk))
			SNR_string = str(SNR).replace(".", "-")
			label = 'snr_'+SNR_string
			filepath = make_filename(label,cov_holder.trans_geo.depth_idx,kk)
			if os.path.exists(filepath):
				continue
			print('SNR is ',SNR)
			H_ideal,p_hat_diagonal = load_array(cov_holder,1000,'optimal')
			H_random = HInstance.random_floats(cov_holder.trans_geo, 1000, [1]*len(cov_holder.trans_geo.variable_list))
			cov_snr = make_total_noise_matrix(SNR)
			cov_out = cov_holder.cov + cov_snr
			p_hat_ideal = make_P_hat(cov_out,H_ideal,noise_factor=4)
			p_hat_random = make_P_hat(cov_out,H_random,noise_factor=4)
			data = (H_ideal._index_of_pos,H_random._index_of_pos,p_hat_ideal.diagonal(),p_hat_random.diagonal())
			print('saved H instance '+str(kk))
			save(filepath,data)

