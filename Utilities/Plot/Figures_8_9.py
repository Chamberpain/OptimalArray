from OptimalArray.Utilities.CM4Mat import CovCM4Global
from OptimalArray.Utilities.H import HInstance,Float
import matplotlib.pyplot as plt
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
from TransitionMatrix.Utilities.TransGeo import get_cmap
import numpy as np
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
import cartopy.crs as ccrs
import scipy
from GeneralUtilities.Compute.list import VariableList
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
from OptimalArray.Utilities.Utilities import make_P_hat,get_index_of_first_eigen_vector

plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')



def plot_e_vec(e_vec,trans_geo):
	for e_num,x in enumerate(np.split(e_vec,len(trans_geo.variable_list))):
		plot_holder = GlobalCartopy(adjustable=True)
		XX,YY,ax = plot_holder.get_map()
		XX,YY = trans_geo.get_coords()
		plt.pcolormesh(XX,YY,trans_geo.transition_vector_to_plottable(x))
		plt.colorbar()
		plt.savefig('enum_'+str(kk)+'_'+str(e_num))

def plot(p_hat_ideal,H_ideal,cov_holder,kk):
	p_hat_sum = np.split(p_hat_ideal.diagonal(), len(cov_holder.trans_geo.variable_list))
	cov_sum = np.split(cov_holder.cov.diagonal(), len(cov_holder.trans_geo.variable_list))
	for k,(p_hat,cov) in enumerate(zip(p_hat_sum,cov_sum)):
		plottable = cov_holder.trans_geo.transition_vector_to_plottable(p_hat/cov)
		plot_holder = GlobalCartopy(adjustable=True)
		XX,YY,ax = plot_holder.get_map()
		XX,YY = cov_holder.trans_geo.get_coords()
		plt.pcolormesh(XX,YY,plottable,vmin=0,vmax=1)
		plt.colorbar()
		plt.savefig(str(k)+'_'+str(kk))
		plt.close()

depth_idx = 6
cov_holder = CovCM4Global.load(depth_idx = depth_idx)
p_hat_ideal = cov_holder.cov
idx_list = []
sensor_list = []
H_ideal = HInstance(trans_geo=cov_holder.trans_geo)
for kk in range(1500):
	idx,e_vec = get_index_of_first_eigen_vector(p_hat_ideal,cov_holder.trans_geo)
	plot_e_vec(e_vec,cov_holder.trans_geo)
	pos = cov_holder.trans_geo.total_list[idx]
	new_float = Float(pos,cov_holder.trans_geo.variable_list)
	print(new_float.pos)
	H_ideal.add_float(new_float)
	p_hat_ideal = make_P_hat(cov_holder.cov,H_ideal,noise_factor=.1)
	diag = scipy.sparse.csc_matrix.diagonal(p_hat_ideal)

	if kk%10 == 0:
		plot(p_hat_ideal,H_ideal,cov_holder,kk)

SNR_list = []
for SNR in [10000,5000,1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01]:
	for _ in range(10):
		print 'SNR is ',SNR
		cov_snr = cov_holder.return_signal_to_noise(SNR)
		H_random = HInstance.random_floats
		p_hat_random,dummy = cov_holder.p_hat_calculate(H_ideal.assemble_output(),H_holder.get_bin_idx_list())
		p_hat_ideal,dummy = cov_snr.p_hat_calculate(H_ideal.assemble_output(),H_holder.get_bin_idx_list())
		SNR_list.append((SNR,calculate_p_hat_error(p_hat_random),calculate_p_hat_error(p_hat_ideal)))