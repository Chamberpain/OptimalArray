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

plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')



def plot_e_vec(e_vec,trans_geo):
	plot_holder = GlobalCartopy(adjustable=True)
	XX,YY,ax = plot_holder.get_map()
	XX,YY = trans_geo.get_coords()
	plt.pcolormesh(XX,YY,trans_geo.transition_vector_to_plottable(e_vec))
	plt.colorbar()
	plt.show()

def get_index_of_first_eigen_vector(p_hat):
	p_hat[p_hat<0]=0
	eigs = scipy.sparse.linalg.eigs(p_hat,k=1)
	e_vec = eigs[1][:,-1]
	e_vec = sum(abs(x) for x in np.split(e_vec,len(p_hat_ideal.trans_geo.variable_list)))
	idx = e_vec.tolist().index(e_vec.max())
	return idx,e_vec


def make_noise(H_holder,cov,noise_factor):
	idx,dummy = H_holder.base_return()
	noise_divider = H_holder.return_noise_divider()
	idx = np.sort(idx)
	out = cov.cov[idx,idx]*noise_factor/noise_divider
	return scipy.sparse.diags(np.ravel(out))


def p_hat_calculate(H_holder,cov_holder,noise_factor=2):
	noise = make_noise(H_holder,cov_holder,noise_factor)
	H = H_holder.return_H()
	denom = H.dot(cov_holder.cov).dot(H.T)+noise
	inv_denom = scipy.sparse.linalg.inv(denom)
	if not type(inv_denom)==scipy.sparse.csc.csc_matrix:
		inv_denom = scipy.sparse.csc.csc_matrix(inv_denom)  # this is for the case of 1x1 which returns as array for some reason
	cov_subtract = cov_holder.cov.dot(H.T.dot(inv_denom).dot(H).dot(cov_holder.cov))

	p_hat = cov_holder.cov-cov_subtract
	return p_hat,cov_subtract


def plot(p_hat_ideal,H_ideal,cov_holder,kk):
	p_hat_sum = np.split(p_hat_ideal.diagonal(), len(cov_holder.trans_geo.variable_list))
	cov_sum = np.split(cov_holder.cov.diagonal(), len(cov_holder.trans_geo.variable_list))
	for k,(p_hat,cov) in enumerate(zip(p_hat_sum,cov_sum)):
		plottable = cov_holder.trans_geo.transition_vector_to_plottable(p_hat/cov)
		plot_holder = GlobalCartopy(adjustable=True)
		XX,YY,ax = plot_holder.get_map()
		XX,YY = cov_holder.trans_geo.get_coords()
		plt.pcolormesh(XX,YY,plottable)
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
	pos = cov_holder.trans_geo.total_list[get_index_of_first_eigen_vector(p_hat_ideal)[0]]
	new_float = Float(pos,cov_holder.trans_geo.variable_list)
	print(new_float.pos)
	H_ideal.add_float(new_float)
	p_hat_ideal,dummy = p_hat_calculate(H_ideal,cov_holder,noise_factor=1.4)
	diag = scipy.sparse.csc_matrix.diagonal(p_hat_ideal)
	for idx in np.where(diag<0)[0]:
		print(idx)
		p_hat_ideal[idx,idx]=0
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

