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

# plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')
data_file_handler = FilePathHandler(ROOT_DIR,'OptimalArray')



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





