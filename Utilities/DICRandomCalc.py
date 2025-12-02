from OptimalArray.Utilities.H import HInstance,Float
from OptimalArray.Utilities.CM4DIC import CovCM4HighCorrelation
from OptimalArray.Utilities.MakeRandom import make_filename
import os
from GeneralUtilities.Data.pickle_utilities import save,load
from OptimalArray.Utilities.Utilities import make_P_hat
from GeneralUtilities.Data.Mapped.cm4 import CM4DIC

def save_array(H_out,p_hat_out,kk,label):
	filepath = make_filename(CovCM4HighCorrelation.label,label,'dic',kk)
	data = (H_out._index_of_pos,p_hat_out.diagonal())
	print('saved H instance '+str(kk))
	save(filepath,data)

def load_array(kk,label,cov_holder):
	filepath = make_filename(CovCM4HighCorrelation.label,label,'dic',kk)
	try:
		idx_list,p_hat_diagonal =  load(filepath)
	except ValueError:
		p_hat_diagonal = cov_holder.cov.diagonal()
	return (p_hat_diagonal)


def calculation():
	cov_holder = CovCM4HighCorrelation.load()
	obs_list = [1]*len(cov_holder.trans_geo.variable_list)
	obs_list[cov_holder.trans_geo.variable_list.index('spco2')]=0
	obs_list[cov_holder.trans_geo.variable_list.index('dissic')]=0
	for float_num in range(0,1301,50):
		for kk in range(10):
			print ('float_num = '+str(float_num)+', kk = '+str(kk))
			label = 'random_'+str(float_num)
			filepath = make_filename(CovCM4HighCorrelation.label,label,'dic',kk)
			os.makedirs(os.path.dirname(filepath), exist_ok=True)
			if os.path.exists(filepath):
				continue
			save(filepath,[]) # create holder so when this is run in parallel work isnt repeated
			H_random = HInstance.random_floats(cov_holder.trans_geo, float_num,obs_list)
			if float_num ==0: 
				save_array(H_random,cov_holder.cov,kk,label)
			else:
				p_hat_random = make_P_hat(cov_holder.cov,H_random,noise_factor=4)
				save_array(H_random,p_hat_random,kk,label)

def plotting():
	cov_holder = CovCM4HighCorrelation.load()
	dic_idx = cov_holder.trans_geo.variable_list.index('dissic')
	cov_dic = np.split(cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list))[dic_idx]
	dic_uncert_int = np.sqrt(cov_holder.trans_geo.plottable_to_transition_vector(CM4DIC().return_int_var()[::2,::2]))
	XX,YY = cov_holder.trans_geo.get_coords()	
	lats = cov_holder.trans_geo.plottable_to_transition_vector(YY)
	area_in_grid = 4*np.cos(np.pi/180.*lats)*111132**2 
	mol_per_m_squared_to_petagrams = area_in_grid*12.01/10**15


	plot_out = []
	std = []
	floats = range(0,551,50)
	for float_num in floats:
		holder = []
		for kk in range(10):
			label = 'random_'+str(float_num)
			filepath = make_filename(CovCM4HighCorrelation.label,label,'dic',kk)
			p_hat = load_array(kk,label,cov_holder)
			dic = np.split(p_hat,len(cov_holder.trans_geo.variable_list))[dic_idx]
			mapping = dic/cov_dic
			dic_sigma = mapping*dic_uncert_int*mol_per_m_squared_to_petagrams
			holder.append(dic_sigma)
		mean_dic = np.array(holder).mean(axis=0)
		std_dic = np.array(holder).std(axis=0)
		plot_out.append(dic_sigma.sum())
		std.append(std_dic.sum())
	plot_out = np.array(plot_out)
	std = np.array(std)
	plt.plot(floats,plot_out)
	plt.fill_between(floats,plot_out-(std/np.sqrt(10)),plot_out+(std/np.sqrt(10)), alpha=0.2)
	plt.xlabel('BGC Floats')
	plt.ylabel('Unconstrained DIC (pg)')