from OptimalArray.Utilities.H import HInstance,Float
from OptimalArray.Utilities.CM4DIC import CovCM4HighCorrelation
from OptimalArray.Utilities.MakeRandom import make_filename
import os
from GeneralUtilities.Data.pickle_utilities import save,load
from OptimalArray.Utilities.Utilities import make_P_hat
from GeneralUtilities.Data.Mapped.cm4 import CM4DIC,CM4PO4,CM4ThetaO,CM4Sal,CM4PH,CM4CHL,CM4O2
import numpy as np
import matplotlib.pyplot as plt

def save_array(H_out,p_hat_out,kk,label,var):
	filepath = make_filename(CovCM4HighCorrelation.label,label,var,kk)
	data = (H_out._index_of_pos,p_hat_out.diagonal())
	print('saved H instance '+str(kk))
	save(filepath,data)

def load_array(kk,label,cov_holder,var):
	filepath = make_filename(CovCM4HighCorrelation.label,label,var,kk)
	try:
		idx_list,p_hat_diagonal =  load(filepath)
	except ValueError:
		p_hat_diagonal = cov_holder.cov.diagonal()
	return (p_hat_diagonal)


def dic_plus_core_calculation():
	cov_holder = CovCM4HighCorrelation.load()
	for float_num in range(0,1501,500):
		bgc_frac = float_num/(4000.+float_num)
		obs_list = [bgc_frac]*len(cov_holder.trans_geo.variable_list)
		obs_list[cov_holder.trans_geo.variable_list.index('spco2')]=0
		obs_list[cov_holder.trans_geo.variable_list.index('dissic')]=0
		obs_list[cov_holder.trans_geo.variable_list.index('thetao')]=1
		obs_list[cov_holder.trans_geo.variable_list.index('so')]=1
		for kk in range(10):
			print ('float_num = '+str(float_num)+', kk = '+str(kk))
			label = 'random_'+str(float_num)
			filepath = make_filename(CovCM4HighCorrelation.label,label,'dic_plus_core',kk)
			os.makedirs(os.path.dirname(filepath), exist_ok=True)
			if os.path.exists(filepath):
				continue
			save(filepath,[]) # create holder so when this is run in parallel work isnt repeated
			H_random = HInstance.random_floats(cov_holder.trans_geo, 4000+float_num,obs_list)
			if float_num ==0: 
				save_array(H_random,cov_holder.cov,kk,label,'dic_plus_core')
			else:
				p_hat_random = make_P_hat(cov_holder.cov,H_random,noise_factor=0.5)
				save_array(H_random,p_hat_random,kk,label,'dic_plus_core')

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
				save_array(H_random,cov_holder.cov,kk,label,'dic')
			else:
				p_hat_random = make_P_hat(cov_holder.cov,H_random,noise_factor=0.5)
				save_array(H_random,p_hat_random,kk,label,'dic')

def plotting():
	uncert_dict = {'dissic':CM4DIC(), 'po4':CM4PO4(), 'thetao':CM4ThetaO(), 'so':CM4Sal(), 'ph':CM4PH(), 'chl':CM4CHL(), 'o2':CM4O2()}
	label_dict = {'spco2':'Surface PCO2', 'dissic':'DIC', 'po4':'Nitrate', 'thetao':'Temperature', 'so':'Salinity', 'ph':'pH', 'chl':'Chlorophyll', 'o2':'Oxygen'}
	cov_holder = CovCM4HighCorrelation.load()
	XX,YY = cov_holder.trans_geo.get_coords()
	lats = cov_holder.trans_geo.plottable_to_transition_vector(YY)
	area_in_grid = 4*np.cos(np.pi/180.*lats)*111132**2 
	for var_idx in range(len(cov_holder.trans_geo.variable_list)):
		if var_idx==0:
			continue
		variable = cov_holder.trans_geo.variable_list[var_idx]
		variance = np.split(cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list))[var_idx]
		uncert_int = np.sqrt(cov_holder.trans_geo.plottable_to_transition_vector(uncert_dict[variable].return_int_var()[::2,::2]))
		
	
		# mol_per_m_squared_to_petagrams = area_in_grid*12.01/10**15


		plot_out = []
		std = []
		floats = range(0,1001,50)
		for float_num in floats:
			holder = []
			for kk in range(10):
				label = 'random_'+str(float_num)
				filepath = make_filename(CovCM4HighCorrelation.label,label,'dic_plus_core',kk)
				p_hat = load_array(kk,label,cov_holder,'dic_plus_core')
				p_hat_var = np.split(p_hat,len(cov_holder.trans_geo.variable_list))[var_idx]
				mapping = p_hat_var/variance
				holder.append(mapping)
				# dic_sigma = dic_uncert_int*mol_per_m_squared_to_petagrams-mapping*dic_uncert_int*mol_per_m_squared_to_petagrams
				# holder.append(dic_sigma)
			mean_dic = np.array(holder).mean(axis=0)
			std_dic = np.array(holder).std(axis=0)
			plot_out.append(mean_dic.mean())
			std.append(std_dic.mean())
		plot_out = np.array(plot_out)
		std = np.array(std)
		plt.figure()
		plt.plot(floats,plot_out*100)
		plt.fill_between(floats,(plot_out-(std/np.sqrt(10)))*100,(plot_out+(std/np.sqrt(10)))*100, alpha=0.2)
		plt.xlabel('BGC Floats')
		plt.ylabel('Mean '+ label_dict[variable]+' Mapping Error (%)')
		plt.xlim([0,1300])
	plt.show()