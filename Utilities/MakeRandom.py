import os
os.environ['PROJ_LIB'] = '/home/pachamberlain/miniconda3/envs/optimal_array/share/proj'


from OptimalArray.Utilities.CorMat import InverseInstance
# from OptimalArray.Utilities.CM4Mat import CovCM4Global,CovCM4Indian,CovCM4SO,CovCM4NAtlantic,CovCM4TropicalAtlantic,CovCM4SAtlantic,CovCM4NPacific,CovCM4TropicalPacific,CovCM4SPacific,CovCM4GOM,CovCM4CCS
# from OptimalArray.Utilities.MOM6Mat import CovMOM6CCS, CovMOM6GOM, InverseGOM
from OptimalArray.Utilities.CM4DIC import CovLowCM4Indian,CovLowCM4SO,CovLowCM4NAtlantic,CovLowCM4TropicalAtlantic,CovLowCM4SAtlantic,CovLowCM4NPacific,CovLowCM4TropicalPacific,CovLowCM4SPacific,CovLowCM4GOM,CovLowCM4CCS
from OptimalArray.Utilities.H import HInstance,Float
from OptimalArray.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from GeneralUtilities.Data.pickle_utilities import save,load
from GeneralUtilities.Compute.list import GeoList, VariableList
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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def make_filename(type,label,depth_idx,kk):
	return data_file_handler.tmp_file(type+'/'+label+'/'+str(depth_idx)+'_'+str(kk))

def load_array(type,cov_holder,kk,label):
	filepath = make_filename(type,label,cov_holder.trans_geo.depth_idx,kk)
	idx_list,p_hat_diagonal =  load(filepath)
	H_out = HInstance(trans_geo=cov_holder.trans_geo)
	for idx in idx_list:
		pos = cov_holder.trans_geo.total_list[idx]
		new_float = Float(pos,cov_holder.trans_geo.variable_list)
		H_out.add_float(new_float)
	print('loaded H instance '+str(kk))
	print('last float location was '+str(cov_holder.trans_geo.total_list[idx_list[-1]]))
	return (H_out,p_hat_diagonal)

def save_array(cov_holder,H_out,p_hat_out,kk,label,type):
	filepath = make_filename(type,label,cov_holder.trans_geo.depth_idx,kk)
	data = (H_out._index_of_pos,p_hat_out.diagonal())
	print('saved H instance '+str(kk))
	save(filepath,data)

# def make_random(cov=CovMOM6CCS, plot=False):
# 	for depth_idx in [4,16]:
# 		cov_holder = cov.load(depth_idx = depth_idx)
# 		for float_num in [5]:
# 			for kk in range(10):
# 				print ('depth = '+str(depth_idx)+', float_num = '+str(float_num)+', kk = '+str(kk))
# 				label = cov.label+'_'+cov.trans_geo_class.region+'_'+'random_'+str(float_num)
# 				filepath = make_filename(cov.trans_geo_class.region,label,cov_holder.trans_geo.depth_idx,kk)
# 				print(filepath)
# 				if os.path.exists(filepath):
# 					try:
# 						h_index,p_hat =load(filepath)
# 					except:
# 						os.remove(filepath)
# 					continue
# 				save(filepath,[]) # create holder so when this is run in parallel work isnt repeated
# 				H_random = HInstance.random_floats(cov_holder.trans_geo, float_num, [1]*len(cov_holder.trans_geo.variable_list))
# 				if float_num ==0: 
# 					save_array(cov_holder,H_random,cov_holder.cov,kk,label)
# 				else:
# 					p_hat_random = make_P_hat(cov_holder.cov,H_random,noise_factor=4)
# 					save_array(cov_holder,H_random,p_hat_random,kk,label)
# 				if plot:
# 					var_len = len(cov_holder.trans_geo.total_list)
# 					var_idx = np.linspace(0,5*var_len,6)
# 					p_hat_ox = p_hat_random[int(var_idx[4]):int(var_idx[5]),int(var_idx[4]):int(var_idx[5])]
# 					cov_ox = cov_holder.get_cov('o2','o2')

# 					cov_ox_reshaped = cov_holder.trans_geo.transition_vector_to_plottable(cov_ox.diagonal())
# 					p_hat_ox_reshaped = cov_holder.trans_geo.transition_vector_to_plottable(p_hat_ox.diagonal())

# 					lats = cov_holder.trans_geo.get_lat_bins()
# 					lons = cov_holder.trans_geo.get_lon_bins()

# 					XX,YY = np.meshgrid(lons,lats)

# 					plt.pcolor(XX,YY,p_hat_ox_reshaped/cov_ox_reshaped)

# 					x = [x.longitude for x in H_random.return_pos_of_bgc()]
# 					y = [x.latitude for x in H_random.return_pos_of_bgc()]

# 					plt.scatter(x,y)
# 					plt.colorbar()


def make_different_variable_optimization(plot=False):
	plot=False
	percent_list = [0,0.33,0.66,1]
	for cov in [CovLowCM4Indian,CovLowCM4SO,CovLowCM4NAtlantic,CovLowCM4TropicalAtlantic,CovLowCM4SAtlantic,CovLowCM4NPacific,CovLowCM4TropicalPacific,CovLowCM4SPacific,CovLowCM4GOM,CovLowCM4CCS]:
		cov_holder = cov.load()
		for x in [0,float(1/7**2),float(1/6**2),float(1/5**2),float(1/4.5**2),float(1/4**2),float(1/3.5**2),float(1/3**2),float(1/2.5**2)]:
			float_num = round(x*len(cov_holder.trans_geo.total_list))
			print(float_num)
			for kk in range(5):
				for ph_percent in percent_list:
					for o2_percent in percent_list:
						for po4_percent in percent_list:
							for chl_percent in percent_list:
								H_random = HInstance.random_floats(cov_holder.trans_geo, float_num, [0,0,po4_percent,1,1,ph_percent,chl_percent,o2_percent])
								label = cov.label+'/'+cov.trans_geo_class.region+'/'+'ph'+str(ph_percent)+'_o2'+str(o2_percent)+'_po4'+str(po4_percent)+'_chl'+str(chl_percent)+'_num'+str(float_num)
								filepath = make_filename('instrument',label,cov_holder.trans_geo.depth_idx,kk)
								print(filepath)
								if os.path.exists(filepath):
									continue
								try:
									save(filepath,[]) # create holder so when this is run in parallel work isnt repeated
								except FileNotFoundError:
									dir_path = os.path.dirname(filepath)
									try: 
										os.makedirs(dir_path)
										save(filepath,[]) # create holder so when this is run in parallel work isnt repeated
									except FileExistsError:
										continue
								except FileExistsError:
									continue
								if float_num ==0: 
									save_array(cov_holder,H_random,cov_holder.cov,kk,label,'instrument')
								else:
									p_hat_random = make_P_hat(cov_holder.cov,H_random,noise_factor=3)
									save_array(cov_holder,H_random,p_hat_random,kk,label,'instrument')



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
				
def make_different_variable_optimization_plot():
	depth = 2
	ph_percent = 0.4
	chl_percent = 0.8
	o2_percent = 1
	cov = CovCM4NAtlantic
	cov_holder = cov.load(depth_idx = depth)
	float_num = 40
	label = cov.label+'/'+cov.trans_geo_class.region+'/'+'ph'+str(ph_percent)+'_o2'+str(o2_percent)+'_chl'+str(chl_percent)+'_num'+str(float_num)
	H_out,p_hat_diagonal = load_array('instrument',cov_holder,kk,label)
	total_p_hat = np.sum(np.split(p_hat_diagonal,len(cov_holder.trans_geo.variable_list)),axis=0)
	total_p = np.sum(np.split(cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list)),axis=0)
	mapping_error = total_p_hat/total_p
	mapping_error = cov_holder.trans_geo.transition_vector_to_plottable(mapping_error)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
	XX,YY,ax = cov_holder.trans_geo.plot_class(ax=ax).get_map()
	cax = ax.pcolormesh(XX,YY,mapping_error*100)
	lons, lats = zip(*[(x.longitude, x.latitude) for x in H_out.return_pos_of_bgc()])
	ax.scatter(lons, lats,marker='*',c='r')
	fig.colorbar(cax, label = 'Mapping Error (%)')
	plt.show()
# make_different_variable_optimization()