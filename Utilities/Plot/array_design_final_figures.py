from TransitionMatrix.Utilities.Plot.argo_data import BGC
from TransitionMatrix.Utilities.Inversion.target_load import InverseGeo,InverseInstance,HInstance,InverseCCS,InverseGOM,InverseSOSE
from TransitionMatrix.Utilities.Compute.trans_read import TransMat
from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,full_argo_list
from GeneralUtilities.Data.lagrangian.argo.argo_read import full_argo_list
from TransitionMatrix.Utilities.Plot.argo_data import Core,BGC
import numpy as np
import scipy.sparse
import cartopy.crs as ccrs
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import matplotlib.pyplot as plt
import gc
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'array_design_final_figures')





def figure_1():

	def return_instance(array,variable):
		var_idx = trans_geo.variable_list.index(variable)
		diagonals = np.split(array.diagonal(),len(trans_geo.variable_list))
		return diagonals[var_idx]

	full_argo_list()
	for depth in [2,4,6,8,10,12,14,16,18,20]:
		noise_factor = 2
		cov = InverseInstance.load_from_type(InverseGeo,lat_sep=2,lon_sep=2,l=300,depth_idx=depth)
		trans_geo = cov.trans_geo
		h = HInstance.recent_floats(trans_geo, BGCReader)
		h = h.T
		cov = scipy.sparse.csc_matrix(cov)
		h = scipy.sparse.csc_matrix(h)
		dummy,rows = np.where((h>0).todense())
		noise = scipy.sparse.diags(cov.diagonal()[rows]*noise_factor)
		denom = h.dot(cov).dot(h.T)+noise
		del noise
		inv_denom = scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(denom.todense()))
		del denom
		cov_subtract = cov.dot(h.T).dot(inv_denom).dot(h).dot(cov)
		del h
		del inv_denom
		p_hat = cov-cov_subtract
		del cov_subtract
		gc.collect()

		for variable in trans_geo.variable_list:
			p_hat_holder = return_instance(p_hat,variable)
			cov_holder = return_instance(cov,variable)
			plottable = trans_geo.transition_vector_to_plottable(p_hat_holder/cov_holder)
			fig = plt.figure(figsize=(12,7))
			ax1 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
			XX,YY,ax1 = trans_geo.plot_setup(ax=ax1)
			plt.pcolormesh(XX,YY,plottable,vmin=0,vmax=1)
			plt.colorbar()
			plt.title('Depth Idx '+str(depth)+' '+variable)
			plt.savefig('Depth Idx '+str(depth)+' '+variable)
			plt.close()

def figure_2():
	depth_list = [2,4,6,8,10,12,14,16,18]
	class_list = [(InverseGOM,30,1,1,100),(InverseCCS,50,1,1,100),(InverseSOSE,300,1,2,300)]
	for InvClass,max_floats,lat_sep,lon_sep,l_sep in class_list:
		float_num_list = np.arange(int(max_floats/10),max_floats).tolist()[::int((max_floats/10))]
		new_file_handler = FilePathHandler(ROOT_DIR,'array_design_final_figures/'+InvClass.file_type)
		for depth in depth_list:
			for float_num in float_num_list:
				cov = InverseInstance.load_from_type(InvClass,lat_sep=lat_sep,lon_sep=lon_sep,l=l_sep,depth_idx=depth)
				trans_geo = cov.trans_geo
				for kk in range(50):
					try:
						print(kk)
						noise_factor = 4
						h = HInstance.random_floats(trans_geo,float_num)
						h = h.T
						cov = scipy.sparse.csc_matrix(cov)
						h = scipy.sparse.csc_matrix(h)
						dummy,rows = np.where((h>0).todense())
						noise = scipy.sparse.diags(cov.diagonal()[rows]*noise_factor)
						denom = h.dot(cov).dot(h.T)+noise
						del noise
						inv_denom = scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(denom.todense()))
						del denom
						cov_subtract = cov.dot(h.T).dot(inv_denom).dot(h).dot(cov)
						del h
						del inv_denom
						p_hat = cov-cov_subtract
						p_hat[p_hat<0]=0
						assert all((p_hat.diagonal()/cov.diagonal())<=1)
						assert all((p_hat.diagonal()/cov.diagonal())>=0)
						del cov_subtract
						gc.collect()
						out_array = np.split(p_hat.diagonal(),len(trans_geo.variable_list))
						out = list(zip(trans_geo.variable_list,out_array))
						filename = new_file_handler.out_file(str(depth)+'_'+str(float_num)+'_'+str(kk))
						np.save(filename,out)
					except AssertionError:
						continue


	for InvClass,max_floats,lat_sep,lon_sep,l_sep in class_list:
		float_num_list = np.arange(int(max_floats/5),max_floats).tolist()[::int((max_floats/5))]
		row_num = len(depth_list)
		col_num = len(float_num_list)
		new_file_handler = FilePathHandler(ROOT_DIR,'array_design_final_figures/'+InvClass.file_type)
		cov = InverseInstance.load_from_type(InvClass,lat_sep=lat_sep,lon_sep=lon_sep,l=l_sep,depth_idx=2)
		variance_array_dict = {}
		for var in cov.trans_geo.variable_list:
			variance_array_dict[var]=np.zeros([row_num,col_num])
		for ii,depth in enumerate(depth_list):
			cov = InverseInstance.load_from_type(InvClass,lat_sep=lat_sep,lon_sep=lon_sep,l=l_sep,depth_idx=depth)
			max_variance = dict(zip(cov.trans_geo.variable_list,np.split(cov.diagonal(),len(cov.trans_geo.variable_list))))
			for jj,float_num in enumerate(float_num_list):
				var_dict = {}
				for var in cov.trans_geo.variable_list:
					var_dict[var]=[]	
				for kk in range(50):
					print(kk)
					filename = new_file_handler.out_file(str(depth)+'_'+str(float_num)+'_'+str(kk)+'.npy')
					out = np.load(filename,allow_pickle=True)
					for var,data in out:
						print(var)
						var_dict[var].append(data)
				for var,var_list in var_dict.items():
					holder = np.vstack(var_list).mean(axis=0)/max_variance[var]
					print(holder[0])
					variance_array_dict[var][ii,jj] = holder.mean()
		cov = InverseInstance.load_from_type(InvClass,lat_sep=lat_sep,lon_sep=lon_sep,l=l_sep,depth_idx=2)
		lats,lons,depths = CovCM4.return_dimensions()
		XX,YY = np.meshgrid(float_num_list,(-1*depths[depth_list]).tolist())
		for var in cov.trans_geo.variable_list:
			plt.pcolor(XX,YY,variance_array_dict[var],vmin=0,vmax=1)
			plt.colorbar()
			if var == 'chl':
				plt.ylim([-250,-20])
			plt.title(var)
			plt.savefig(new_file_handler.out_file(var+'__'+InvClass.file_type))
			plt.close()
