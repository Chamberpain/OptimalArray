from OptimalArray.Utilities.CM4Mat import CovCM4Global,CovCM4GlobalSubsample,CovCM4Indian,CovCM4SO,CovCM4NAtlantic,CovCM4TropicalAtlantic,CovCM4SAtlantic,CovCM4NPacific,CovCM4TropicalPacific,CovCM4SPacific,CovCM4GOM,CovCM4CCS
from OptimalArray.Utilities.MOM6Mat import CovMOM6CCS
import scipy.sparse.linalg
import gc
import os
import shutil
from OptimalArray.Utilities.CorMat import CovElement
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from GeneralUtilities.Data.Filepath.instance import make_folder_if_does_not_exist
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
from TransitionMatrix.Utilities.TransGeo import get_cmap

def calculate_cov():
	for covclass in [CovCM4GlobalSubsample]:
		# for depth in [8,26]:
		for depth in [2,6,14,18]:
			depth_number = covclass.get_depths()[depth,0]
			print('depth idx is '+str(depth))
			dummy = covclass(depth_idx = depth)
			make_folder_if_does_not_exist(dummy.trans_geo.make_diagnostic_plot_folder())
			cov_holder_tt = CovElement.load(dummy.trans_geo, "tt", "tt")
			tt_e_vals,tt_e_vecs = np.linalg.eig(cov_holder_tt)

			mat = dummy.assemble_covariance()
			total_var = mat.diagonal().sum()
			eig_vals, eig_vecs = scipy.sparse.linalg.eigs(scipy.sparse.csc_matrix(mat),k=10)
			vmax = 0.7*abs(eig_vecs).max()
			for k,(eig_val,eig_vec) in enumerate(zip(eig_vals,eig_vecs.T)):
				for ii,trans_vec in enumerate(np.split(eig_vec,len(dummy.trans_geo.variable_list))):
					f, (ax0,ax1) = plt.subplots(2, 1,figsize=(15,15), gridspec_kw={'height_ratios': [3, 1]},subplot_kw={'projection': ccrs.PlateCarree()})
					ss_plot = dummy.trans_geo.transition_vector_to_plottable(trans_vec)
					plot_holder = GlobalCartopy(ax=ax0,adjustable=True)
					XX,YY,ax0 = plot_holder.get_map()
					ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
					XX,YY = dummy.trans_geo.get_coords()
					pcm = ax0.pcolormesh(XX,YY,ss_plot,vmin=-vmax,vmax=vmax,cmap='RdYlBu')
					ax0.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

					ax1 = plt.subplot(3,1,3)
					ax1.plot(tt_e_vecs[:72,k])
					ax1.set_ylabel("Eigen Vector Weight")
					ax1.set_xlabel("Month Index")
					ax1.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
					f.colorbar(pcm,ax=[ax0,ax1],pad=.05,label='Eigenvector Weight',location='right')
					plt.suptitle('eigenmode '+str(k)+' for '+str(dummy.trans_geo.variable_list[ii])+' variance explain = '+str(eig_val/total_var)+' depth = '+str(depth_number))
					filename = str(k)+'_'+dummy.trans_geo.variable_list[ii]
					plt.savefig(os.path.join(dummy.trans_geo.make_diagnostic_plot_folder(),filename))
					plt.close()
				f, (ax0,ax1) = plt.subplots(2, 1,figsize=(15,15), gridspec_kw={'height_ratios': [3, 1]},subplot_kw={'projection': ccrs.PlateCarree()})
				trans_vec = abs(np.stack(np.split(eig_vec, len(dummy.trans_geo.variable_list)))).sum(axis=0)	
				ss_plot = dummy.trans_geo.transition_vector_to_plottable(trans_vec)
				plot_holder = GlobalCartopy(ax=ax0,adjustable=True)
				XX,YY,ax0 = plot_holder.get_map()
				ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
				XX,YY = dummy.trans_geo.get_coords()
				pcm = ax0.pcolormesh(XX,YY,ss_plot)
				ax0.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
				ax1 = plt.subplot(3,1,3)
				ax1.plot(tt_e_vecs[:72,k])
				ax1.set_ylabel("Eigen Vector Weight")
				ax1.set_xlabel("Month Index")
				ax1.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
				f.colorbar(pcm,ax=[ax0,ax1],pad=.05,label='Eigenvector Weight',location='right')
				plt.suptitle('sum of eigenmode '+str(k)+' variance explain = '+str(eig_val/total_var)+' depth = '+str(depth_number))
				filename = str(k)+'_total'
				plt.savefig(os.path.join(dummy.trans_geo.make_diagnostic_plot_folder(),filename))
				plt.close()

			for ii,trans_vec in enumerate(np.split(mat.diagonal(), len(dummy.trans_geo.variable_list))):
				ss_plot = dummy.trans_geo.transition_vector_to_plottable(trans_vec)
				plot_holder = GlobalCartopy(adjustable=True)
				XX,YY,ax0 = plot_holder.get_map()
				pcm = ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
				XX,YY = dummy.trans_geo.get_coords()
				pcm = ax0.pcolormesh(XX,YY,ss_plot)
				plt.colorbar(pcm)
				filename = dummy.trans_geo.variable_list[ii]+'_variance'
				plt.savefig(os.path.join(dummy.trans_geo.make_diagnostic_plot_folder(),filename))
				plt.close()				

			trans_vec = abs(np.stack(np.split(mat.diagonal(), len(dummy.trans_geo.variable_list)))).sum(axis=0)	
			ss_plot = dummy.trans_geo.transition_vector_to_plottable(trans_vec)
			plot_holder = GlobalCartopy(adjustable=True)
			XX,YY,ax0 = plot_holder.get_map()
			pcm = ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
			XX,YY = dummy.trans_geo.get_coords()
			pcm = ax0.pcolormesh(XX,YY,ss_plot)
			plt.colorbar(pcm,label='scaled variance')
			plt.title('depth = '+str(depth_number))
			filename = 'total_variance'
			plt.savefig(os.path.join(dummy.trans_geo.make_diagnostic_plot_folder(),filename))
			plt.close()