from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
from OptimalArray.Utilities.CM4DIC import CovLowCM4Indian,CovLowCM4SO,CovLowCM4NAtlantic,CovLowCM4TropicalAtlantic,CovLowCM4SAtlantic,CovLowCM4NPacific,CovLowCM4TropicalPacific,CovLowCM4SPacific,CovLowCM4GOM,CovLowCM4CCS
from OptimalArray.Utilities.CorMat import CovElement
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import scipy
from OptimalArray.Utilities.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from TransitionMatrix.Utilities.TransGeo import get_cmap
plt.rcParams['font.size'] = '16'
plot_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/plots')

covclass = CovCM4HighCorrelation
dummy = covclass.load()

cov_holder_tt = CovElement.load(dummy.trans_geo, "tt", "tt")
tt_e_vals,tt_e_vecs = np.linalg.eig(cov_holder_tt)

for variable in ['spco2', 'dissic', 'po4', 'thetao', 'so', 'ph', 'chl', 'o2']:
	cov_holder_ph = CovElement.load(dummy.trans_geo, variable,variable)
	ss_e_vals,ss_e_vecs = scipy.sparse.linalg.eigs(cov_holder_ph,k=10)

	vmin = -0.025
	vmax = 0.025

	for eval_k in range(10):
		f, (ax0, ax1) = plt.subplots(2, 1,figsize=(20,12), gridspec_kw={'height_ratios': [3, 1]},subplot_kw={'projection': ccrs.PlateCarree()})
		ss_plot = dummy.trans_geo.transition_vector_to_plottable(ss_e_vecs[:, eval_k])
		plot_holder = GlobalCartopy(ax=ax0,adjustable=True)
		XX,YY,ax0 = plot_holder.get_map()
		ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
		XX,YY = cov_holder_ph.trans_geo.get_coords()
		ax0.pcolormesh(XX,YY,ss_plot,vmin=vmin,vmax=vmax,cmap='RdYlBu')

		f.colorbar(pcm,ax=[ax0],pad=.05,label='Eigenvector Weight',location='top')

		ax1 = plt.subplot(3,1,3)
		ax1.plot(tt_e_vecs[:,eval_k])
		ax1.set_ylabel("Eigen Vector Weight")
		ax1.set_xlabel("Month Index")

		plt.savefig(plot_handler.store_file(variable+'_'+str(eval_k)))
		plt.close()
	print('gravest mode explains ',tt_e_vals[1]/np.sum(tt_e_vals)*100,' % of the variance')
	print('7th mode explains ',tt_e_vals[7]/np.sum(tt_e_vals)*100,' % of the variance')
