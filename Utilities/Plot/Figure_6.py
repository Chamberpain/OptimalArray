from OptimalArray.Utilities.CM4Mat import CovCM4Global
from OptimalArray.Utilities.CorMat import CovElement
import matplotlib.pyplot as plt
import numpy as np 
import scipy
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
import cartopy.crs as ccrs
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from TransitionMatrix.Utilities.TransGeo import get_cmap
plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')

depths = CovCM4Global.get_depths()

covclass = CovCM4Global
shallow_dummy = covclass(depth_idx = 6)
deep_dummy = covclass(depth_idx = 18)


shallow_cov_holder_ph = CovElement.load(shallow_dummy.trans_geo, "ph", "ph")
ss_e_vals,shallow_ss_e_vecs = scipy.sparse.linalg.eigs(shallow_cov_holder_ph,k=2)
cov_holder_tt = CovElement.load(shallow_dummy.trans_geo, "tt", "tt")
shallow_tt_e_vals,shallow_tt_e_vecs = np.linalg.eig(cov_holder_tt)


deep_cov_holder_ph = CovElement.load(deep_dummy.trans_geo, "ph", "ph")
ss_e_vals,deep_ss_e_vecs = scipy.sparse.linalg.eigs(deep_cov_holder_ph,k=2)
cov_holder_tt = CovElement.load(deep_dummy.trans_geo, "tt", "tt")
deep_tt_e_vals,deep_tt_e_vecs = np.linalg.eig(cov_holder_tt)




vmin = -0.025
vmax = 0.025

f, ((ax0, ax1),(ax2, ax3)) = plt.subplots(2, 2,figsize=(20,12), gridspec_kw={'height_ratios': [3, 1]},subplot_kw={'projection': ccrs.PlateCarree()})
ss_plot = shallow_dummy.trans_geo.transition_vector_to_plottable(shallow_ss_e_vecs[:, 1])
plot_holder = GlobalCartopy(ax=ax0,adjustable=True)
XX,YY,ax0 = plot_holder.get_map()
ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
XX,YY = shallow_cov_holder_ph.trans_geo.get_coords()
ax0.pcolormesh(XX,YY,ss_plot,vmin=vmin,vmax=vmax,cmap='RdYlBu')
ax0.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)


ss_plot = deep_dummy.trans_geo.transition_vector_to_plottable(deep_ss_e_vecs[:, 1])
plot_holder = GlobalCartopy(ax=ax1,adjustable=True)
XX,YY,ax1 = plot_holder.get_map()
ax1.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
XX,YY = deep_cov_holder_ph.trans_geo.get_coords()
pcm = ax1.pcolormesh(XX,YY,ss_plot,vmin=vmin,vmax=vmax,cmap='RdYlBu')
ax1.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

f.colorbar(pcm,ax=[ax0,ax1],pad=.05,label='Eigenvector Weight',location='top')

ax2 = plt.subplot(3,2,5)
ax2.plot(shallow_tt_e_vecs[:72,0])
ax2.set_ylabel("Eigen Vector Weight")
ax2.set_xlabel("Month Index")
ax2.annotate('c', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

ax3 = plt.subplot(3,2,6)
ax3.plot(deep_tt_e_vecs[:72,9])
ax3.set_xlabel("Month Index")
ax3.get_yaxis().set_visible(False)
ax2.get_shared_y_axes().join(ax2, ax3)
ax3.annotate('d', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
plt.savefig(file_handler.out_file('Figure_6'),bbox_inches='tight')
plt.close()

print('shallow mode explains ',shallow_tt_e_vals[1]/np.sum(shallow_tt_e_vals)*100,' % of the variance')
print('deep mode explains ',deep_tt_e_vals[1]/np.sum(deep_tt_e_vals)*100,' % of the variance')
