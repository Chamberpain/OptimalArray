from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
from OptimalArray.Utilities.CM4Mat import CovCM4Global
from OptimalArray.Utilities.CorMat import CovElement
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')


covclass = CovCM4Global
depth = 2
dummy = covclass(depth_idx = depth)


cov_holder_tt = CovElement.load(dummy.trans_geo, "tt", "tt")
tt_e_vals,tt_e_vecs = np.linalg.eig(cov_holder_tt)
cov_holder_ph = CovElement.load(dummy.trans_geo, "ph", "ph")
ss_e_vals,ss_e_vecs = np.linalg.eig(cov_holder_ph)

vmin = -0.025
vmax = 0.025

f, ((ax0, ax1),(ax2, ax3)) = plt.subplots(2, 2,figsize=(20,12), gridspec_kw={'height_ratios': [3, 1]},subplot_kw={'projection': ccrs.PlateCarree()})
ss_plot = dummy.trans_geo.transition_vector_to_plottable(ss_e_vecs[:, 1])
plot_holder = GlobalCartopy(ax=ax0,adjustable=True)
XX,YY,ax0 = plot_holder.get_map()
XX,YY = cov_holder_ph.trans_geo.get_coords()
ax0.pcolor(XX,YY,ss_plot,vmin=vmin,vmax=vmax)
ax0.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)


ss_plot = dummy.trans_geo.transition_vector_to_plottable(ss_e_vecs[:, 7])
plot_holder = GlobalCartopy(ax=ax1,adjustable=True)
XX,YY,ax1 = plot_holder.get_map()
XX,YY = cov_holder_ph.trans_geo.get_coords()
pcm = ax1.pcolor(XX,YY,ss_plot,vmin=vmin,vmax=vmax)
ax1.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

fig.colorbar(pcm,ax=[ax0,ax1],pad=.05,label='Eigenvector Weight',location='top')

ax2 = plt.subplot(3,2,5)
ax2.plot(tt_e_vecs[:72,0])
ax2.set_ylabel("Eigen Vector Weight")
ax2.set_xlabel("Month Index")
ax2.annotate('c', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

ax3 = plt.subplot(3,2,6)
ax3.plot(tt_e_vecs[:72,9])
ax3.set_xlabel("Month Index")
ax3.get_yaxis().set_visible(False)
ax2.get_shared_y_axes().join(ax2, ax3)
ax3.annotate('d', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
plt.savefig(file_handler.out_file('Figure_5'),bbox_inches='tight')
plt.close()
