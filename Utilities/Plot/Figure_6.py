from OptimalArray.Utilities.CM4Mat import CovCM4Global
from OptimalArray.Utilities.CorMat import CovElement
import matplotlib.pyplot as plt
import numpy as np 
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
import cartopy.crs as ccrs
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
from TransitionMatrix.Utilities.TransGeo import get_cmap
plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')

depths = CovCM4Global.get_depths()

covclass = CovCM4Global
depth = 6
shallow_dummy = covclass(depth_idx = depth)
deep_dummy = covclass(depth_idx = 18)


cov_holder_ph = CovElement.load(shallow_dummy.trans_geo, "ph", "ph")
ss_e_vals,shallow_ss_e_vecs = np.linalg.eig(cov_holder_ph)

cov_holder_ph = CovElement.load(deep_dummy.trans_geo, "ph", "ph")
ss_e_vals,deep_ss_e_vecs = np.linalg.eig(cov_holder_ph)


vmin = -0.025
vmax = 0.025
f, (ax0, ax1) = plt.subplots(2, 1,figsize=(15,12),subplot_kw={'projection': ccrs.PlateCarree()})

ss_plot = dummy.trans_geo.transition_vector_to_plottable(shallow_ss_e_vecs[:, 1])
plot_holder = GlobalCartopy(ax=ax0,adjustable=True)
XX,YY,ax0 = plot_holder.get_map()
ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
XX,YY = cov_holder_ph.trans_geo.get_coords()
ax0.pcolor(XX,YY,ss_plot,vmin=vmin,vmax=vmax,cmap='RdYlBu')
ax0.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)


ss_plot = dummy.trans_geo.transition_vector_to_plottable(deep_ss_e_vecs[:, 7])
plot_holder = GlobalCartopy(ax=ax1,adjustable=True)
XX,YY,ax1 = plot_holder.get_map()
ax1.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
XX,YY = cov_holder_ph.trans_geo.get_coords()
ax1.pcolor(XX,YY,ss_plot,vmin=vmin,vmax=vmax,cmap='RdYlBu')
ax1.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
fig.colorbar(pcm,ax=[ax0,ax1],pad=.05,label='Eigenvector Weight',location='right')
plt.savefig(file_handler.out_file('Figure_6'),bbox_inches='tight')
plt.close()
