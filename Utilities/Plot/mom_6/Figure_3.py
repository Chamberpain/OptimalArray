from OptimalArray.Utilities.MOM6Mat import CovMOM6CCS,InverseCristina
from OptimalArray.Utilities.CorMat import CovElement
import matplotlib.pyplot as plt
import numpy as np 
from GeneralUtilities.Plot.Cartopy.regional_plot import CCSCartopy
import cartopy.crs as ccrs
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from TransitionMatrix.Utilities.TransGeo import get_cmap

plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'mom6_final_figures')

depths = CovMOM6CCS.get_depths()
depth_idx = 2
holder = CovMOM6CCS.load(depth_idx = depth_idx)

fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
plot_holder = CCSCartopy(ax=ax1)
XX,YY,ax = plot_holder.get_map()
ax.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
cov_holder = CovElement.load(holder.trans_geo, "chl", "o2")
XX,YY = cov_holder.trans_geo.get_coords()
holder1 = np.sqrt(CovElement.load(holder.trans_geo, "o2","o2"))
holder2 = np.sqrt(CovElement.load(holder.trans_geo, "chl","chl"))
plottable = cov_holder.diagonal()/(holder1.diagonal()*holder2.diagonal())
ax1.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
pcm = ax.pcolor(XX,YY,cov_holder.trans_geo.transition_vector_to_plottable(plottable),cmap='RdYlBu',vmin=-1.0,vmax=1.0)


ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
plot_holder = CCSCartopy(ax=ax2)
XX,YY,ax = plot_holder.get_map()
cov_holder = CovElement.load(holder.trans_geo,"thetao","o2")
XX,YY = cov_holder.trans_geo.get_coords()
ax.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10)
holder1 = np.sqrt(CovElement.load(holder.trans_geo, "o2","o2"))
holder2 = np.sqrt(CovElement.load(holder.trans_geo, "thetao","thetao"))
plottable = cov_holder.diagonal()/(holder1.diagonal()*holder2.diagonal())
ax2.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)


pcm = ax.pcolor(XX,YY,cov_holder.trans_geo.transition_vector_to_plottable(plottable),cmap='RdYlBu',vmin=-1.0,vmax=1.0)
fig.colorbar(pcm,ax=[ax1,ax2],pad=.05,label='Correlation',location='bottom')
plt.savefig(file_handler.out_file('Figure_3'),bbox_inches='tight')
plt.close()