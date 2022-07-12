from TransitionMatrix.Utilities.ArgoData import Core, BGC
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader
from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import full_argo_list, aggregate_argo_list
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
import matplotlib.pyplot as plt
from GeneralUtilities.Compute.list import GeoList
import cartopy.crs as ccrs
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from GeneralUtilities.Data.pickle_utilities import load
from OptimalArray.Utilities.CM4Mat import CovCM4Global

plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')

aggregate_argo_list(read_class=BGCReader)
bgc_locations = BGC.recent_pos_list(BGCReader)

full_argo_list()
core_locations = Core.recent_pos_list(ArgoReader)
H_array, p_hat = load('//Users/paulchamberlain/Projects/OptimalArray/Data/RandomArray/tmp/random_1000_2_6')
cov_holder = CovCM4Global.load(depth_idx = 2)
float_list = GeoList(cov_holder.trans_geo.total_list[x] for x in H_array)

fig = plt.figure(figsize=(17,14))
ax = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
holder = GlobalCartopy(ax=ax)
XX,YY,ax = holder.get_map()

Y,X = zip(*[(x.latitude,x.longitude) for x in bgc_locations])
ax.scatter(X,Y,c='blue',s=BGC.marker_size,zorder=10,label='BGC Float')
Y,X = zip(*[(x.latitude,x.longitude) for x in core_locations])
ax.scatter(X,Y,c='green',s=Core.marker_size,zorder=9, label='Core Float')
ax.legend(loc='upper right',fancybox=True)
ax.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)


ax = fig.add_subplot(2,1 , 2, projection=ccrs.PlateCarree())



lats,lons = float_list.lats_lons()
plot_holder = GlobalCartopy(ax=ax)
XX,YY,ax0 = plot_holder.get_map()
XX,YY = cov_holder.trans_geo.get_coords()

ax.scatter(lons,lats,c='blue',s=BGC.marker_size,zorder=10)
ax.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
plt.savefig(file_handler.out_file('Figure_1'),bbox_inches='tight')
