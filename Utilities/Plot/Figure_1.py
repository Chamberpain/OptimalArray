from TransitionMatrix.Utilities.ArgoData import Core, BGC
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader
from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import full_argo_list, aggregate_argo_list
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')

aggregate_argo_list(read_class=BGCReader)
bgc_locations = BGC.recent_pos_list(BGCReader)


full_argo_list()
core_locations = Core.recent_pos_list(ArgoReader)

fig = plt.figure(figsize=(17,12))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
holder = GlobalCartopy(ax=ax)
XX,YY,ax = holder.get_map()

Y,X = zip(*[(x.latitude,x.longitude) for x in bgc_locations])
ax.scatter(X,Y,c='blue',s=BGC.marker_size,zorder=10,label='BGC Float')
Y,X = zip(*[(x.latitude,x.longitude) for x in core_locations])
ax.scatter(X,Y,c='green',s=Core.marker_size,zorder=9, label='Core Float')
ax.legend(loc='upper right',fancybox=True)
plt.savefig(file_handler.out_file('Figure_1'),bbox_inches='tight')