from GeneralUtilities.Data.Mapped.cm4 import CM4O2,CM4Sal,CM4CHL
import numpy as np
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs
from matplotlib.colors import LogNorm
import cmocean


plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'dissertation_movie')

class_list = [CM4O2(),CM4Sal(),CM4CHL()]
o2_data, lons = class_list[0].return_dataset()
sal_data, lons = class_list[1].return_dataset()
chl_data, lons = class_list[2].return_dataset()
lons, lats = class_list[0].return_dimensions()
cm_list = [cmocean.cm.oxy,cmocean.cm.thermal,cmocean.cm.algae]


label_list = ['$(mol\ m^{-3})^2$','$(PSU)^2$','$(mg\ m^{-3})^2$']
data_list = [o2_data.var(axis=0),theta0_data.var(axis=0),chl_data.var(axis=0)]
median_list = [np.median(x) for x in data_list]
std_list = [np.std(x) for x in data_list]
upper_weighting_list = [.01,2,2]
lower_weighting_list = [.001,.1,.001]
vmin_list = [med-weight*std for med,std,weight in zip(median_list,std_list,lower_weighting_list)]

vmax_list = [med+weight*std for med,std,weight in zip(median_list,std_list,upper_weighting_list)]

fig = plt.figure(figsize=(10,15))
ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
ax_list = [ax1,ax2,ax3]
for cm,ax,data,vmin,vmax,label in zip(cm_list,ax_list,data_list,vmin_list,vmax_list,label_list):
	holder = GlobalCartopy(ax=ax,adjustable=True)
	XX,YY,ax = holder.get_map()
	XX,YY = np.meshgrid(lons,lats)
	pcm = ax.pcolormesh(XX,YY,data[:,:],cmap=cm,zorder=1,norm=LogNorm(vmin=vmin, vmax=vmax))
	fig.colorbar(pcm,ax=ax,label=label,fraction=0.046, pad=0.04)
plt.savefig(file_handler.out_file('var'),bbox_inches='tight')
plt.close()
