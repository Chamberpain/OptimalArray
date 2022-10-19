from TransitionMatrix.Utilities.ArgoData import Core, BGC
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader
from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import full_argo_list, aggregate_argo_list
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import numpy as np
from GeneralUtilities.Data.Mapped.modis import MODIS
from GeneralUtilities.Data.Mapped.landschutzer import Landschutzer
from GeneralUtilities.Data.Mapped.roemmich_gilson import RoemmichGilsonSal
from GeneralUtilities.Data.Mapped.cm4 import CM4O2

from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
import datetime
plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')


class_list = [MODIS(),CM4O2(),Landschutzer(),RoemmichGilsonSal()]
data_list = [MODIS().compile_variance(0),CM4O2().compile_variance(2),Landschutzer().compile_variance(0),RoemmichGilsonSal().compile_variance(0)]
var_list = ['CHLA','DOXY','PH_IN_SITU_TOTAL','PSAL']


median_list = [np.nanmedian(x) for x in data_list]
std_list = [np.std(x) for x in data_list]
upper_weighting_list = [1.5,1.5,1.5,1.5]
lower_weighting_list = [1.5,1.5,1.5,1.5]
vmin_list = [med-weight*std for med,std,weight in zip(median_list,std_list,lower_weighting_list)]
vmax_list = [med+weight*std for med,std,weight in zip(median_list,std_list,upper_weighting_list)]

variable_dict = {'CHLA':'Chlorophyll','DOXY':'Oxygen','PH_IN_SITU_TOTAL':'pH','PSAL':'Salinity'}
label_dict = {'CHLA':'$mg\ m^{-3}$','DOXY':'$mol\ m^{-3}$','PH_IN_SITU_TOTAL':'$mol\ m^{-2}\ yr^{-1}$','PSAL':'PSU'}
annotate_list = ['a','b','c','d']
aggregate_argo_list(read_class=BGCReader)
fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(4, 2, 1, projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(4, 2, 2, projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(4, 2, 3, projection=ccrs.PlateCarree())
ax4 = fig.add_subplot(4, 2, 4, projection=ccrs.PlateCarree())

ax_list = [ax1,ax2,ax3,ax4]

for ax,data_class,data,variable,vmin,vmax,anno in zip(ax_list,class_list,data_list,var_list,vmin_list,vmax_list,annotate_list):
	if variable == 'PSAL':
		aggregate_argo_list(read_class=ArgoReader)

	holder = GlobalCartopy(ax=ax,adjustable=True)
	XX,YY,ax = holder.get_map()

	lons,lats = data_class.return_dimensions()
	XX,YY = np.meshgrid(lons,lats)
	print(data.min())
	print(data.max())
	pcm = ax.pcolormesh(XX,YY,data,zorder=1,norm=colors.LogNorm(vmin=vmin, vmax=data.max()))
	fig.colorbar(pcm,ax=ax, location='bottom')
	ax.annotate(anno, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

plt.savefig(file_handler.out_file('dissertation_figure'),bbox_inches='tight')
plt.close()