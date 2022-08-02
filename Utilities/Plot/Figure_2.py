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

aggregate_argo_list(read_class=BGCReader)

def get_recent_sensor_pos(sensor,FloatClass=BGCReader):
	pos_list = FloatClass.get_recent_pos()
	sensor_list = FloatClass.get_sensors()
	date_list = FloatClass.get_recent_date_list()
	cutoff_date = max(date_list)-datetime.timedelta(days=180)
	date_mask = [x>=cutoff_date for x in date_list]
	sensor_mask = [sensor in x for x in sensor_list]
	mask = np.array(date_mask)&np.array(sensor_mask)
	pos_data = [x for x,y in zip(pos_list,mask) if y]
	return pos_data


class_list = [MODIS(),CM4O2(),Landschutzer(),RoemmichGilsonSal()]
data_list = [MODIS().compile_variance(0),CM4O2().compile_variance(2),Landschutzer().compile_variance(0),RoemmichGilsonSal().compile_variance(0)]
plot_list = [MODIS().compile_monthly_mean_and_variance(0,-40,-20),CM4O2().compile_monthly_mean_and_variance(2,-40,-20)
			,Landschutzer().compile_monthly_mean_and_variance(0,-40,-20),RoemmichGilsonSal().compile_monthly_mean_and_variance(0,-40,-20)]
var_list = ['CHLA','DOXY','PH_IN_SITU_TOTAL','PSAL']
vmin_list = [np.nanmin(x) for x in data_list]
vmin_list[0] = 10**(-3)
variable_dict = {'CHLA':'Chlorophyll','DOXY':'Oxygen','PH_IN_SITU_TOTAL':'pH','PSAL':'Salinity'}
label_dict = {'CHLA':'$mg\ m^{-3}$','DOXY':'$mol\ m^{-3}$','PH_IN_SITU_TOTAL':'$mol\ m^{-2}\ yr^{-1}$','PSAL':'PSU'}
annotate_list = ['a','b','c','d']
aggregate_argo_list(read_class=BGCReader)
fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(3, 2, 1, projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(3, 2, 2, projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(3, 2, 3, projection=ccrs.PlateCarree())
ax4 = fig.add_subplot(3, 2, 4, projection=ccrs.PlateCarree())
ax5 = fig.add_subplot(3, 1, 3)

ax_list = [ax1,ax2,ax3,ax4]

for ax,data_class,data,(mean,std),variable,vmin,anno in zip(ax_list,class_list,data_list,plot_list,var_list,vmin_list,annotate_list):
	if variable == 'PSAL':
		aggregate_argo_list(read_class=ArgoReader)

	holder = GlobalCartopy(ax=ax,adjustable=True)
	XX,YY,ax = holder.get_map()
	pos_list = get_recent_sensor_pos(variable)
	Y,X = zip(*[(x.latitude,x.longitude) for x in pos_list])
	ax.scatter(X,Y,zorder=12,s=5,c='pink')
	ax.scatter(-20,-40,zorder=13,s=90,c='red',marker='*')
	lons,lats = data_class.return_dimensions()
	XX,YY = np.meshgrid(lons,lats)
	print(data.min())
	print(data.max())
	pcm = ax.pcolormesh(XX,YY,data,zorder=1,norm=colors.LogNorm(vmin=vmin, vmax=data.max()))
	fig.colorbar(pcm,ax=ax,pad=.05,label=label_dict[variable],location='bottom')
	ax.annotate(anno, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	scaled_mean = 2*(mean-min(mean))/(max(mean)-min(mean))-1
	scaled_std = np.array(std)/abs(max(mean))
	ax5.plot(range(12),scaled_mean,label=variable_dict[variable])
	ax5.fill_between(range(12),scaled_mean-scaled_std,scaled_mean+scaled_std,alpha=0.2)
ax5.legend()
ax5.set_ylabel('Scaled Signal')
ax5.set_xlabel('Month')
ax5.set_xticklabels(['Jan','Mar','May','Jul','Sep','Nov'],rotation = 20)
ax5.set_xlim([0,11])
ax5.annotate('e', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

plt.savefig(file_handler.out_file('Figure_2'),bbox_inches='tight')
plt.close()