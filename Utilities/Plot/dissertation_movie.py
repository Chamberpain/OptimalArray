from GeneralUtilities.Data.Mapped.cm4 import CM4O2,CM4ThetaO,CM4CHL
import numpy as np
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs
import cmocean


plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'dissertation_movie')

class_list = [CM4O2(),CM4ThetaO(),CM4CHL()]
o2_data, lons = class_list[0].return_dataset()
theta0_data, lons = class_list[1].return_dataset()
chl_data, lons = class_list[2].return_dataset()
lons, lats = class_list[0].return_dimensions()


label_list = ['$mol\ m^{-3}$','$^\circ C$','$mg\ m^{-3}$']
data_list = [o2_data,theta0_data,chl_data]
cm_list = [cmocean.cm.oxy,cmocean.cm.thermal,cmocean.cm.algae]
median_list = [np.median(x) for x in data_list]
std_list = [np.std(x) for x in data_list]
upper_weighting_list = [.8,0.7,1.2]
lower_weighting_list = [1.6,2,2]
vmin_list = [med-weight*std for med,std,weight in zip(median_list,std_list,lower_weighting_list)]
vmax_list = [med+weight*std for med,std,weight in zip(median_list,std_list,upper_weighting_list)]


def make_field_movie():
	for k in range(0,100):
		fig = plt.figure(figsize=(40,10))
		ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
		ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
		ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
		ax_list = [ax1,ax2,ax3]
		for ax,data,cm,vmin,vmax,label in zip(ax_list,data_list,cm_list,vmin_list,vmax_list,label_list):
			holder = GlobalCartopy(ax=ax,adjustable=True)
			XX,YY,ax = holder.get_map()
			XX,YY = np.meshgrid(lons,lats)
			pcm = ax.pcolormesh(XX,YY,data[k,:,:],zorder=1,cmap=cm,vmin=max([0,vmin]), vmax=vmax)
			fig.colorbar(pcm,ax=ax, location='bottom',label=label)
		plt.savefig(file_handler.out_file('')+'/field/'+str(k),bbox_inches='tight')
		plt.close()

def make_spatial_movie():
	loc_1 = (-30.5,-150.5)
	lat_1,lon_1 = loc_1
	lat_idx_1 = lats.index(lat_1)
	lon_idx_1 = lons.index(lon_1)
	loc_2 = (-30.5,-155.5)
	lat_2,lon_2 = loc_2
	lat_idx_2 = lats.index(lat_2)
	lon_idx_2 = lons.index(lon_2)
	time_series_1 = o2_data[:,lon_idx_1,lat_idx_1]
	time_series_2 = o2_data[:,lon_idx_2,lat_idx_2]
	for k in range(0,100):
		fig = plt.figure(figsize=(30,10))
		ax1 = fig.add_subplot(1, 1, 1)
		ax1.plot(time_series_1[:k],label='30S 150W Oxygen')
		ax1.plot(time_series_2[:k],label='30S 155W Oxygen')
		ax1.set_xlabel('Months')
		ax1.set_ylabel('$mol\ m^{-3}$')
		ax1.set_ylim(min(list(time_series_1)+list(time_series_2)),max(list(time_series_1)+list(time_series_2)))
		ax1.set_xlim(0,100)
		plt.legend(loc=2)
		plt.savefig(file_handler.out_file('')+'/spatial/'+str(k),bbox_inches='tight')
		plt.close()


def make_variable_movie():
	loc_1 = (-30.5,-150.5)
	lat_1,lon_1 = loc_1
	lat_idx_1 = lats.index(lat_1)
	lon_idx_1 = lons.index(lon_1)
	time_series_1 = o2_data[:,lon_idx_1,lat_idx_1]
	time_series_2 = theta0_data[:,lon_idx_1,lat_idx_1]
	for k in range(0,100):
		fig = plt.figure(figsize=(30,10))
		ax1 = fig.add_subplot(1, 1, 1)
		ax1.set_xlim(0,100)
		ax2 = ax1.twinx()
		ln1 = ax1.plot(time_series_1[:k],'b',label='30S 150W Oxygen')
		ln2 = ax2.plot(time_series_2[:k],'r',label='30S 150W Temperature')
		ax1.set_xlabel('Months')
		ax1.set_ylabel('$mol\ m^{-3}$',color="blue")
		ax1.yaxis.label.set_color('blue')
		ax1.tick_params(axis='y', colors='blue')

		ax2.set_ylabel('$^\circ C$',color="red")
		ax2.yaxis.label.set_color('red')
		ax2.tick_params(axis='y', colors='red')

		ax1.set_ylim([time_series_1.min(),time_series_1.max()])
		ax2.set_ylim([time_series_2.min(),time_series_2.max()])
		lns = ln1+ln2
		labs = [l.get_label() for l in lns]
		ax1.legend(lns, labs, loc=0)
		plt.savefig(file_handler.out_file('')+'/variable/'+str(k),bbox_inches='tight')
		plt.close()
