from OptimalArray.Utilities.CM4Mat import CovCM4NPacific
from OptimalArray.Utilities.H import HInstance, Float
from OptimalArray.Utilities.Plot.Figure_11_15 import make_recent_float_H
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list,full_argo_list
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from OptimalArray.Data.__init__ import ROOT_DIR
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR as PLOT_DIR
from TransitionMatrix.Utilities.Utilities import shiftgrid
import cartopy.crs as ccrs
from OptimalArray.Utilities.Plot.Figure_17_21 import NPacificCartopy
import geopy
from GeneralUtilities.Data.pickle_utilities import load
from TransitionMatrix.Utilities.TransMat import TransMat
from TransitionMatrix.Utilities.TransGeo import get_cmap
import datetime
import matplotlib.pyplot as plt
from OptimalArray.Utilities.Plot.Figure_17_21 import FutureFloatTrans
import numpy as np
from TransitionMatrix.Utilities.ArgoData import Core,BGC
from GeneralUtilities.Compute.list import GeoList
from GeneralUtilities.Plot.Cartopy.regional_plot import RegionalBase
from OptimalArray.Utilities.CorGeo import InverseNPacific

plt.rcParams['font.size'] = '18'

plot_handler = FilePathHandler(PLOT_DIR,'final_figures')

class NPacificCartopy(RegionalBase):
    llcrnrlon=120.
    llcrnrlat=30.
    urcrnrlon=220.
    urcrnrlat=60.
    def __init__(self,*args,**kwargs):
        print('I am plotting N Pacific')
        super().__init__(*args,**kwargs)




trans_mat = FutureFloatTrans.load_from_type(lat_spacing=2,lon_spacing=2,time_step=90)
trans_list = []
for x in range(16):
	trans_holder = trans_mat.multiply(x,value=0.00001)
	trans_holder.setup(days = 90*(1+x))
	trans_list.append(trans_holder)

cov_holder = CovCM4NPacific.load(depth_idx = 2)
H_array = make_recent_float_H(cov_holder.trans_geo) # need to make H_array with same grid spacing as trans matrix
H_array.add_float(Float(geopy.Point(42,146),cov_holder.trans_geo.variable_list,datetime.datetime(2021,5,10)))
H_array.add_float(Float(geopy.Point(46,-160),cov_holder.trans_geo.variable_list,datetime.datetime(2021,5,10)))
future_H_list = [x.advance_H(H_array) for x in trans_list]
label = 'both_floats'
data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgo/'+label)

filename = data_file_handler.tmp_file(str(cov_holder.trans_geo.depth_idx)+'_'+str(15)+'_'+label)
out = load(filename)

float_1 = GeoList([x._list_of_floats[-2].pos for x in future_H_list])
float_2 = GeoList([x._list_of_floats[-1].pos for x in future_H_list])

data_dict = {}
depth_list = [2,4,6,8,10,12,14,16,18,20,22,24]
depths = cov_holder.get_depths().data[depth_list,0]
annotate_list = ['b','c','d','e','f']
colorbar_label = ['$(^\circ C)^2$','$(PSU)^2$','','$(kg\ m^{-3})^2$','$(mol\ m^{-3})^2$']

for i,depth_idx in enumerate(depth_list):
	dummy = InverseNPacific(depth_idx = depth_idx)
	datascale = load(dummy.make_datascale_filename())
	data,var = zip(*datascale)
	datascale_dict = dict(zip(var,data))

	for k in range(16):
		print(i)
		var_list = cov_holder.trans_geo.variable_list[:]
		label = 'float_1'
		data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgo/'+label)
		filename = data_file_handler.tmp_file(str(depth_idx)+'_'+str(k)+'_'+label)
		out1 = load(filename)
		label = 'float_2'
		data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgo/'+label)
		filename = data_file_handler.tmp_file(str(depth_idx)+'_'+str(k)+'_'+label)
		out2 = load(filename)
		if depth_idx>=cov_holder.chl_depth_idx:
			var_list.remove('chl')
		for var,data1,data2 in zip(var_list,out1,out2):
			data1=data1*datascale_dict[var]
			data2=data2*datascale_dict[var]
			try:
				data_dict[var].append((i,k,np.sum(data1-data2)))
			except KeyError:
				data_dict[var]=[]
				data_dict[var].append((i,k,np.sum(data1-data2)))

float_1_col = 'mediumseagreen'
float_2_col = 'indigo'
colormap = 'PRGn'
marker_increase = 200
linewidth_increase = 5


fig = plt.figure(figsize=(14,14))
future_H = future_H_list[-1]
lat_core,lon_core = future_H.return_pos_of_core().lats_lons()
lat_bgc,lon_bgc = future_H.return_pos_of_bgc().lats_lons()
ax = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree(central_longitude=180))
plot_holder = NPacificCartopy(ax=ax,adjustable=True)
XX,YY,ax0 = plot_holder.get_map()
XX,YY = H_array.trans_geo.get_coords()
data1 = cov_holder.trans_geo.transition_vector_to_plottable(out[2])
lons = XX[0,:]
lats = YY[:,0]
data1,lons = shiftgrid(0,data1,lons,cyclic = 358)
XX,YY = np.meshgrid(lons,lats)
ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
pcm = ax0.pcolormesh(XX,YY,data1,cmap='YlOrBr',vmin=0,vmax=1,transform=ccrs.PlateCarree())
ax0.scatter(lon_core,lat_core,c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
ax0.scatter(lon_bgc,lat_bgc,c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
y,x = float_1.lats_lons()
ax0.plot(x,y,c=float_1_col,linewidth=linewidth_increase,zorder=11,transform=ccrs.PlateCarree())
ax0.scatter(x[0],y[0],c=float_1_col,s=BGC.marker_size+marker_increase,zorder=11,transform=ccrs.PlateCarree())
ax0.scatter(x[-1],y[-1],c=float_1_col,s=BGC.marker_size+marker_increase,marker='*',zorder=11,label = 'Float 1',transform=ccrs.PlateCarree())
y,x = float_2.lats_lons()
ax0.plot(x,y,c=float_2_col,linewidth=linewidth_increase,zorder=11,transform=ccrs.PlateCarree())
ax0.scatter(x[0],y[0],c=float_2_col,s=BGC.marker_size+marker_increase,zorder=11,transform=ccrs.PlateCarree())
ax0.scatter(x[-1],y[-1],c=float_2_col,s=BGC.marker_size+marker_increase,marker='*',zorder=11,label = 'Float 2',transform=ccrs.PlateCarree())
cbar = fig.colorbar(pcm, orientation="horizontal", pad=0.14)
cbar.ax.set_xlabel('Scaled Unconstrained Variance')
cbar.ax.xaxis.set_label_coords(0.5,1.6)
ax0.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
          ncol=3, mode="expand", borderaxespad=0.)



x_offset = 8
label_offset_list = [(x_offset,0.5),(x_offset,0.5),(x_offset,0.5),(x_offset,0.6),(x_offset+8,0.5)]
ax_list = []
for kk,var in enumerate(cov_holder.trans_geo.variable_list):
	ax = fig.add_subplot(6,2,(kk+7))
	ax_list.append(ax)
	depth_idx_list,timestep_idx_list,data = zip(*data_dict[var])
	depth_idx_list = np.unique(depth_idx_list)
	timestep_idx_list = np.unique(timestep_idx_list)

	dummy_depths = depths[:len(depth_idx_list)]
	dummy_times = [3*(x+1) for x in range(len(timestep_idx_list))]

	XX,YY = np.meshgrid(dummy_times,dummy_depths)

	dummy_array = np.zeros([len(depth_idx_list),len(timestep_idx_list)])
	for ii,jj,data in data_dict[var]:
		dummy_array[ii,jj] = data
	max_val = abs(dummy_array).max()
	pcm = ax.pcolor(XX,YY,dummy_array,cmap=colormap,vmin=-max_val,vmax=max_val)
	ax.annotate(annotate_list[kk], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.gca().invert_yaxis()
	ax.get_xaxis().set_visible(False)
	ax.set_ylabel('Depth (m)')
	ax.set_yscale('log')
	cb = plt.colorbar(pcm)
	cb.ax.set_ylabel(colorbar_label[kk],rotation=90)
	cb.ax.yaxis.set_label_coords(label_offset_list[kk][0],label_offset_list[kk][1])
ax.get_xaxis().set_visible(True)
ax.set_xlabel('Time in Future (months)')
ax_list[3].get_xaxis().set_visible(True)
ax_list[3].set_xlabel('Time in Future (months)')
plt.subplots_adjust(hspace=0.2)
plt.subplots_adjust(wspace=.45)

plt.savefig(plot_handler.out_file('Figure_23'))
plt.close()