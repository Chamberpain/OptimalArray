from OptimalArray.Utilities.CM4Mat import CovCM4Global,CovCM4NPacific
from OptimalArray.Utilities.H import HInstance, Float
import datetime
from OptimalArray.Utilities.Plot.Figure_11_15 import make_recent_float_H
import matplotlib.pyplot as plt
import numpy as np
from GeneralUtilities.Compute.list import VariableList,GeoList
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list,full_argo_list
from TransitionMatrix.Utilities.ArgoData import Core,BGC
from GeneralUtilities.Data.Filepath.instance import FilePathHandler, make_folder_if_does_not_exist
from OptimalArray.Data.__init__ import ROOT_DIR
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR as PLOT_DIR
from OptimalArray.Utilities.Utilities import make_P_hat
from TransitionMatrix.Utilities.TransMat import TransMat
from TransitionMatrix.Utilities.Utilities import shiftgrid
import cartopy.crs as ccrs
import scipy
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
from GeneralUtilities.Plot.Cartopy.regional_plot import NPacificCartopy
import geopy
from GeneralUtilities.Data.pickle_utilities import save,load
from TransitionMatrix.Utilities.TransGeo import get_cmap
import os
from GeneralUtilities.Plot.Cartopy.regional_plot import RegionalBase
import copy
import gc


plt.rcParams['font.size'] = '22'

class NPacificCartopy(RegionalBase):
    llcrnrlon=120.
    llcrnrlat=30.
    urcrnrlon=220.
    urcrnrlat=60.
    def __init__(self,*args,**kwargs):
        print('I am plotting N Pacific')
        super().__init__(*args,**kwargs)

class FutureFloatTrans(TransMat):
	def __init__(self, *args,**kwargs):
		super().__init__(*args,**kwargs)

	def setup(self,days):
		east_west, north_south = self.return_mean()
		self.days = days
		self.east_west = east_west
		self.north_south = north_south
		self.date = datetime.datetime(2021, 5, 10)+datetime.timedelta(days = days)

	def advance_float(self,dummy_float,lon_coords,lat_coords,model_trans_geo):

		try:
			idx = self.trans_geo.total_list.index(dummy_float.pos)
		except ValueError:
			print('the float was outside the coordinates of the transition_matrix, so it will stay put')
			return dummy_float			

		try:
			new_lon = dummy_float.pos.longitude + self.east_west[idx]
			if new_lon > 180:
				new_lon-=360
			if new_lon <-180:
				new_lon += 360
			new_lat = dummy_float.pos.latitude + self.north_south[idx]
			new_lon = lon_coords.find_nearest(new_lon)
			new_lat = lat_coords.find_nearest(new_lat)
			model_trans_geo.total_list.index(geopy.Point(new_lat,new_lon))

		except ValueError:
			print('the projected float was outside the coordinates of the model, so it will go to closest point')
			idx = self.trans_geo.total_list.index(dummy_float.pos)
			new_lon = dummy_float.pos.longitude + self.east_west[idx]
			new_lat = dummy_float.pos.latitude + self.north_south[idx]
			new_pos = geopy.Point(new_lat,new_lon)
			dist_list = [geopy.distance.great_circle(new_pos,x) for x in model_trans_geo.total_list]
			idx = dist_list.index(min(dist_list))
			new_pos = model_trans_geo.total_list[idx]
			new_lat = new_pos.latitude
			new_lon = new_pos.longitude
		new_float = Float(geopy.Point(new_lat,new_lon),dummy_float.sensors,date_deployed=dummy_float.date_deployed,ID=dummy_float.ID)
		dist = geopy.distance.great_circle(dummy_float.pos,new_float.pos)
		if dist==0:
			print('the float didnt move')
		else:
			print('the float moved from ',dummy_float.pos)
			print('to ',new_float.pos)
		assert geopy.distance.great_circle(dummy_float.pos,new_float.pos).km<(self.days*15)
		return new_float			

	def advance_H(self,H):
		new_H = HInstance(trans_geo = H.trans_geo)
		lats,lons = H.trans_geo.total_list.lats_lons()
		for dummy_float in H._list_of_floats:
			if self.date>dummy_float.date_death:
				continue
			new_H.add_float(self.advance_float(dummy_float,lons,lats,H.trans_geo))
		return new_H


def make_movie_base(cov_holder,future_H_list,label):
	data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgo/'+label)
	plot_handler = FilePathHandler(PLOT_DIR,'final_figures/'+label+'_'+str(cov_holder.trans_geo.depth_idx))
	label_translation_dict = {'ph':'pH Equipped','chl':'Chlorophyll Equipped','o2':'Oxygen Equipped'}
	for k,future_H in enumerate(future_H_list):
		print(label)
		print('working on ',k)
		filename = data_file_handler.tmp_file(str(cov_holder.trans_geo.depth_idx)+'_'+str(k)+'_'+label)
		try:
			out= load(filename)
			print('loaded data')
		except FileNotFoundError:
			print('could not load, calculating...')
			p_hat_array = make_P_hat(cov_holder.cov,future_H,noise_factor=4)
			out = np.split(p_hat_array.diagonal(),len(cov_holder.trans_geo.variable_list))
			save(filename,out)
			del p_hat_array
		lat_core,lon_core = future_H.return_pos_of_core().lats_lons()
		lat_bgc,lon_bgc = future_H.return_pos_of_bgc().lats_lons()
		for data,scale,var in zip(out,np.split(cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list)),cov_holder.trans_geo.variable_list):
			fig = plt.figure(figsize=(14,11))
			ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
			plot_holder = GlobalCartopy(ax=ax,adjustable=True)
			XX,YY,ax0 = plot_holder.get_map()
			XX,YY = H_array.trans_geo.get_coords()
			data = H_array.trans_geo.transition_vector_to_plottable(data/scale)
			ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
			pcm = ax0.pcolormesh(XX,YY,data,cmap='YlOrBr',vmin=0,vmax=1,transform=ccrs.PlateCarree())
			ax0.scatter(lon_core,lat_core,c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
			ax0.scatter(lon_bgc,lat_bgc,c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
			if var in ['ph','chl','o2']:
				var_pos = future_H.return_pos_of_variable(var)
				lat_var,lon_var = var_pos.lats_lons()
				ax0.scatter(lon_var,lat_var,c='cyan',s=BGC.marker_size,zorder=12,label = label_translation_dict[var],transform=ccrs.PlateCarree())
			plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=3, mode="expand", borderaxespad=0.)
			fig.colorbar(pcm,pad=-0.05,label='Mapping Error',location='bottom')
			make_folder_if_does_not_exist(plot_handler.tmp_file(var))
			plt.savefig(plot_handler.tmp_file(var+'/'+str(k)),bbox_inches='tight')
			plt.close()
	for var in cov_holder.trans_geo.variable_list:
		os.chdir(plot_handler.tmp_file(var+'/'))
		os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")


def make_movie_float(cov_holder,future_H_list,label,nn):
	data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgo/'+label)
	plot_handler = FilePathHandler(PLOT_DIR,'final_figures/'+label+'_'+str(cov_holder.trans_geo.depth_idx))
	label_translation_dict = {'ph':'pH Equipped','chl':'Chlorophyll Equipped','o2':'Oxygen Equipped'}
	for k,future_H in enumerate(future_H_list):
		print(label)
		print('working on ',k)
		filename = data_file_handler.tmp_file(str(cov_holder.trans_geo.depth_idx)+'_'+str(k)+'_'+label)
		try:
			out= load(filename)
			print('loaded data')
		except FileNotFoundError:
			print('could not load, calculating...')
			p_hat_array = make_P_hat(cov_holder.cov,future_H,noise_factor=4)
			out = np.split(p_hat_array.diagonal(),len(cov_holder.trans_geo.variable_list))
			save(filename,out)
			del p_hat_array
		lat_core,lon_core = future_H.return_pos_of_core().lats_lons()
		lat_bgc,lon_bgc = future_H.return_pos_of_bgc().lats_lons()
		for data,scale,var in zip(out,np.split(cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list)),cov_holder.trans_geo.variable_list):
			fig = plt.figure(figsize=(14,11))
			ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree(central_longitude=180))
			plot_holder = NPacificCartopy(ax=ax)
			XX,YY,ax0 = plot_holder.get_map()
			XX,YY = H_array.trans_geo.get_coords()
			data1 = H_array.trans_geo.transition_vector_to_plottable(data/scale)
			lons = XX[0,:]
			lats = YY[:,0]
			data1,lons = shiftgrid(0,data1,lons,cyclic = 358)
			XX,YY = np.meshgrid(lons,lats)
			ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
			pcm = ax0.pcolormesh(XX,YY,data1,cmap='YlOrBr',vmin=0,vmax=1,transform=ccrs.PlateCarree())
			ax0.scatter(lon_core,lat_core,c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
			ax0.scatter(lon_bgc,lat_bgc,c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
			float_pos = GeoList([x.pos for x in future_H._list_of_floats[nn:]])
			lat_floats,lon_floats = float_pos.lats_lons()
			ax0.scatter(lon_floats,lat_floats,c='cyan',s=BGC.marker_size,zorder=12,label = 'Hypothetical Floats',transform=ccrs.PlateCarree())
			plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=3, mode="expand", borderaxespad=0.)
			fig.colorbar(pcm,label='Scaled Unconstrained Variance',location='bottom')
			make_folder_if_does_not_exist(plot_handler.tmp_file(var))
			plt.savefig(plot_handler.tmp_file(var+'/'+str(k)),bbox_inches='tight')
			plt.close()
	for var in cov_holder.trans_geo.variable_list:
		os.chdir(plot_handler.tmp_file(var+'/'))
		os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")



trans_mat = FutureFloatTrans.load_from_type(lat_spacing=2,lon_spacing=2,time_step=90)
trans_list = []
for x in range(16):
	trans_holder = trans_mat.multiply(x,value=0.00001)
	trans_holder.setup(days = 90*(1+x))
	trans_list.append(trans_holder)


def make_movies():
	for depth in [2,4,6,8,10,12,14,16,18,20,22,24]:
		cov_holder = CovCM4Global.load(depth_idx = depth)
		H_array = make_recent_float_H(cov_holder.trans_geo) # need to make H_array with same grid spacing as trans matrix
		future_H_list = [x.advance_H(H_array) for x in trans_list]
		[set(y.get_id_list()).issubset(set(x.get_id_list())) for x,y in zip(future_H_list[:-1],future_H_list[1:])]
		make_movie_base(cov_holder,future_H_list,'base')

		del cov_holder
		gc.collect(generation=2)
		cov_holder = CovCM4NPacific.load(depth_idx = depth)
		remove_idx_list = [k for k,x in enumerate(H_array._list_of_floats) if x.pos not in cov_holder.trans_geo.total_list]
		for idx in remove_idx_list[::-1]:
			H_array.remove_by_index(idx)
		H_array_new = HInstance(cov_holder.trans_geo)
		for float_ in H_array._list_of_floats:
			H_array_new.add_float(float_)
		H_array = H_array_new
		H_array.add_float(Float(geopy.Point(42,146),cov_holder.trans_geo.variable_list,datetime.datetime(2021,5,10)))

		future_H_list = [x.advance_H(H_array) for x in trans_list]
		make_movie_float(cov_holder,future_H_list,'float_1',-1)
		H_array._list_of_floats.pop()
		H_array._index_of_pos.pop()
		H_array._index_of_sensors.pop()
		H_array.add_float(Float(geopy.Point(46,-160),cov_holder.trans_geo.variable_list,datetime.datetime(2021,5,10)))
		future_H_list = [x.advance_H(H_array) for x in trans_list]
		make_movie_float(cov_holder,future_H_list,'float_2',-1)
		H_array.add_float(Float(geopy.Point(42,146),cov_holder.trans_geo.variable_list,datetime.datetime(2021,5,10)))
		future_H_list = [x.advance_H(H_array) for x in trans_list]
		make_movie_float(cov_holder,future_H_list,'both_floats',-2)
		del cov_holder
		del H_array
		del future_H_list
		gc.collect(generation=2)

def make_plots():
	data_file_handler = FilePathHandler(ROOT_DIR,'FutureArgo/base')
	plot_handler = FilePathHandler(PLOT_DIR,'final_figures')
	label_translation_dict = {'ph':'pH Equipped','chl':'Chlorophyll Equipped','o2':'Oxygen Equipped'}
	depth = 2
	cov_holder = CovCM4Global.load(depth_idx = depth)
	H_array = make_recent_float_H(cov_holder.trans_geo) # need to make H_array with same grid spacing as trans matrix
	future_H_list = [x.advance_H(H_array) for x in [trans_list[4],trans_list[12]]]
	annotate_list = ['a','b']
	for i,(scale,var) in enumerate(zip(np.split(cov_holder.cov.diagonal(),len(cov_holder.trans_geo.variable_list)),cov_holder.trans_geo.variable_list)):
		fig = plt.figure(figsize=(14,14))
		ax_list = []
		for k,(jj,future_H) in enumerate(zip([4,12],future_H_list)):
			print('working on ',k)
			filename = data_file_handler.tmp_file(str(cov_holder.trans_geo.depth_idx)+'_'+str(jj)+'_base')
			out= load(filename)
			data = out[i]
			lat_core,lon_core = future_H.return_pos_of_core().lats_lons()
			lat_bgc,lon_bgc = future_H.return_pos_of_bgc().lats_lons()
			ax = fig.add_subplot(2,1,(k+1), projection=ccrs.PlateCarree())
			ax_list.append(ax)
			plot_holder = GlobalCartopy(ax=ax,adjustable=True)
			XX,YY,ax0 = plot_holder.get_map()
			XX,YY = H_array.trans_geo.get_coords()
			data = H_array.trans_geo.transition_vector_to_plottable(data/scale)
			ax0.pcolormesh(XX,YY,XX != np.nan,cmap=get_cmap(),alpha=0.7,zorder=-10,transform=ccrs.PlateCarree())
			pcm = ax0.pcolormesh(XX,YY,data,cmap='YlOrBr',vmin=0,vmax=1,transform=ccrs.PlateCarree())
			ax0.scatter(lon_core,lat_core,c='green',s=Core.marker_size,zorder=11,label = 'Core',transform=ccrs.PlateCarree())
			ax0.scatter(lon_bgc,lat_bgc,c='blue',s=BGC.marker_size,zorder=11,label = 'BGC',transform=ccrs.PlateCarree())
			if var in ['ph','chl','o2']:
				var_pos = future_H.return_pos_of_variable(var)
				lat_var,lon_var = var_pos.lats_lons()
				ax0.scatter(lon_var,lat_var,c='cyan',s=BGC.marker_size,zorder=12,label = label_translation_dict[var],transform=ccrs.PlateCarree())
			ax0.annotate(annotate_list[k], xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
		plt.legend(ncol=3,bbox_to_anchor=(.0, 1.0, .0, 1.0), loc=3)
		fig.colorbar(pcm,ax=ax_list,pad=.05,label='Scaled Unconstrained Variance',location='right')
		fig_num = 17+i
		plt.savefig(plot_handler.out_file('Figure_'+str(fig_num)),bbox_inches='tight')
		plt.close()