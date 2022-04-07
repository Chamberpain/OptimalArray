from OptimalArray.Utilities.CorMat import CovCM4Global
from OptimalArray.Utilities.H import HInstance
import matplotlib.pyplot as plt
import numpy as np 
import uuid
from GeneralUtilities.Compute.list import GeoList
import cartopy.crs as ccrs
import os
from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import full_argo_list

full_argo_list()

def plot_variance_constrained(trans_geo,data)
	from matplotlib.colors import ListedColormap
	colormap = plt.get_cmap('viridis')
	colormap_r = ListedColormap(colormap.colors[::-1])
	fig = plt.figure(figsize=(17,7))
	ax1 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
	XX,YY,ax = dummy.trans_geo.plot_setup(ax = ax1)
	ax.pcolor(XX,YY,dummy.trans_geo.transition_vector_to_plottable(1-p_hat_data/dummy.trans_geo.return_variance('ph',dummy.cov))*100,
		vmin=0,vmax=95,cmap = colormap_r)
	PCM=ax.get_children()[3]
	plt.colorbar(PCM, ax=ax, label='% pH Variance Constrained') 
	float_idx = np.array_split(H.sum(axis=0).ravel().tolist()[0],len(dummy.trans_geo.variable_list))
	ts_idx = np.where(float_idx[0]>0)
	ph_idx = np.where(float_idx[dummy.trans_geo.variable_list.index('ph')]>0)
	for idxs,markersize,markercolor,label in [(ts_idx,4,'r','Core'),(ph_idx,20,'b','BGC')]:
		float_list = GeoList([dummy.trans_geo.total_list[x] for x in idxs[0].tolist()])
		lats,lons = float_list.lats_lons()
		ax.scatter(lons,lats,s=markersize,c=markercolor,label=label)


for depth in [2,4,6,8,10,12,14,16,18,20,22,24,26]:
	dummy = CovCM4Global.load(depth_idx = depth)
	H,index_list = HInstance.recent_floats(dummy.trans_geo,BGCReader)
	p_hat,cov_subtract = dummy.p_hat_calculate(H,index_list,noise_factor=2.5)
	p_hat_data = dummy.trans_geo.return_variance("ph", p_hat)
	cov_data = dummy.trans_geo.return_variance("ph", dummy.cov)




	float_idx = np.array_split(H.sum(axis=0).ravel().tolist()[0],len(dummy.trans_geo.variable_list))
	ts_idx = np.where(float_idx[0]>0)
	ph_idx = np.where(float_idx[dummy.trans_geo.variable_list.index('ph')]>0)
	for idxs,markersize,markercolor,label in [(ts_idx,10,'r','Core'),(ph_idx,20,'b','BGC')]:
		float_list = GeoList([dummy.trans_geo.total_list[x] for x in idxs[0].tolist()])
		lats,lons = float_list.lats_lons()
		ax.scatter(lons,lats,s=markersize,c=markercolor,label=label)
