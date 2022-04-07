from OptimalArray.Utilities.CorMat import CovCM4Global,CovCM4Indian,CovCM4SO,CovCM4NAtlantic,CovCM4TropicalAtlantic,CovCM4SAtlantic,CovCM4NPacific,CovCM4TropicalPacific,CovCM4SPacific,CovCM4GOM,CovCM4CCS
from OptimalArray.Utilities.H import HInstance
import matplotlib.pyplot as plt
import numpy as np 
import uuid
from GeneralUtilities.Compute.list import GeoList
import cartopy.crs as ccrs
import os

def plot_for_slide():
	from matplotlib.colors import ListedColormap
	colormap = plt.get_cmap('viridis')
	colormap_r = ListedColormap(colormap.colors[::-1])

	covclass = CovCM4NAtlantic
	depth = 4
	dummy = covclass.load(depth_idx = depth)
	floatnum = round(len(dummy.trans_geo.total_list)*1/9.)
	for k in range(20):
		H,index_list = HInstance.random_floats(dummy.trans_geo,floatnum,[1,1,0.25,0.25,0.25])
		p_hat,cov_subtract = dummy.p_hat_calculate(H,index_list,noise_factor=4)
		p_hat_data = dummy.trans_geo.return_variance('ph',cov_subtract)
		fig = plt.figure(figsize=(13,7))
		ax1 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
		XX,YY,ax = dummy.trans_geo.plot_setup(ax = ax1)
		ax.pcolor(XX,YY,dummy.trans_geo.transition_vector_to_plottable(p_hat_data/dummy.trans_geo.return_variance('ph',dummy.cov))*100,
			vmin=0,vmax=95,cmap = colormap_r)
		PCM=ax.get_children()[3]
		plt.colorbar(PCM, ax=ax, label='% pH Variance Constrained') 
		float_idx = np.array_split(H.sum(axis=0).ravel().tolist()[0],len(dummy.trans_geo.variable_list))
		ts_idx = np.where(float_idx[0]>0)
		ph_idx = np.where(float_idx[dummy.trans_geo.variable_list.index('ph')]>0)
		for idxs,markersize,markercolor,label in [(ts_idx,10,'r','Core'),(ph_idx,20,'b','BGC')]:
			float_list = GeoList([dummy.trans_geo.total_list[x] for x in idxs[0].tolist()])
			lats,lons = float_list.lats_lons()
			ax.scatter(lons,lats,s=markersize,c=markercolor,label=label)
		plt.legend()
		plt.savefig(dummy.trans_geo.file_handler.store_file("../../slide_movie/" + str(k)))
		plt.close()
	os.chdir(dummy.trans_geo.file_handler.store_file("../../slide_movie/"))
	os.system("ffmpeg -r 3/2 -i %01d.png -vcodec mpeg4 -y movie.mp4")


def remove_previous_calcs():
	dummy = CovCM4TropicalAtlantic()
	dir = dummy.trans_geo.file_handler.store_file("../../")
	[os.remove(x) for x in find_files(dir, '*.npy') if "/store/" in x]

def plot_cov(dummy,p_hat,cov_subtract,index_list):
	fig = plt.figure(figsize=(14,14))
	ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
	XX,YY,ax = dummy.trans_geo.plot_setup(ax=ax1)
	plottable = dummy.trans_geo.transition_vector_to_plottable(cov_subtract.diagonal()/dummy.cov.diagonal())
	ax.pcolormesh(XX,YY,plottable,vmax = 1,vmin=0)
	float_list = GeoList([dummy.trans_geo.total_list[x] for x in index_list])
	lats,lons = float_list.lats_lons()
	ax.scatter(lons,lats)

	ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
	XX,YY,ax = dummy.trans_geo.plot_setup(ax=ax2)
	plottable = dummy.trans_geo.transition_vector_to_plottable(p_hat.diagonal()/dummy.cov.diagonal())
	ax.pcolormesh(XX,YY,plottable,vmax = 1,vmin=0)
	float_list = GeoList([dummy.trans_geo.total_list[x] for x in index_list])
	lats,lons = float_list.lats_lons()
	ax.scatter(lons,lats)
	plt.show()

for covclass in [CovCM4Indian,CovCM4SO,CovCM4NAtlantic,CovCM4TropicalAtlantic,CovCM4SAtlantic,CovCM4NPacific,CovCM4TropicalPacific,CovCM4SPacific,CovCM4GOM,CovCM4CCS]:
	for depth in [2,4,6,8,10,12,14,16,18,20,22,24,26]:
		dummy = covclass.load(depth_idx = depth)
		for x in np.arange(0.001, float(1/5), 0.04):
			floatnum = round(x*len(dummy.trans_geo.total_list))
			if floatnum == 0:
				floatnum =1 
			print(floatnum)
			for k in range(80):
				try:
					H,index_list = HInstance.random_floats(dummy.trans_geo,floatnum,[1,1,0.25,0.25,0.25])
				except AssertionError:
					H,index_list = HInstance.random_floats(dummy.trans_geo,floatnum,[1,1,0.25,0.25])
				p_hat,cov_subtract = dummy.p_hat_calculate(H,index_list,noise_factor=8)
				# plot_cov(dummy,p_hat,cov_subtract,index_list)
				filename = dummy.trans_geo.file_handler.store_file(str(uuid.uuid4()))
				key_list = ['depth','floatnum','total_variance']

				if ((1-p_hat.diagonal().sum()/dummy.cov.diagonal().sum())<0)|((1-p_hat.diagonal().sum()/dummy.cov.diagonal().sum())>1):
					print('something is wrong with the variance')
					continue
				data_list = [depth,x,(1-p_hat.sum()/dummy.cov.sum())]
				for variable in dummy.trans_geo.variable_list:
					p_hat_data = dummy.trans_geo.return_variance(variable,p_hat)
					cov_data = dummy.trans_geo.return_variance(variable,dummy.cov)
					key_list.append(variable)
					data_list.append((1-p_hat_data.sum()/cov_data.sum()))
				np.save(filename,dict(zip(key_list, data_list)))