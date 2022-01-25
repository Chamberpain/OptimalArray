
from __future__ import print_function

import os
import pandas as pd
import numpy as np
import oceans
import datetime

def save_lat_lon_cm2p6():
	nc_fid_token = Dataset('/data/SO12/CM2p6/ocean_scalar.static.nc')
	mask = nc_fid_token.variables['ht'][:]>2000
	lon_token = nc_fid_token.variables['xt_ocean'][:]
	lat_token = nc_fid_token.variables['yt_ocean'][:]
	X,Y = np.meshgrid(lon_token,lat_token)
	lon_list = X[mask]
	lat_list = Y[mask]
	rounded_lat_list = np.arange(lat_token.min(),lat_token.max()+.5,0.5)
	rounded_lon_list = np.arange(lon_token.min(),lat_token.max()+.5,0.5)
	target_lat_list = [find_nearest(lat_token,x) for x in rounded_lat_list]
	target_lon_list = [find_nearest(lon_token,x) for x in rounded_lon_list]
	subsample_mask = np.isin(lon_list,target_lon_list)&np.isin(lat_list,target_lat_list)
	lat_list = lat_list[subsample_mask]
	lon_list = lon_list[subsample_mask]
	lat_list = np.array([find_nearest(rounded_lat_list,x) for x in lat_list])
	lon_list = np.array([find_nearest(rounded_lon_list,x) for x in lon_list])
	np.save('lat_list.npy',lat_list)
	np.save('lon_list.npy',lon_list)
	np.save('mask.npy',mask.data)
	np.save('subsample_mask.npy',subsample_mask)

def recompile_subsampled():
	def variable_extract(nc_,variable_list):
		matrix_holder = nc_[mask]
		variable_list.append(matrix_holder.flatten()[subsample_mask].data)		
		return variable_list

	mask = np.load('mask.npy')
	subsample_mask = np.load('subsample_mask.npy')
	surf_dir = '/data/SO12/CM2p6/ocean_minibling_surf_flux/'
	hun_m_dir = '/data/SO12/CM2p6/ocean_minibling_100m/'
	data_directory = '/data/SO12/pchamber/cm2p6/'
	matches = []

	for root, dirnames, filenames in os.walk(data_directory):
		for filename in fnmatch.filter(filenames, '*ocean.nc'):
			matches.append(os.path.join(root, filename))	

	for n, match in enumerate(matches):
		o2_list = []
		dic_list = []
		temp_list_100m = []
		salt_list_100m = []
		temp_list_surf = []
		salt_list_surf = []
		print('file is ',match,', there are ',len(matches[:])-n,'files left')
		file_date = match.split('/')[-1].split('.')[0]
		try:
			hun_m_fid = Dataset(hun_m_dir+file_date+'.ocean_minibling_100m.nc')
		except FileNotFoundError:
			try:
				hun_m_fid = Dataset(data_directory+file_date+'.ocean_minibling_100.nc')
			except OSError:
				print('There was a problem with '+data_directory+file_date+'.ocean_minibling_100.nc')
				print('continuing')
				continue
		hun_m_time = hun_m_fid.variables['time'][:]
		nc_fid = Dataset(match, 'r')
		nc_fid_time = nc_fid.variables['time'][:]
		time_idx = [hun_m_time.tolist().index(find_nearest(hun_m_time,t)) for t in nc_fid_time]
		for k in time_idx:
			print('day ',k)
			o2_list = variable_extract(hun_m_fid.variables['o2'][k,:,:],o2_list)
			dic_list = variable_extract(hun_m_fid.variables['dic'][k,:,:],dic_list)
		for k in range(len(nc_fid.variables['time'][:])):
#9 corresponds to 100m
			salt_list_100m = variable_extract(nc_fid.variables['salt'][k,9,:,:],salt_list_100m)
			temp_list_100m = variable_extract(nc_fid.variables['temp'][k,9,:,:],temp_list_100m)
			salt_list_surf = variable_extract(nc_fid.variables['salt'][k,0,:,:],salt_list_surf)
			temp_list_surf = variable_extract(nc_fid.variables['temp'][k,0,:,:],temp_list_surf)
		np.save('/data/SO12/pchamber/'+str(file_date)+'_time',nc_fid_time.data)
		o2 = np.vstack(o2_list)
		np.save('/data/SO12/pchamber/'+str(file_date)+'_100m_subsampled_o2',o2)
		dic = np.vstack(dic_list)
		np.save('/data/SO12/pchamber/'+str(file_date)+'_100m_subsampled_dic',dic)
		salt_100m = np.vstack(salt_list_100m)
		np.save('/data/SO12/pchamber/'+str(file_date)+'_100m_subsampled_salt',salt_100m)
		temp_100m = np.vstack(temp_list_100m)
		np.save('/data/SO12/pchamber/'+str(file_date)+'_100m_subsampled_temp',temp_100m)
		salt_surf = np.vstack(salt_list_surf)
		np.save('/data/SO12/pchamber/'+str(file_date)+'_surf_subsampled_salt',salt_surf)
		temp_surf = np.vstack(temp_list_surf)
		np.save('/data/SO12/pchamber/'+str(file_date)+'_surf_subsampled_temp',temp_surf)



def particle_save():
"""Script that gathers together all of the SOSE particle trajectory data and formats it into one dataframe

todo: Need to add routine to filter out floats in depths greater than 1000m (data isa provided did not do this)
"""
	data_file_name = os.getenv("HOME")+'/Data/Raw/SOSE/particle_release/SO_RAND_0001.XYZ.0000000001.0003153601.data'
	df_output_file_name = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/sose_particle_df.pickle'
	npts= 10000
	ref_date = datetime.date(year=1950,month=1,day=1)
	opt=np.fromfile(data_file_name,'>f4').reshape(-1,3,npts)

	print("data has %i records" %(opt.shape[0]))

	frames = []
	for k in range(opt.shape[2]):
		x,y=opt[:,0,k],opt[:,1,k]#this is in model grid index coordinate, convert to lat-lon using x=x/6.0;y=y/6.0-77.875
		t = range(opt.shape[0])
		date=[ref_date+datetime.timedelta(days=j) for j in t]
		x=x/6.0;y=y/6.0-77.875
		x = x%360
		x[x>180]=x[x>180]-360
		frames.append(pd.DataFrame({'Lat':y,'Lon':x,'Date':date}))
		frames[-1]['Cruise']=k
	df = pd.concat(frames)
	df['position type'] = 'SOSE'
	df['Lon']=oceans.wrap_lon180(df.Lon) 

	assert df.Lon.min() >= -180
	assert df.Lon.max() <= 180
	assert df.Lat.max() <= 90
	assert df.Lat.min() >=-90
	df.to_pickle(df_output_file_name)