import numpy as np 
import scipy.signal
import scipy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
variable_list = ['dic_100m','dic_surf','o2_100m','o2_surf','pco2_surf']
from mpl_toolkits.basemap import Basemap

class Basemap(Basemap):

	@classmethod
	def auto_map(cls,urlat,lllat,urlon,lllon,lon_0,aspect=False,spacing=10,bar=False,depth=False,resolution='l'):

		m = cls(projection='cea',llcrnrlat=lllat,urcrnrlat=urlat,\
				llcrnrlon=lllon,urcrnrlon=urlon,resolution=resolution,lon_0=lon_0,\
				fix_aspect=aspect)
			# m.drawmapboundary(fill_color='darkgray')
		m.fillcontinents(color='dimgray',zorder=10)
		m.drawmeridians(np.arange(0,360,2*spacing),labels=[True,False,False,False])
		m.drawparallels(np.arange(-90,90,spacing),labels=[False,False,False,True])
		if depth:
			depth = Depth()
			depth.z[depth.z>0] = 0 
			depth.z[depth.z<-6000] = -6000 
			XX,YY = np.meshgrid(depth.x[::5],depth.y[::5])
			XX,YY = m(XX,YY)
			m.contourf(XX,YY,depth.z[::5,::5]/1000.,vmin=-6,vmax=0)
		if bar:
			m.colorbar(label='Depth (km)')

		return m

def basemap_setup(lat_grid,lon_grid,fill_color=True):
	X,Y = np.meshgrid(lon_grid,lat_grid)
	lon_0 = 0
	llcrnrlon=-180.
	llcrnrlat=-80.
	urcrnrlon=180.
	urcrnrlat=80
	m = Basemap.auto_map(urcrnrlat,llcrnrlat,urcrnrlon,llcrnrlon,lon_0,aspect=False)
	XX,YY = m(X,Y)
	return (XX,YY,m)


def lats_lons():
	lat_grid = np.arange(-90,91,2)
	lats = np.load('lat_list.npy')
	lat_truth_list = np.array([x in lat_grid for x in lats])

	lon_grid = np.arange(-180,180,2)
	lons = np.load('lon_list.npy')
	lons[lons<-180]=lons[lons<-180]+360
	lon_truth_list = np.array([x in lon_grid for x in lons])

	truth_list = lat_truth_list&lon_truth_list
	lats = lats[truth_list]
	lons = lons[truth_list]
	return (lats,lons,truth_list,lat_grid,lon_grid)


def transition_vector_to_plottable(lat_grid,lon_grid,index_list,vector):
	plottable = np.zeros([len(lon_grid),len(lat_grid)])
	for n,tup in enumerate(index_list):
		ii_index = lon_grid.index(tup[1])
		qq_index = lat_grid.index(tup[0])
		plottable[ii_index,qq_index] = vector[n]
	# plottable = np.ma.masked_equal(plottable,0)
	return plottable.T

def generate_H(column_idx,N):
	data = []
	col_idx = []
	for col in np.unique(column_idx):
		data.append(np.sum(np.array(column_idx)==col))
		col_idx.append(col)
	row_idx = range(len(col_idx))
	return scipy.sparse.csc_matrix((data,(row_idx,col_idx)),shape=(len(col_idx),N))

def p_hat_calculate(output_list,cov,noise_factor):
	noise = scipy.sparse.csc_matrix.diagonal(cov)[np.unique(output_list)]
	H = generate_H(np.sort(output_list),cov.shape[0])
	denom = H.dot(cov).dot(H.T)+scipy.sparse.diags(noise*noise_factor)
	inv_denom = scipy.sparse.linalg.inv(denom)
	if not type(inv_denom)==scipy.sparse.csc.csc_matrix:
		inv_denom = scipy.sparse.csc.csc_matrix(inv_denom)  # this is for the case of 1x1 which returns as array for some reason
	cov_subtract = cov.dot(H.T).dot(inv_denom).dot(H).dot(cov)
	diag = scipy.sparse.csc_matrix.diagonal(cov)
	for idx in np.where(diag<0)[0]:
		cov[idx,idx]=0
	p_hat = cov-cov_subtract
	return p_hat

def get_index_of_first_eigen_vector(p_hat):
	# eigs = scipy.sparse.linalg.eigs(p_hat)
	# e_vec = eigs[1][:,-1]
	e_vec = np.array(p_hat.sum(axis =0))
	print e_vec.max()
	print np.where(e_vec == e_vec.max())
	idx = np.where(e_vec == e_vec.max())[1][0]

	return idx,e_vec




def plot_cov_and_floats(lat_grid,lon_grid,p_hat,p_hat_cov,lats,lons,output_list,unconstrained_list,cov):
	# plt.subplot(2,1,1)
	# x_axis = range(len(unconstrained_list))
	# plt.plot(x_axis,unconstrained_list)
	# plt.xlabel('Float Number')
	# plt.ylabel('Unconstrained Variance')

	# plt.subplot(2,1,2)
	plt.subplot(2,1,1)
	dia = scipy.sparse.csc_matrix.diagonal(p_hat).tolist()
	plottable = transition_vector_to_plottable(lat_grid.tolist(),lon_grid.tolist(),zip(lats,lons),dia)
	XX,YY,m = basemap_setup(lat_grid,lon_grid)
	m.pcolormesh(XX,YY,plottable)
	plt.colorbar(label='Correlation')
	y,x = zip(*np.array(zip(lats,lons))[np.array(output_list)])
	x,y = m(x,y)
	m.scatter(x,y,marker='*',color='g',s=45,zorder=30)
	plt.subplot(2,1,2)
	dia = scipy.sparse.csc_matrix.diagonal(p_hat_cov).tolist()
	plottable = transition_vector_to_plottable(lat_grid.tolist(),lon_grid.tolist(),zip(lats,lons),dia)
	XX,YY,m = basemap_setup(lat_grid,lon_grid)
	m.pcolormesh(XX,YY,plottable)
	plt.colorbar(label='Variance $(mol\ m^{-2})^2$')
	y,x = zip(*np.array(zip(lats,lons))[np.array(output_list)])
	x,y = m(x,y)
	m.scatter(x,y,marker='*',color='g',s=45,zorder=30)
	plt.savefig(str(float_))
	plt.close()

# lats,lons,truth_list,lat_grid,lon_grid = lats_lons()
# for variable in variable_list:
# 	for type_ in ['cor']:
# 		space_space = np.load('space_space_'+type_+variable+'.npy')
# 		space_space = XY_filter*space_space
# 		noise = space_space.mean()/100
# 		output_list = np.load(variable+'_idxs_'+type_+'.npy')
# 		for idx in range(len(output_list)):
# 			float_list = output_list[:idx]
# 			p_hat = p_hat_calculate(float_list,space_space,noise)
# 			plottable = transition_vector_to_plottable(lat_grid.tolist(),lon_grid.tolist(),zip(lats,lons),np.diag(p_hat))
# 			XX,YY = np.meshgrid(lon_grid,lat_grid)
# 			plt.pcolormesh(XX,YY,plottable)
# 			try:
# 				y,x = zip(*np.array(zip(lats,lons))[float_list])
# 				plt.scatter(x,y,marker='*',color='g',s=45,zorder=30)
# 			except ValueError:
# 				pass
# 			plt.savefig(type_+variable+str(idx)+'plot')
# 			plt.close()


def decimate():
	for variable in variable_list:
		print variable
		lats,lons,truth_list,lat_grid,lon_grid = lats_lons()
		file_ = 'subsampled_'+variable+'.npy'
		dummy_array = np.load(file_)
		dummy_array = dummy_array[1095:,truth_list]
		dummy_array = scipy.signal.decimate(dummy_array,20,axis=0)
		np.save('decimated'+variable,dummy_array)


def calculate_scaling():
	lats,lons,truth_list,lat_grid,lon_grid = lats_lons()
	X,Y = np.meshgrid(lons,lats) 
	X_filter = abs(X-X[0,:].reshape(X.shape[0],1))
	Y_filter = abs(Y-Y[:,0])

	degree_to_km = 111.32
	X_dist = np.cos(np.deg2rad(Y))*degree_to_km*X_filter
	Y_dist = degree_to_km*Y_filter

	del X_filter
	del Y_filter
	del X
	del Y

	dist = np.sqrt(X_dist**2+Y_dist**2)
	del X_dist
	del Y_dist 
	scaling = (np.exp(-dist**2/(2*600**2)))
	scaling[scaling<0.2]=0
	return scaling

def calculate_cov():
	scaling = calculate_scaling()
	for variable in variable_list:
		print variable
		dummy_array = np.load('decimated'+variable+'.npy')
		time_time_cov = np.cov(dummy_array)
		eigs = scipy.sparse.linalg.eigs(time_time_cov)
		for k in range(6):
			time_time_cov -= eigs[0][k]*np.outer(eigs[1][:,k],eigs[1][:,k])
		np.save('time_time_cov_'+variable,time_time_cov)
		del time_time_cov

		space_space_cov = np.cov(dummy_array.T)
		eigs = scipy.sparse.linalg.eigs(space_space_cov)
		for k in range(6):
			space_space_cov -= eigs[0][k]*np.outer(eigs[1][:,k],eigs[1][:,k])
		space_space_cov = scaling*space_space_cov
		space_space_cov[space_space_cov<0] =0
		np.save('space_space_cov'+variable,space_space_cov)
		del space_space_cov

def calculate_cor():
	scaling = calculate_scaling()
	for variable in variable_list:
		print variable
		dummy_array = np.load('decimated'+variable+'.npy')
		time_time_cor = np.corrcoef(dummy_array)
		eigs = scipy.sparse.linalg.eigs(time_time_cor)
		for k in range(6):
			time_time_cor -= eigs[0][k]*np.outer(eigs[1][:,k],eigs[1][:,k])
		np.save('time_time_cor_'+variable,time_time_cor)
		del time_time_cor

		space_space_cor = np.corrcoef(dummy_array.T)
		eigs = scipy.sparse.linalg.eigs(space_space_cor)
		for k in range(6):
			space_space_cor -= eigs[0][k]*np.outer(eigs[1][:,k],eigs[1][:,k])
		space_space_cor = scaling*space_space_cor
		space_space_cor[space_space_cor<0] =0

		space_space_cor = space_space_cor/space_space_cor.max(axis = 0)
		np.fill_diagonal(space_space_cor,1)
		np.save('space_space_cor'+variable,space_space_cor)
		del space_space_cor

# def optimal_array_calculate():
# 	for variable in variable_list:

def snr_calculate(output_list):

	SNR_list = [10000,5000,1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01]
	mean_targeted_list = []
	std_targeted_list = []
	mean_random_list = []
	std_random_list = []
	space_space_cov = np.load('space_space_cov'+variable+'.npy')
	space_space_cov = scipy.sparse.csc_matrix(space_space_cov)

	for SNR in SNR_list:
		print SNR
		dummy_random_list = []
		dummy_targeted_list = []
		for _ in range(10):
			print _
			space_space_cor = np.load('space_space_cor'+variable+'.npy')
			space_space_cor = scipy.sparse.csc_matrix(space_space_cor)
			scaling = calculate_scaling()
			scaling = scipy.sparse.csc_matrix(scaling)
			scaling.data = scaling.data*np.random.normal(scale=1/np.sqrt(SNR)*np.std(space_space_cor.data),size=len(scaling.data))
			space_space_cor+=scaling
			space_space_cor = space_space_cor/space_space_cor.max(axis=0).todense()
			space_space_cor = scipy.sparse.csc_matrix(space_space_cor)
			space_space_cor.data[space_space_cor.data<=0]=0.01
			random_list = get_random_locations(space_space_cor,1000)
			p_hat = p_hat_calculate(random_list,space_space_cor,0.1)
			p_hat.data[p_hat.data<=0]=0.01

			dummy_random_list.append(get_p_hat_error(p_hat,space_space_cov))
			print dummy_random_list[-1]
			p_hat = p_hat_calculate(output_list,space_space_cor,0.1)
			p_hat.data[p_hat.data<=0]=0.01
			dummy_targeted_list.append(get_p_hat_error(p_hat,space_space_cov))
			print dummy_targeted_list[-1]

		mean_targeted_list.append(np.mean(dummy_targeted_list))
		std_targeted_list.append(np.std(dummy_targeted_list))
		mean_random_list.append(np.mean(dummy_random_list))
		std_random_list.append(np.std(dummy_random_list))

	target_y_1 = np.array(mean_targeted_list[:10])+np.array(std_targeted_list[:10])
	target_y_2 = np.array(mean_targeted_list[:10])-np.array(std_targeted_list[:10])
	random_y_1 = np.array(mean_random_list[:10])+np.array(std_random_list[:10])
	random_y_2 = np.array(mean_random_list[:10])-np.array(std_random_list[:10])

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.fill_between(SNR_list[:10], random_y_2,random_y_1,color='g',alpha=0.2)
	ax.plot(SNR_list[:10],mean_random_list[:10],color='g',label='Random')
	ax.fill_between(SNR_list[:10], target_y_2,target_y_1,color='k',alpha=0.2)
	ax.plot(SNR_list[:10],mean_targeted_list[:10],color='k',label='Targeted')
	plt.ylim([random_y_2[-1],random_y_1[-1]])
	plt.xlim([min(SNR_list[:10]),max(SNR_list[:10])])

	ax.set_xscale('log')
	ax.set_yscale('log')
	plt.ylabel('Unobserved Variance $(mol\ m^{-2})^2$')
	plt.xlabel('SNR')
	plt.legend()
	plt.savefig('slide_4')
	plt.close()



def get_random_locations(cov,n):
	return random.sample(range(cov.shape[0]),n)

def get_p_hat_error(p_hat,cov):
	return (scipy.sparse.csc_matrix.diagonal(p_hat)*scipy.sparse.csc_matrix.diagonal(cov)).sum()


def slide_3(space_space,space_space_cov,error_list):
	random_error_list = []
	random_std_list = []
	num_list = np.arange(900,1160,10)
	for num in num_list:
		print num
		dummy_list = []
		for k in range(10):
			print k
			random_list = get_random_locations(space_space_cov,num)
			p_hat = p_hat_calculate(random_list,space_space,0.1)
			dummy_list.append(get_p_hat_error(p_hat,space_space_cov))
			print dummy_list
		random_error_list.append(np.mean(dummy_list))
		random_std_list.append(np.std(dummy_list))
	random_y1 = np.array(random_error_list)-np.array(random_std_list)
	random_y2 = np.array(random_error_list)+np.array(random_std_list)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.fill_between(num_list, random_y1,random_y2,color='g',alpha=0.2)
	ax.plot(num_list,random_error_list,color='g',label='Random')
	ax.plot(range(len(error_list)),error_list,color='k',label='Targeted')
	ax.plot(num_list,[error_list[-1]]*len(num_list),linestyle='--',color='k')
	plt.ylabel('Unobserved Variance $(mol\ m^{-2})^2$')
	plt.xlabel('Float Deployed')
	plt.xlim([900,1150])
	plt.ylim([random_y1[-1],random_y2[0]])
	plt.legend()
	plt.savefig('random_variance_constrained')
	plt.close()

def slide_2(cor,output_list):
	def map(p_hat,output_list_):
		dia = scipy.sparse.csc_matrix.diagonal(p_hat).tolist()
		plottable = transition_vector_to_plottable(lat_grid.tolist(),lon_grid.tolist(),zip(lats,lons),dia)
		XX,YY,m = basemap_setup(lat_grid,lon_grid)
		m.pcolormesh(XX,YY,plottable)
		plt.colorbar(label='Correlation')
		y,x = zip(*np.array(zip(lats,lons))[np.array(output_list_)])
		x,y = m(x,y)
		m.scatter(x,y,marker='*',color='g',s=45,zorder=30)

	plt.subplot(3,1,1)
	p_hat = p_hat_calculate(output_list[:100],cor,0.15)
	map(p_hat,output_list[:100])
	plt.subplot(3,1,2)
	p_hat = p_hat_calculate(output_list[:500],cor,0.15)
	map(p_hat,output_list[:500])
	plt.subplot(3,1,3)
	p_hat = p_hat_calculate(output_list[:1000],cor,0.15)
	map(p_hat,output_list[:1000])
	plt.savefig('slide_2')
	plt.close()


def run():
	lats,lons,truth_list,lat_grid,lon_grid = lats_lons()
	XX,YY = np.meshgrid(lon_grid,lat_grid)
	# for variable in variable_list:
	print variable
	space_space = np.load('space_space_cor'+variable+'.npy')
	type_ = 'cor'
	space_space_cov = np.load('space_space_cov'+variable+'.npy')
	space_space = scipy.sparse.csc_matrix(space_space)
	space_space_cov = scipy.sparse.csc_matrix(space_space_cov)

	output_list = []
	noise_factor = 0.1
	# self.plot_p_hat(cov,output_list,'')
	idx,e_vec = get_index_of_first_eigen_vector(space_space)
	output_list.append(idx)
	float_ = 0
	error_list = []
	while np.unique(output_list).shape[0]<1002:
		print float_
		# if float_ % 2==0:
		p_hat = p_hat_calculate(output_list,space_space,noise_factor)
		p_hat_cov = p_hat_calculate(output_list,space_space_cov,noise_factor)
		unconstrained = get_p_hat_error(p_hat,space_space_cov)
		error_list.append(unconstrained)

		# else:
		# 	p_hat = p_hat_calculate(output_list,space_space_cov,noise_factor)
		if float_%10==0: 
			plot_cov_and_floats(lat_grid,lon_grid,p_hat,p_hat_cov,lats,lons,output_list,error_list,space_space)
		idx,e_vec = get_index_of_first_eigen_vector(p_hat)
		output_list.append(idx)
		float_ += 1
		np.save(variable+'_idxs_'+type_,np.array(output_list))


	lats,lons,truth_list = lats_lons()
	for variable in variable_list:
		for type_ in ['cor','cov']:
			output_list = np.load(variable+'_idxs_'+type_+'.npy')
			x,y = zip(*np.array(zip(lons,lats))[output_list])
			plt.scatter(x,y)
			plt.savefig(variable+'_idxs_'+type_+'.png')
			plt.close()

def calculate_eigs():
	for variable in variable_list:
		for type_ in ['time_time','space_space']:
			dummy_array = np.load('decimated'+variable+'.npy')
			try:
				cov = np.load(type_+'_cov_'+variable+'.npy')
			except IOError:
				cov = np.load(type_+'_cov'+variable+'.npy')
			eigs = scipy.sparse.linalg.eigs(cov)
			np.save(type_+'_eigs_'+variable,eigs)
			output_list = []
			for k in range(6):
				try:
					output = dummy_array.dot(eigs[1][:,k])
				except ValueError:
					output = dummy_array.T.dot(eigs[1][:,k])
				output_list.append(output)
			np.save(type_+'_eigs_output_'+variable,np.array(output_list))




def plot_eigs():


	def space_plot(plottable):
		plottable = transition_vector_to_plottable(lat_grid.tolist(),lon_grid.tolist(),index_list,plottable)
		XX,YY = np.meshgrid(lon_grid,lat_grid)
		plt.pcolormesh(XX,YY,plottable)

	def time_plot(plottable):
		x = np.arange(0,20*len(plottable),20)
		plt.plot(x,plottable)
		plt.xlabel('Time (days)')



	index_list = zip(lats,lons)


	for variable in variable_list:
		for type_ in ['time_time','space_space']:
			eigs = np.load(type_+'_eigs_'+variable+'.npy')
			output = np.load(type_+'_eigs_output_'+variable+'.npy')
			for k in range(6):
				dummy_output = output[k,:]
				dummy_eig = eigs[1][:,k]
				plt.figure()
				plt.title(type_+' e val = '+str(eigs[0][k]))
				plt.subplot(2,1,1)
				if len(dummy_output) == len(lons):
					space_plot(dummy_output)
					plt.subplot(2,1,2)
					time_plot(dummy_eig)
				else:
					space_plot(dummy_eig)
					plt.subplot(2,1,2)
					time_plot(dummy_output)
				plt.savefig(variable+type_+str(k))
				plt.close()
