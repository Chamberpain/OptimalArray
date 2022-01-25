import numpy as np
from netCDF4 import Dataset
import fnmatch
import os
import time
import matplotlib.pyplot as plt
from GeneralUtilities.Compute.list import GeoList, VariableList
import scipy.sparse.linalg
from scipy.sparse import csc_matrix
from GeneralUtilities.Filepath.instance import does_file_exist
from itertools import combinations
from GeneralUtilities.Filepath.instance import FilePathHandler
import geopy
from TransitionMatrix.Utilities.Inversion.target_load import InverseGeo,InverseSOSE,InverseGOM,InverseCCS

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

class MatrixElement(object):
	def __init__(self,_array,lats,lons,lat_grid,lon_grid,index_list,file_name,label='cm2p6'):
		self.label = label
		self.array = _array
		self.lats = lats
		self.lons = lons
		self.lat_grid = lat_grid
		self.lon_grid = lon_grid
		self.index_list = index_list
		self.file_name = file_name

	def return_eig_vals(self):
		eigs = np.linalg.eigvals(self.array)
		return eigs

	def eig_val_plot(self):
		base = ROOT_DIR+'/plots/'+self.label+'_covariance/'
		eigs = self.return_eig_vals()
		fig = plt.figure()
		ax = fig.add_subplot(2, 1, 1)
		ax.plot(np.sort(eigs))
		plt.xlim([0,len(self.array)])
		plt.ylabel('Eig Value')
		plt.title('Positive Eigenvalues')
		ax.set_yscale('log')

		ax = fig.add_subplot(2, 1, 2)
		ax.plot(-1*np.sort(eigs))
		plt.xlim([0,len(self.array)])
		plt.xlabel('Number')
		plt.ylabel('Neg Eig Value')
		plt.title('Negative Eigenvalues')
		ax.set_yscale('log')
		plt.gca().invert_yaxis()
		plt.savefig(base+self.file_name+'_eig_vals')
		plt.close()

	def plot_data(self,data,m=False):
		plot_vec = transition_vector_to_plottable(self.lat_grid.tolist(),
			self.lon_grid.tolist(),self.index_list,data.tolist())
		if not m:
			XX,YY,ax,fig = cartopy_setup(self.lat_grid.tolist(),self.lon_grid.tolist(),'')
		plt.contourf(XX,YY,plot_vec)
		return ax,fig

class CovElement(MatrixElement):
	def __init__(self,_array,lats,lons,lat_grid,lon_grid,row_var,col_var,file_name):
		super().__init__(_array,lats,lons,lat_grid,lon_grid,file_name)
		self.row_var = row_var
		self.col_var = col_var
		eigs, eig_vecs = np.linalg.eig(self.array)
		eigs_sum_forward = np.array([eigs[:x].sum() for x in range(len(eigs))])
		eigs_idx = (eigs_sum_forward>0.99*eigs.sum()).tolist().index(True)
		matrix_holder = np.zeros(eig_vecs.shape)
		for idx in np.arange(eigs_idx):
			temp = matrix_holder+eigs[idx]*np.outer(eig_vecs[:,idx],eig_vecs[:,idx])
			matrix_holder = temp
		self._array = matrix_holder

class CovArray(object):
	def __init__(self,*args,depth_idx=0,**kwargs):

		self.file_name = 'lat_sep_'+str(self.lat_sep)+'_lon_sep_'+str(self.lon_sep)
		self.depth_idx = depth_idx
		self.truth_array,self.index_list = self.dimensions_and_masks()

	def normalize_data(self,data,label=None,percent=0.4,scale=1,plot=True):
		mean_removed = data-data.mean(axis=0)
		data_scale = mean_removed.std(axis=0)

		dummy = 0
		greater_mask = data_scale>(data_scale.max()-dummy*0.001*data_scale.mean())
		while greater_mask.sum()<percent*len(data_scale):
			dummy +=1
			greater_value = data_scale.max()-dummy*0.001*data_scale.mean()
			greater_mask = data_scale>greater_value
		print('greater value is '+str(greater_value))
		dummy = 0
		# lesser_mask = data_scale<(data_scale.min()+dummy*0.001*data_scale.mean())
		lesser_value = 0 
		lesser_mask = data_scale<lesser_value
		while lesser_mask.sum()<0: # this used to say "percent*len(data_scale):", but was changed
			dummy +=1
			lesser_value = data_scale.min()+dummy*0.001*data_scale.mean()
			lesser_mask = data_scale<lesser_value
		print('lesser value is '+str(lesser_value))
		data_scale[greater_mask]=greater_value*scale
		data_scale[lesser_mask]=lesser_value*scale
		data_scale[~greater_mask&~lesser_mask] = data_scale[~greater_mask&~lesser_mask]*scale
		data_scale[data_scale==0]=10**-12
		if plot:
			base = '/Users/pchamberlain/Projects/transition_matrix/plots/'
			histogram, bins = np.histogram(data_scale,bins=100)
			bin_centers = 0.5*(bins[1:] + bins[:-1])
			plt.figure(figsize=(6, 12))
			plt.subplot(2,1,1)
			plt.plot(bin_centers, histogram)
			plt.scatter([lesser_value*scale,greater_value*scale],[0,0],s=15)
			plt.xlabel('STD')
			plt.title('Histogram of STD of Ensemble')
			plt.subplot(2,1,2)
			plt.hist((mean_removed/data_scale).flatten(),bins=200)
			plt.xlabel('Value')
			plt.title('Histogram of Data')
			plt.savefig(base+label+'_std')
			plt.close()
		return mean_removed/data_scale
		
	def subtract_e_vecs_return_space_space(self,e_vec_num=4,plot=False):
		# space_space_submeso = np.cov(data.T)
		space_space_submeso = self.make_matrix()
		assert (space_space_submeso.diagonal()>=0).all()
		space_space_global = scipy.sparse.csc_matrix(space_space_submeso.shape)
		eig_vals, eig_vecs = scipy.sparse.linalg.eigs(space_space_submeso,k=4)
		eig_vecs = eig_vecs.real
		eig_vals = eig_vals.real
		dist_filter = self.get_dist()<5*self.l # maximum distance we will have non zero values with gaspari and cohn scaling
		row_val,col_val = np.where(np.block([[dist_filter]*len(self.variable_list)]*len(self.variable_list)))
		submeso_data = space_space_submeso[(row_val,col_val)]
		space_space_submeso = scipy.sparse.csc_matrix((submeso_data,(row_val,col_val)),shape=space_space_submeso.shape)
		del dist_filter
		del submeso_data

		for k in range(len(eig_vals)):
			print('calculating eigen vector '+str(k))
			e_val = eig_vals[k]
			e_vec = eig_vecs[:,k]



			remove_e_vec = e_val*np.outer(e_vec,e_vec)/(e_vec**2).sum()
			e_vec_data = remove_e_vec[(row_val,col_val)]
			remove_e_vec = scipy.sparse.csc_matrix((e_vec_data,(row_val,col_val)),shape=space_space_submeso.shape)
			space_space_submeso -= remove_e_vec
			if (space_space_submeso.diagonal()<0).any():
				space_space_submeso -= space_space_submeso.diagonal().min()*scipy.sparse.eye(space_space_submeso.shape[0])
			space_space_global += remove_e_vec
			if (space_space_global.diagonal()<0).any():
				space_space_global -= space_space_global.diagonal().min()*scipy.sparse.eye(space_space_global.shape[0])


			if plot:
				dic_temp_submeso = np.split(np.split(space_space_submeso,4)[1],4,axis=1)[1]
				dic_temp_global = np.split(np.split(space_space_global,4)[1],4,axis=1)[1]
				dic_remove_e_vec = np.split(np.split(remove_e_vec,4)[1],4,axis=1)[1]
				print(str(e_val)+' = eval')
				print(str((e_vec**2).sum())+' = size of evec')

				try:
					ss_dummy.plot_data(dic_temp_submeso.diagonal())
				except NameError:
					ss_dummy = MatrixElement(dic_temp_submeso.diagonal(),self.lats,self.lons,self.lat_grid,\
						self.lon_grid,self.index_list,'')
					ss_dummy.plot_data(dic_temp_submeso.diagonal())
				plt.colorbar()
				plt.savefig('submeso_e_vec_'+str(k)+'_removed')
				plt.close()
				ss_dummy.plot_data(dic_remove_e_vec.diagonal())
				plt.colorbar()
				plt.savefig('e_vec_'+str(k)+'_to_remove')
				plt.close()
				ss_dummy.plot_data(dic_temp_global.diagonal())
				plt.colorbar()
				plt.savefig('global_e_vec_'+str(k)+'_removed')
				plt.close()
		return (space_space_submeso,space_space_global)


	def scale_cov(self):
		def make_scaling(holder):
			cov_scale = 0.7 
			total_list = []
			for k in range(len(self.variable_list)):
				temp_list = [cov_scale*holder]*len(self.variable_list)
				temp_list[k] = holder
				total_list.append(temp_list)
			return scipy.sparse.csc_matrix(np.block(total_list))
		submeso_cov,global_cov = self.subtract_e_vecs_return_space_space()
		assert (submeso_cov.diagonal()>=0).all()
		assert (global_cov.diagonal()>=0).all()
		holder = self.calculate_scaling(lat_sep=self.lat_sep,lon_sep=self.lon_sep,l=self.l)
		submeso_scaling = make_scaling(holder)
		self.submeso_cov = submeso_cov.multiply(submeso_scaling)
		assert (self.submeso_cov.diagonal()>=0).all()

		del submeso_scaling
		holder = self.calculate_scaling(lat_sep=self.lat_sep,lon_sep=self.lon_sep,l=self.l*5)
		global_scaling = make_scaling(holder)
		del holder
		self.global_cov = global_cov.multiply(global_scaling)
		assert (self.global_cov.diagonal()>=0).all()

	# self.scaling = MatrixElement(scaling,self.lats,self.lons,self.lat_grid,self.lon_grid,self.file_name+'_scaling_l_'+str(l))


	def save(self):
		from TransitionMatrix.Utilities.Inversion.target_load import InverseInstance		
		

		trans_geo = self.trans_geo(lat_sep=self.lat_sep,lon_sep=self.lon_sep,l = 5*self.l,depth_idx=self.depth_idx)
		trans_geo.set_total_list(self.index_list)
		filename = trans_geo.make_filename()
		mat_obj = InverseInstance(self.global_cov,shape = self.global_cov.shape,trans_geo=trans_geo)
		mat_obj.save(filename=filename)

		trans_geo = self.trans_geo(lat_sep=self.lat_sep,lon_sep=self.lon_sep,l = self.l,depth_idx=self.depth_idx)
		trans_geo.set_total_list(self.index_list)
		filename = trans_geo.make_filename()
		mat_obj = InverseInstance(self.submeso_cov,shape = self.submeso_cov.shape,trans_geo=trans_geo)
		mat_obj.save(filename=filename)


	def diagnostic_data_plots(self,first_e_vals=10):
		base = ROOT_DIR+'/plots/'+self.label+'_covariance/'

		array_variable_list =self.stack_data()
		arrays,variable = zip(*array_variable_list)
		holder = np.hstack(arrays)


		time_time = np.cov(holder)


		self.lons[self.lons>=180]=self.lons[self.lons>=180]-360 
		lons,lats = zip(*self.index_list) 
		lons = np.array(lons)
		lons[lons>=180]=lons[lons>=180]-360
		self.index_list = list(zip(lons,lats))

		tt = MatrixElement(time_time,self.lats,self.lons,self.lat_grid,self.lon_grid,self.index_list,\
			self.file_name+'_time_time_cov',label=self.label)
		tt.eig_val_plot()
		space_space = self.submeso_cov+self.global_cov 
		tt_e_vals,tt_e_vecs = np.linalg.eig(time_time)
		percent_constrained = [str(round(_,2))+'%' for _ in tt_e_vals/tt_e_vals.sum()*100]

		data = np.split(space_space.todense(),len(self.variable_list),axis=1)
		ss_dummy = MatrixElement(data[0],self.lats,self.lons,self.lat_grid,self.lon_grid,self.index_list,\
			'')

		for k in range(first_e_vals):
			e_val = tt_e_vals[k]
			e_holder = tt_e_vecs[:,k]
			percent = percent_constrained[k]
			e_vecs = np.split(e_holder.reshape(1,e_holder.shape[0]).dot(holder),len(self.variable_list),axis=1)
			for var,e_vec in zip(self.variable_list,e_vecs):
				ax,fig = ss_dummy.plot_data(e_vec.flatten())
				plt.title(var+' e_vec_'+str(k)+', percent constrained = '+percent)
				plt.savefig(base+var+'_e_vec_'+str(k))
				plt.close()
				plt.plot(e_holder[:500])
				plt.xlabel('Months')
				plt.savefig(base+var+'_e_vec_time'+str(k))


	def generate_cov(self):
		self.cov_dict = {}
		for col_var in self.variable_list:
			variable_dict = {}
			for row_var in self.variable_list:
				cov_holder = self.get_cov(row_var,col_var,cov,break_unit)
				variable_dict[row_var] = CovElement(cov_holder,self.lats,self.lons,self.lat_grid,\
					self.lon_grid,row_var,col_var,self.file_name+'_'+str(row_var)+'_'+str(col_var))
			self.cov_dict[col_var] = variable_dict

	def get_cov(self,row_var,col_var):
		row_idx = self.variable_list.index(row_var)
		col_idx = self.variable_list.index(col_var)
		split_array = np.split(np.split(cov,len(self.variable_list))[row_idx],len(self.variable_list),axis=1)[col_idx]
		return split_array

	def get_dist(self):
		def calculate_distance():
			import geopy.distance
			truth_list,index_list = self.dimensions_and_masks()
			dist = np.zeros([len(index_list),len(index_list)])
			for ii,coord1 in enumerate(index_list):
				print(ii)
				for jj,coord2 in enumerate(index_list):
					dist[ii,jj] = geopy.distance.great_circle(coord1,coord2).km 
			assert (dist>=0).all()&(dist<=40000).all()
			return dist

		filename = self.file_handler.tmp_file(self.region+'_distance_lat_'+str(self.lat_sep)+'_lon_'+str(self.lon_sep))
		try:
			dist = np.load(filename+'.npy')
		except IOError:
			dist = calculate_distance()
			np.save(filename,dist)
		return dist

	def calculate_scaling(self,lat_sep=None,lon_sep=None,l=300):

		dist = self.get_dist()
		assert (dist>=0).all()
		c = np.sqrt(10/3.)*l
#For this scaling we use something derived by gassbury and coehn to not significantly change eigen spectrum of 
#local support scaling function
#last peice wise poly 		
		scaling = np.zeros(dist.shape)
		# dist[dist>2*c]=0
		second_poly_mask = (dist>c)&(dist<2*c)
		dist_holder = dist[second_poly_mask].flatten()
		assert (dist_holder.min()>c)&(dist_holder.max()<2*c)
		second_poly = 1/12.*(dist_holder/c)**5 \
		-1/2.*(dist_holder/c)**4 \
		+5/8.*(dist_holder/c)**3 \
		+5/3.*(dist_holder/c)**2 \
		-5.*(dist_holder/c) \
		+4 \
		- 2/3.*(c/dist_holder)
		second_poly[second_poly<0]=0 
		scaling[second_poly_mask]=second_poly
		# assert check_symmetric(scaling)

		first_poly_mask = (dist<c)
		dist_holder = dist[first_poly_mask].flatten()
		assert (dist_holder.min()>=0)&(dist_holder.max()<c)

		first_poly = -1/4.*(dist_holder/c)**5 \
		+1/2.*(dist_holder/c)**4 \
		+5/8.*(dist_holder/c)**3 \
		-5/3.*(dist_holder/c)**2 \
		+1
		assert (first_poly>0).all()
		scaling[first_poly_mask]=first_poly
		# assert check_symmetric(scaling)
		return scaling

class CovCM2p6(CovArray):
	from TransitionMatrix.__init__ import ROOT_DIR as DATA_DIR
	data_directory = DATA_DIR + '/data/cm2p6'
	variable_list = VariableList['o2','dic','temp','salt']
	label = 'cm2p6'
	from TransitionMatrix.Utilities.Data.__init__ import ROOT_DIR
	file_handler = FilePathHandler(ROOT_DIR,'CorCalc/'+label)

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)


		def combine_data(file_format,data_directory):
			matches = []
			for root, dirnames, filenames in os.walk(data_directory):
				for filename in fnmatch.filter(filenames, file_format):
					matches.append(os.path.join(root, filename))
			
			array_list = [np.load(match)[:,self.truth_array] for match in sorted(matches)]
			return np.vstack(array_list)

		salt = combine_data('*100m_subsampled_salt.npy',self.data_directory)
		dic = combine_data('*100m_subsampled_dic.npy',self.data_directory)
		o2 = combine_data('*100m_subsampled_o2.npy',self.data_directory)
		temp = combine_data('*100m_subsampled_temp.npy',self.data_directory)

		file_format = '*_time.npy'
		matches = self.get_filenames()
		
		self.time = flat_list([np.load(match).tolist() for match in sorted(matches)])

		salt = normalize_data(salt,'salt')
		dic = normalize_data(dic,'dic')
		o2 = normalize_data(o2,'o2')
		temp = normalize_data(temp,'temp')
		self.data = np.hstack([o2,dic,temp,salt])

		del salt
		del dic
		del o2
		del temp

	@staticmethod
	def dimensions_and_masks(lat_sep=None,lon_sep=None):
		lat_grid = np.arange(-90,91,lat_sep)
		lats = np.load(cov_array.data_directory+'lat_list.npy')
		lat_translate = [find_nearest(lats,x) for x in lat_grid] 
		lat_truth_list = np.array([x in lat_translate for x in lats])

		lon_grid = np.arange(-180,180,lon_sep)
		lons = np.load(cov_array.data_directory+'lon_list.npy')
		lons[lons<-180]=lons[lons<-180]+360
		lon_translate = [find_nearest(lons,x) for x in lon_grid] 
		lon_truth_list = np.array([x in lon_translate for x in lons])

		truth_list = lat_truth_list&lon_truth_list
		lats = lats[truth_list]
		lons = lons[truth_list]
		idx_lons = [find_nearest(lon_grid,x) for x in lons] 
		idx_lats = [find_nearest(lat_grid,x) for x in lats]
		index_list = [list(x) for x in list(zip(idx_lats,idx_lons))]		
		return (lats,lons,truth_list,lat_grid,lon_grid,index_list)

	def get_filenames(self):
		for root, dirnames, filenames in os.walk(self.data_directory):
			for filename in fnmatch.filter(filenames, file_format):
				matches.append(os.path.join(root, filename))
		return matches

class CovCM4(CovArray):
	from TransitionMatrix.__init__ import ROOT_DIR as DATA_DIR
	data_directory = DATA_DIR + '/data/cm4'
	chl_depth_idx = 10
	from TransitionMatrix.Utilities.Data.__init__ import ROOT_DIR
	label = 'cm4'
	max_depth_lev = 25  #this corresponds to 2062.5 meters 

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		if self.depth_idx<self.chl_depth_idx:
			self.variable_list = VariableList(['thetao','so','ph','chl','o2'])
		else:
			self.variable_list = VariableList(['thetao','so','ph','o2'])

	@classmethod			
	def make_cov_filename(cls,variable_1,variable_2):
		return cls.file_handler.tmp_file(variable_1+'_'+variable_2+'_cov')

	@staticmethod
	def return_dimensions():
		master_list = CovCM4.get_filenames()
		dh = Dataset( master_list[0][0][0])
		depths = dh['lev'][:].data
		lons = dh['lon'][:].data
		lats = dh['lat'][:].data
		return (lats,lons,depths)


	def stack_data(self):
		master_list = self.get_filenames()
		array_variable_list = []
		for files,variable in master_list:
			time_list = []
			holder_list = []
			if self.depth_idx>self.chl_depth_idx:
				if variable=='chl':
					continue
			for file in files:
				print(file)
				dh = Dataset(file)
				time_list.append(dh['time'][0])
				var_temp = dh[variable][:,self.depth_idx,:,:]
				holder_list.append(var_temp[:,self.truth_array].data)
			holder_total_list = np.vstack([x for _,x in sorted(zip(time_list,holder_list))])
			holder_total_list = self.normalize_data(holder_total_list,self.label+'_'+variable,plot=False)
			array_variable_list.append((holder_total_list,variable))
		del holder_total_list
		del holder_list
		del var_temp
		return array_variable_list

	def calculate_cov(self):
		array_variable_list =self.stack_data()
		arrays,variable = zip(*array_variable_list)
		holder = np.hstack(arrays)
		tt_cov = np.cov(holder)
		np.save(self.make_cov_filename(str(self.depth_idx)+'_'+'tt_cov',''),tt_cov)

		del arrays
		del variable
		del holder
		del tt_cov


		for comb in combinations(array_variable_list,2): 
			(array_1,variable_1),(array_2,variable_2) = comb
			array_1 = np.array(array_1)
			array_2 = np.array(array_2)
			if self.depth_idx>self.chl_depth_idx:
				if (variable_1=='chl')|(variable_2=='chl'):
					continue
			print(variable_1)
			print(variable_2)
			cov_output = np.cov(array_1.T,array_2.T)
			upper,lower = np.vsplit(cov_output,2)
			ul,ur = np.hsplit(upper,2)
			ll,lr = np.hsplit(lower,2)

			does_file_exist(self.make_cov_filename(str(self.depth_idx)+'_'+variable_1,variable_2),ur)
			does_file_exist(self.make_cov_filename(str(self.depth_idx)+'_'+variable_1,variable_1),ul)
			does_file_exist(self.make_cov_filename(str(self.depth_idx)+'_'+variable_2,variable_2),lr)

	def make_matrix(self):
		block_mat = np.zeros([len(self.variable_list),len(self.variable_list)]).tolist()
		num = 0
		for var in self.variable_list: 
			for var_1,var_2 in zip([var]*len(self.variable_list),self.variable_list):
				try:
					cov = np.load(self.make_cov_filename(str(self.depth_idx)+'_'+var_1,var_2)+'.npy')
					idx_1 = self.variable_list.index(var_1)
					idx_2 = self.variable_list.index(var_2)
				except FileNotFoundError:
					cov = np.load(self.make_cov_filename(str(self.depth_idx)+'_'+var_2,var_1)+'.npy')
					idx_1 = self.variable_list.index(var_2)
					idx_2 = self.variable_list.index(var_1)
				block_mat[idx_1][idx_2]=cov
				num +=1
				print(num)
				if var_1!=var_2:
					block_mat[idx_2][idx_1]=cov.T
					num += 1
					print(num)
		out_mat = np.block(block_mat)
		return out_mat 

	@classmethod
	def dimensions_and_masks(cls):
		files,var = cls.get_filenames()[0]
		file = files[0]
		dh = Dataset(file)
		temp = dh[var][:,cls.max_depth_lev,:,:]
		depth_mask = temp.mask[0] # no need to grab values deepeer than 2000 meters
		X,Y = np.meshgrid(np.floor(dh['lon'][:].data),np.floor(dh['lat'][:]).data)
		X[X>180] = X[X>180]-360
		subsample_mask = ~((X%cls.lon_sep==0)&(Y%cls.lat_sep==0))
		region_mask = ~((X>cls.lllon)&(X<cls.urlon)&(Y>cls.lllat)&(Y<cls.urlat))
		total_mask = (depth_mask)|(subsample_mask)|(region_mask)
		total_mask = ~total_mask
		X = X[total_mask]
		Y = Y[total_mask]
		geolist = GeoList([geopy.Point(x) for x in list(zip(Y,X))],lat_sep=cls.lat_sep,lon_sep=cls.lon_sep)
		return (total_mask,geolist)

	@staticmethod
	def get_filenames():
		master_list = []
		for holder in os.walk(CovCM4.data_directory):
			folder,dummy,files = holder
			variable = os.path.basename(folder)
			print(variable)
			files = [os.path.join(folder,file) for file in files if variable in file]
			if not files:
				continue
			master_list.append((files,variable))
		return master_list

	@staticmethod
	def load(depth_idx,lat=2,lon=2):
		from TransitionMatrix.Utilities.Inversion.target_load import InverseInstance		
		holder = CovCM4(depth_idx = depth_idx)
		holder.submeso_cov = InverseInstance.load_from_type(traj_type=CovCM4.label+'_submeso_covariance_'+str(depth_idx),lat_spacing=lat,lon_spacing=lon,l=300)
		holder.global_cov = InverseInstance.load_from_type(traj_type=CovCM4.label+'_global_covariance_'+str(depth_idx),lat_spacing=lat,lon_spacing=lon,l=1500)

class CovCM4Global(CovCM4):
	region = 'global'
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+region)
	lat_sep=2
	lon_sep=2
	l=300
	trans_geo = InverseGeo
	urlat = 90
	urlon = 180
	lllat = -90
	lllon = -180
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4GOM(CovCM4):
	region = 'gom'
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+region)
	lat_sep=1
	lon_sep=1
	l=100
	trans_geo = InverseGOM
	urlat = 30.5
	urlon = -81.5
	lllat = 20.5
	lllon = -100.

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4CCS(CovCM4):
	region = 'ccs'
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+region)
	lat_sep=1
	lon_sep=1
	l=100
	trans_geo = InverseCCS
	urlat = 55
	urlon = -105
	lllat = 20
	lllon = -135
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4SO(CovCM4):
	region = 'southern_ocean'
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+region)
	lat_sep=1
	lon_sep=2
	l=300
	trans_geo = InverseSOSE
	urlat = -40
	urlon = 180.
	lllat = -80.
	lllon = -180.	
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

