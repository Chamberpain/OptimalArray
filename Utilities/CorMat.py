import numpy as np
from netCDF4 import Dataset
from GeneralUtilities.Filepath.search import find_files
import os
import time
import matplotlib.pyplot as plt
from GeneralUtilities.Compute.list import GeoList, VariableList
import scipy.sparse.linalg
from scipy.sparse import csc_matrix
from GeneralUtilities.Filepath.instance import does_file_exist
from OptimalArray.Utilities.CorGeo import InverseGlobal,InverseIndian,InverseSO,InverseNAtlantic,InverseTropicalAtlantic,InverseSAtlantic,InverseNPacific,InverseTropicalPacific,InverseSPacific,InverseGOM,InverseCCS
from itertools import combinations
from GeneralUtilities.Filepath.instance import FilePathHandler
import geopy
import geopandas as gp

class CovElement(np.ndarray):
	"""
	Because there are multiple variables, every full covariance array has many individual covariance array subsets.
	"""

	def __new__(cls, input_array,trans_geo=None,row_var=None,col_var=None):
		# Create the ndarray instance of our type, given the usual
		# ndarray input arguments.  This will call the standard
		# ndarray constructor, but return an object of our type.
		# It also triggers a call to InfoArray.__array_finalize__

		eigs, eig_vecs = np.linalg.eig(input_array)
		eigs_sum_forward = np.array([eigs[:x].sum() for x in range(len(eigs))])
		eigs_idx = (eigs_sum_forward>0.99*eigs.sum()).tolist().index(True)
		# take the eigenvectors that explain 99% of the variance 
		input_array = np.zeros(eig_vecs.shape)
		for idx in np.arange(eigs_idx):
			temp = input_array+eigs[idx]*np.outer(eig_vecs[:,idx],eig_vecs[:,idx])
			input_array = temp
		obj = np.asarray(input_array).view(cls)

		# set the new 'info' attribute to the value passed
		obj.trans_geo = trans_geo
		obj.row_var = row_var
		obj.col_var = col_var
		# Finally, we must return the newly created object:
		return obj

	def __array_finalize__(self, obj):
		# ``self`` is a new object resulting from
		# ndarray.__new__(InfoArray, ...), therefore it only has
		# attributes that the ndarray.__new__ constructor gave it -
		# i.e. those of a standard ndarray.
		#
		# We could have got to the ndarray.__new__ call in 3 ways:
		# From an explicit constructor - e.g. InfoArray():
		#    obj is None
		#    (we're in the middle of the InfoArray.__new__
		#    constructor, and self.info will be set when we return to
		#    InfoArray.__new__)
		if obj is None: return
		# From view casting - e.g arr.view(InfoArray):
		#    obj is arr
		#    (type(obj) can be InfoArray)
		# From new-from-template - e.g infoarr[:3]
		#    type(obj) is InfoArray
		#
		# Note that it is here, rather than in the __new__ method,
		# that we set the default value for 'info', because this
		# method sees all creation of default objects - with the
		# InfoArray.__new__ constructor, but also with
		# arr.view(InfoArray).
		self.trans_geo = getattr(obj, 'trans_geo', None)
		self.row_var = getattr(obj, 'row_var', None)
		self.col_var = getattr(obj, 'col_var', None)



class CovArray(object):
	def __init__(self,*args,depth_idx=0,**kwargs):

		self.depth_idx = depth_idx
		self.truth_array,self.index_list = self.dimensions_and_masks()

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

	def normalize_data(self,data,label=None,percent=0.4,scale=1,plot=True):
		mean_removed = data-data.mean(axis=0)
		# only consider deviations from the mean
		data_scale = mean_removed.std(axis=0)

		dummy = 0
		greater_mask = data_scale>(data_scale.max()-dummy*0.001*data_scale.mean())
		while greater_mask.sum()<percent*len(data_scale): #stop below the 60th percentile
			dummy +=1
			greater_value = data_scale.max()-dummy*0.001*data_scale.mean() # increment in steps of 1000th of the mean
			greater_mask = data_scale>greater_value 
		print('greater value is '+str(greater_value))
		dummy = 0
		lesser_value = 0 
		lesser_mask = data_scale<lesser_value
		while lesser_mask.sum()<0: # this used to say "percent*len(data_scale):", but was changed
			dummy +=1
			lesser_value = data_scale.min()+dummy*0.001*data_scale.mean()
			lesser_mask = data_scale<lesser_value
		# because this is never less than zero, the lesser mask will not be applied
		print('lesser value is '+str(lesser_value))
		data_scale[greater_mask]=greater_value*scale
		data_scale[lesser_mask]=lesser_value*scale
		data_scale[~greater_mask&~lesser_mask] = data_scale[~greater_mask&~lesser_mask]*scale
		data_scale[data_scale==0]=10**-12
		return mean_removed/data_scale # everything below the 60th percentile will have a standard deviation of 1. The effect of this will be to target high variance regions first
		
	def subtract_e_vecs_return_space_space(self,e_vec_num=4,plot=False):
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
		space_space_submeso = self.make_matrix() # assemble full covariance matrix from individual elements
		assert (space_space_submeso.diagonal()>=0).all()
		space_space_global = scipy.sparse.csc_matrix(space_space_submeso.shape)
		eig_vals, eig_vecs = scipy.sparse.linalg.eigs(space_space_submeso,k=4)
		eig_vecs = eig_vecs.real
		eig_vals = eig_vals.real
		dist_filter = self.get_dist()<5*self.trans_geo.l # maximum distance we will have non zero values with gaspari and cohn scaling
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
				space_space_submeso -= space_space_submeso.diagonal().min()*scipy.sparse.eye(space_space_submeso.shape[0]) # matrices cannot have negative variance and is usually rounding error
			space_space_global += remove_e_vec
			if (space_space_global.diagonal()<0).any():
				space_space_global -= space_space_global.diagonal().min()*scipy.sparse.eye(space_space_global.shape[0]) # matrices cannot have negative variance and is usually rounding error
		return (space_space_submeso,space_space_global)

	def scale_cov(self):
		def make_scaling(holder):
			cov_scale = 0.7 
			total_list = []
			for k in range(len(self.variable_list)):
				temp_list = [cov_scale*holder]*len(self.variable_list)
				temp_list[k] = holder # reduce the covariance of cross variables by 30%
				total_list.append(temp_list)
			return scipy.sparse.csc_matrix(np.block(total_list))

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
			return scaling

		submeso_cov,global_cov = self.subtract_e_vecs_return_space_space() #get the global and submeso covariances
		assert (submeso_cov.diagonal()>=0).all()
		assert (global_cov.diagonal()>=0).all()
		holder = self.calculate_scaling(lat_sep=self.trans_geo.lat_sep,lon_sep=self.trans_geo.lon_sep,l=self.trans_geo.l) # calculate the gaspari and cohn localization for the submeso lengthscale
		submeso_scaling = make_scaling(holder) #make the localization that reduces cross covariances
		self.submeso_cov = submeso_cov.multiply(submeso_scaling)
		assert (self.submeso_cov.diagonal()>=0).all()
		del submeso_scaling
		holder = self.calculate_scaling(lat_sep=self.trans_geo.lat_sep,lon_sep=self.trans_geo.lon_sep,l=self.trans_geo.l*5) # calculate the gaspari and cohn localization for the global lengthscale
		global_scaling = make_scaling(holder)
		del holder
		self.global_cov = global_cov.multiply(global_scaling)
		assert (self.global_cov.diagonal()>=0).all()

	@classmethod			
	def make_cov_filename(cls,variable_1,variable_2):
		return cls.file_handler.tmp_file(variable_1+'_'+variable_2+'_cov')

	def save(self):
		from OptimalArray.Utilities.Inversion.target_load import InverseInstance		
		trans_geo = self.trans_geo(l_mult = 5,depth_idx=self.depth_idx)
		trans_geo.set_total_list(self.index_list)
		filename = trans_geo.make_filename()
		mat_obj = InverseInstance(self.global_cov,shape = self.global_cov.shape,trans_geo=trans_geo)
		mat_obj.save(filename=filename)

		trans_geo = self.trans_geo(l_mult = 1,depth_idx=self.depth_idx)
		trans_geo.set_total_list(self.index_list)
		filename = trans_geo.make_filename()
		mat_obj = InverseInstance(self.submeso_cov,shape = self.submeso_cov.shape,trans_geo=trans_geo)
		mat_obj.save(filename=filename)

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

		filename = self.file_handler.tmp_file(self.trans_geo.region+'_distance_lat_'+str(self.trans_geo.lat_sep)+'_lon_'+str(self.trans_geo.lon_sep))
		try:
			dist = np.load(filename+'.npy')
		except IOError:
			dist = calculate_distance()
			np.save(filename,dist)
		return dist



class CovCM4(CovArray):
	from OptimalArray.Data.__init__ import ROOT_DIR as DATA_DIR
	data_directory = DATA_DIR + '/cm4'
	chl_depth_idx = 10
	from OptimalArray.__init__ import ROOT_DIR
	label = 'cm4'
	max_depth_lev = 25  #this corresponds to 2062.5 meters 

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		if self.depth_idx<self.chl_depth_idx:
			self.variable_list = VariableList(['thetao','so','ph','chl','o2'])
		else:
			self.variable_list = VariableList(['thetao','so','ph','o2'])

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

	@classmethod
	def dimensions_and_masks(cls):
		files,var = cls.get_filenames()[0]
		file = files[0]
		dh = Dataset(file)
		temp = dh[var][:,cls.max_depth_lev,:,:]
		depth_mask = ~temp.mask[0] # no need to grab values deepeer than 2000 meters
		X,Y = np.meshgrid(np.floor(dh['lon'][:].data),np.floor(dh['lat'][:]).data)
		X[X>180] = X[X>180]-360
		subsample_mask = ((X%cls.trans_geo.lon_sep==0)&(Y%cls.trans_geo.lat_sep==0))
		region_mask = ((X>cls.trans_geo().lllon)&(X<cls.trans_geo().urlon)&(Y>cls.trans_geo().lllat)&(Y<cls.trans_geo().urlat))

		geolist = GeoList([geopy.Point(x) for x in list(zip(Y.ravel(),X.ravel()))],lat_sep=cls.trans_geo.lat_sep,lon_sep=cls.trans_geo.lon_sep)
		oceans_list = []
		for k,dummy in enumerate(geolist.to_shapely()):
			print(k)
			oceans_list.append(cls.trans_geo().ocean_shape.contains(dummy))	# only choose coordinates within the ocean basin of interest

		total_mask = (depth_mask)&(subsample_mask)&(region_mask)&(np.array(oceans_list).reshape(X.shape))
		X = X[total_mask]
		Y = Y[total_mask]
		geolist = GeoList([geopy.Point(x) for x in list(zip(Y,X))],lat_sep=cls.trans_geo.lat_sep,lon_sep=cls.trans_geo.lon_sep)

		return (total_mask,geolist)

	@staticmethod
	def get_filenames():
		master_list = []
		for holder in os.walk(CovCM4.data_directory):
			folder,dummy,files = holder
			folder = folder[1:]
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
	trans_geo = InverseGlobal
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+trans_geo.region)
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4Indian(CovCM4):
	trans_geo = InverseIndian
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+trans_geo.region)
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4SO(CovCM4):
	trans_geo = InverseSO
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+trans_geo.region)
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4NAtlantic(CovCM4):
	trans_geo = InverseNAtlantic
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+trans_geo.region)
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4TropicalAtlantic(CovCM4):
	trans_geo = InverseTropicalAtlantic
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+trans_geo.region)
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4SAtlantic(CovCM4):
	trans_geo = InverseSAtlantic
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+trans_geo.region)
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4NPacific(CovCM4):
	trans_geo = InverseNPacific
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+trans_geo.region)
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4TropicalPacific(CovCM4):
	trans_geo = InverseTropicalAtlantic
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+trans_geo.region)
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4SPacific(CovCM4):
	trans_geo = InverseSPacific
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+trans_geo.region)
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4GOM(CovCM4):
	trans_geo = InverseGOM
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+trans_geo.region)
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class CovCM4CCS(CovCM4):
	trans_geo = InverseCCS
	file_handler = FilePathHandler(CovCM4.ROOT_DIR,'CorCalc/'+CovCM4.label+'/'+trans_geo.region)
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
