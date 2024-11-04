import numpy as np
from netCDF4 import Dataset
from GeneralUtilities.Data.Filepath.search import find_files
import os
import time
from GeneralUtilities.Compute.list import GeoList, VariableList
import scipy.sparse.linalg
from scipy.sparse import csc_matrix
from scipy.sparse import _sparsetools
from scipy.sparse.sputils import (get_index_dtype,upcast)
from itertools import combinations
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
import geopy
import pickle


class InverseInstance(scipy.sparse.csc_matrix):
	"""Base class for transition and correlation matrices"""
	def __init__(self, *args,trans_geo=None,**kwargs):
		super().__init__(*args,**kwargs)
		self.trans_geo = trans_geo

	@staticmethod
	def load(trans_geo):
		filename = trans_geo.make_inverse_filename()
		with open(filename,'rb') as pickle_file:
			out_data = pickle.load(pickle_file)
		return out_data

	def save(self):
		filename = self.trans_geo.make_inverse_filename()
		with open(filename, 'wb') as pickle_file:
			pickle.dump(self,pickle_file)
		pickle_file.close()


	def new_sparse_matrix(self,data):
		row_idx,column_idx,dummy = scipy.sparse.find(self)
		return BaseMat((data,(row_idx,column_idx)),shape=(len(self.trans_geo.total_list),len(self.trans_geo.total_list)))             

	def mean(self,axis=0):
		return np.array(self.sum(axis=axis)/(self!=0).sum(axis=axis)).flatten()

	def _binopt(self, other, op):
		""" This is included so that when sparse matrices are added together, their instance variables are maintained this code was grabbed from the scipy source with the small addition at the end"""

		other = self.__class__(other)

		# e.g. csr_plus_csr, csr_minus_csr, etc.
		fn = getattr(_sparsetools, self.format + op + self.format)

		maxnnz = self.nnz + other.nnz
		idx_dtype = get_index_dtype((self.indptr, self.indices,
									 other.indptr, other.indices),
									maxval=maxnnz)
		indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
		indices = np.empty(maxnnz, dtype=idx_dtype)

		bool_ops = ['_ne_', '_lt_', '_gt_', '_le_', '_ge_']
		if op in bool_ops:
			data = np.empty(maxnnz, dtype=np.bool_)
		else:
			data = np.empty(maxnnz, dtype=upcast(self.dtype, other.dtype))

		fn(self.shape[0], self.shape[1],
		   np.asarray(self.indptr, dtype=idx_dtype),
		   np.asarray(self.indices, dtype=idx_dtype),
		   self.data,
		   np.asarray(other.indptr, dtype=idx_dtype),
		   np.asarray(other.indices, dtype=idx_dtype),
		   other.data,
		   indptr, indices, data)
		if issubclass(type(self),InverseInstance):
			A = self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo)
		else:
			A = self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo)				
		A.prune()

		return A


	def _mul_sparse_matrix(self, other):
		""" This is included so that when sparse matrices are multiplies together, 
		their instance variables are maintained this code was grabbed from the scipy 
		source with the small addition at the end"""
		M, K1 = self.shape
		K2, N = other.shape

		major_axis = self._swap((M, N))[0]
		other = self.__class__(other)  # convert to this format

		idx_dtype = get_index_dtype((self.indptr, self.indices,
									 other.indptr, other.indices))

		fn = getattr(_sparsetools, self.format + '_matmat_maxnnz')
		nnz = fn(M, N,
				 np.asarray(self.indptr, dtype=idx_dtype),
				 np.asarray(self.indices, dtype=idx_dtype),
				 np.asarray(other.indptr, dtype=idx_dtype),
				 np.asarray(other.indices, dtype=idx_dtype))

		idx_dtype = get_index_dtype((self.indptr, self.indices,
									 other.indptr, other.indices),
									maxval=nnz)

		indptr = np.empty(major_axis + 1, dtype=idx_dtype)
		indices = np.empty(nnz, dtype=idx_dtype)
		data = np.empty(nnz, dtype=upcast(self.dtype, other.dtype))

		fn = getattr(_sparsetools, self.format + '_matmat')
		fn(M, N, np.asarray(self.indptr, dtype=idx_dtype),
		   np.asarray(self.indices, dtype=idx_dtype),
		   self.data,
		   np.asarray(other.indptr, dtype=idx_dtype),
		   np.asarray(other.indices, dtype=idx_dtype),
		   other.data,
		   indptr, indices, data)

		if issubclass(type(self),InverseInstance):
			return self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo)
		else:
			return self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo)



class CovElement(np.ndarray):
	"""
	Because there are multiple variables, every full covariance array has many individual covariance array subsets.
	"""

	def __new__(cls, input_array,trans_geo=None,row_var=None,col_var=None):
		# Create the ndarray instance of our type, given the usual
		# ndarray input arguments.  This will call the standard
		# ndarray constructor, but return an object of our type.
		# It also triggers a call to InfoArray.__array_finalize__

		# eigs, eig_vecs = np.linalg.eig(input_array)
		# eigs_sum_forward = np.array([eigs[:x].sum() for x in range(len(eigs))])
		# eigs_idx = (eigs_sum_forward>0.99*eigs.sum()).tolist().index(True)
		# # take the eigenvectors that explain 99% of the variance 
		# input_array = np.zeros(eig_vecs.shape)
		# for idx in np.arange(eigs_idx):
		# 	temp = input_array+eigs[idx]*np.outer(eig_vecs[:,idx],eig_vecs[:,idx])
		# 	input_array = temp
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
			

	def save(self):
		file_name = self.trans_geo.make_cov_filename(self.row_var,self.col_var)
		if os.path.isfile(file_name):
			print(file_name)
			print('does exist, I will not remake the covariance')
			pass
		else:
			with open(file_name, 'wb') as pickle_file:
				pickle.dump(self,pickle_file)
			pickle_file.close()

	@staticmethod
	def load(trans_geo,row_var,col_var):
		file_name = trans_geo.make_cov_filename(row_var,col_var)
		with open(file_name,'rb') as pickle_file:
			out_data = pickle.load(pickle_file)
		out_data.trans_geo = trans_geo
		out_data.row_var = row_var
		out_data.col_var = col_var
		return out_data


class CovArray(object):
	def __init__(self,*args,depth_idx=0,variable_list = None,**kwargs):

		self.trans_geo = self.trans_geo_class(depth_idx = depth_idx,variable_list=variable_list)
		truth_array,index_list = self.dimensions_and_masks()
		self.trans_geo.set_total_list(index_list)
		self.trans_geo.set_truth_array(truth_array)
		self.dist = self.get_dist()
		assert isinstance(variable_list,VariableList)

	def calculate_cov(self):
		array_variable_list =self.stack_data()
		arrays,variable = zip(*array_variable_list)
		holder = np.hstack(arrays)
		tt_cov = np.cov(holder)
		tt_cov = CovElement(tt_cov,trans_geo=self.trans_geo,row_var='tt',col_var='tt')
		tt_cov.save()

		del arrays
		del variable
		del holder
		del tt_cov

		for comb in combinations(array_variable_list,2): 
			(array_1,variable_1),(array_2,variable_2) = comb
			array_1 = np.array(array_1)
			array_2 = np.array(array_2)
			print(variable_1)
			print(variable_2)
			cov_output = np.cov(array_1.T,array_2.T)
			upper,lower = np.vsplit(cov_output,2)
			ul,ur = np.hsplit(upper,2)
			ll,lr = np.hsplit(lower,2)

			CovElement(ur,trans_geo=self.trans_geo,row_var=variable_1,col_var=variable_2).save()
			CovElement(ul,trans_geo=self.trans_geo,row_var=variable_1,col_var=variable_1).save()
			CovElement(lr,trans_geo=self.trans_geo,row_var=variable_2,col_var=variable_2).save()

	def normalize_data(self,data,lower_percent=0.85,upper_percent = 0.90, scale=1):
		mean_removed = data-data.mean(axis=0)
		# only consider deviations from the mean
		data_scale = mean_removed.var(axis=0)
		for idx in np.where(data_scale == 0)[0].tolist():
			print('index = ',idx,' has zero variance, so we are creating an artifical time series')
			mean_var = data_scale[data_scale != 0].mean()
			base_series = np.random.uniform(0, 1, mean_removed.shape[0])
			scaled_series = mean_var/base_series.var()*base_series
			mean_removed[:,idx] = scaled_series
			assert (mean_removed[:,idx] - scaled_series < 10**-4).all()
		data_scale = mean_removed.var(axis=0)
		assert len(data_scale[data_scale == 0]) == 0
		# amp = data_scale[data_scale!=0].min()
		# mean_removed[:,data_scale == 0]=np.random.uniform(0,amp,len(mean_removed[:,data_scale == 0].ravel())).reshape(mean_removed[:,data_scale == 0].shape)
		# data_scale = mean_removed.var(axis=0)

		dummy = 0
		greater_mask = data_scale>(data_scale.max()-dummy*0.001*data_scale.mean()) #set first mask to choose everything greater than the maximum value
		while greater_mask.sum()<lower_percent*len(data_scale): #stop below the 20th percentile
			dummy +=1
			greater_value = data_scale.max()-dummy*0.001*data_scale.mean() # increment in steps of 1000th of the mean
			greater_mask = data_scale>greater_value # mask will choose everything greater
		print('greater value is '+str(greater_value))
		dummy = 0
		lesser_value = 15*greater_value
		lesser_mask = data_scale<lesser_value
		# while lesser_mask.sum()<upper_percent*len(data_scale): # this used to say "percent*len(data_scale):", but was changed
		# 	dummy +=1
		# 	lesser_value = data_scale.min()+dummy*0.001*data_scale.mean()
		# 	lesser_mask = data_scale<lesser_value
		print('lesser value is '+str(lesser_value))
		data_scale[~greater_mask]=data_scale[~greater_mask]*scale #everything below 20th percentile will have var = 1
		assert len(data_scale[data_scale == 0]) == 0
		data_scale[greater_mask&lesser_mask]=greater_value*scale 
		assert len(data_scale[data_scale == 0]) == 0
		data_scale[~lesser_mask] = data_scale[~lesser_mask]*(greater_value)/(lesser_value)*scale # everything above the 95th percentile will have var = 15
		assert len(data_scale[data_scale == 0]) == 0
		return_data = mean_removed/np.sqrt(data_scale)
		assert(len(data_scale[data_scale==0])==0)
		return (mean_removed,return_data, data_scale) # everything below the 60th percentile will have a standard deviation of 1. The effect of this will be to target high variance regions first

	def subsample_variable(self,variable_list):
		assert isinstance(variable_list,VariableList)
		block_mat = np.zeros([len(variable_list),len(variable_list)]).tolist()
		block_length = len(self.trans_geo.total_list)
		var_idx_list = [self.trans_geo.variable_list.index(x) for x in variable_list]
		for row_idx in var_idx_list:
			start_row_idx = row_idx*block_length
			end_row_idx = (row_idx+1)*block_length
			for col_idx in var_idx_list:
				start_col_idx = col_idx*block_length
				end_col_idx = (col_idx+1)*block_length
				data_matrix = self.cov[start_row_idx:end_row_idx,start_col_idx:end_col_idx]
				block_mat[row_idx][col_idx] = data_matrix
		out_mat = scipy.sparse.bmat(block_mat)
		out_mat = InverseInstance(out_mat.tocsc(),trans_geo = self.trans_geo)
		holder = self.__class__(depth_idx = self.trans_geo.depth_idx)
		holder.cov = out_mat
		holder.trans_geo = self.trans_geo
		holder.trans_geo.variable_list = variable_list
		holder.cov.trans_geo.variable_list = variable_list

		return holder		

	def assemble_covariance(self):
		block_mat = np.zeros([len(self.trans_geo.variable_list),len(self.trans_geo.variable_list)]).tolist()
		num = 0
		for var in self.trans_geo.variable_list: 
			for var_1,var_2 in zip([var]*len(self.trans_geo.variable_list),self.trans_geo.variable_list):
				try:
					cov = CovElement.load(self.trans_geo,var_1,var_2)
					idx_1 = self.trans_geo.variable_list.index(var_1)
					idx_2 = self.trans_geo.variable_list.index(var_2)
				except FileNotFoundError:
					cov = CovElement.load(self.trans_geo,var_2,var_1)
					idx_1 = self.trans_geo.variable_list.index(var_2)
					idx_2 = self.trans_geo.variable_list.index(var_1)
				block_mat[idx_1][idx_2]=cov
				num +=1
				print(num)
				if var_1!=var_2:
					block_mat[idx_2][idx_1]=cov.T
					num += 1
					print(num)
		out_mat = np.block(block_mat)		
		assert (out_mat.diagonal()>=0).all()
		return out_mat

	def subtract_e_vecs_return_space_space(self,e_vec_num=4):
		space_space_submeso = self.assemble_covariance() # assemble full covariance matrix from individual elements
		space_space_global = np.zeros(space_space_submeso.shape)
		eig_vals, eig_vecs = scipy.sparse.linalg.eigs(scipy.sparse.csc_matrix(space_space_submeso),k=4)
		print('i have calculated the eigenvalues')
		for k in range(len(eig_vals)):
			print('calculating eigen vector '+str(k))
			e_val = eig_vals[k]
			e_vec = eig_vecs[:,k]
			e_vec = e_vec.reshape([len(e_vec),1])
			remove_e_vec = e_val.real*e_vec.real.dot(e_vec.real.T)/(e_vec.real**2).sum()
			# e_vec_data = remove_e_vec[(row_val,col_val)]
			# remove_e_vec = InverseInstance((e_vec_data,(row_val,col_val)),shape=space_space_submeso.shape,trans_geo = self.trans_geo)
			space_space_submeso -= remove_e_vec
			# if (space_space_submeso.diagonal()<0).any():
			# 	space_space_submeso -= space_space_submeso.diagonal().min()*scipy.sparse.eye(space_space_submeso.shape[0]) # matrices cannot have negative variance and is usually rounding error
			space_space_global += remove_e_vec
			# if (space_space_global.diagonal()<0).any():
			# 	space_space_global -= space_space_global.diagonal().min()*scipy.sparse.eye(space_space_global.shape[0]) # matrices cannot have negative variance and is usually rounding error
		# submeso_space_space_eig_vals = np.linalg.eigvals(space_space_submeso)
		# global_space_space_eig_vals = np.linalg.eigvals(space_space_global)
		# print(submeso_space_space_eig_vals.min())
		# print(global_space_space_eig_vals.min())
		# print(submeso_space_space_eig_vals.max())
		# print(global_space_space_eig_vals.max())
		# print(out_mat-space_space_global-space_space_submeso)
		return (space_space_submeso,space_space_global)


	def make_scaling(self,holder):
#Turns single dimensional matrix into a matrix the size of the full covariance matrix
		cov_scale = 1
		total_list = []
		for k in range(len(self.trans_geo.variable_list)):
			temp_list = [cov_scale*scipy.sparse.csc_matrix(holder)]*len(self.trans_geo.variable_list)
			temp_list[k] = scipy.sparse.csc_matrix(holder) # allows the covariance of cross variables to be reduced
			total_list.append(temp_list)
		return scipy.sparse.bmat(total_list)

	def calculate_scaling(self,l=3):
		assert (self.dist>=0).all()
		c = np.sqrt(10/3.)*l
#For this scaling we use something derived by gassbury and coehn to not significantly change eigen spectrum of 
#local support scaling function
#last peice wise poly 		
		scaling = np.zeros(self.dist.shape)
		# self.dist[self.dist>2*c]=0
		second_poly_mask = (self.dist>c)&(self.dist<2*c)
		self.dist_holder = self.dist[second_poly_mask].flatten()
		assert (self.dist_holder.min()>c)&(self.dist_holder.max()<2*c)
		second_poly = 1/12.*(self.dist_holder/c)**5 \
		-1/2.*(self.dist_holder/c)**4 \
		+5/8.*(self.dist_holder/c)**3 \
		+5/3.*(self.dist_holder/c)**2 \
		-5.*(self.dist_holder/c) \
		+4 \
		- 2/3.*(c/self.dist_holder)
		second_poly[second_poly<0]=0 
		scaling[second_poly_mask]=second_poly

		first_poly_mask = (self.dist<c)
		self.dist_holder = self.dist[first_poly_mask].flatten()
		assert (self.dist_holder.min()>=0)&(self.dist_holder.max()<c)

		first_poly = -1/4.*(self.dist_holder/c)**5 \
		+1/2.*(self.dist_holder/c)**4 \
		+5/8.*(self.dist_holder/c)**3 \
		-5/3.*(self.dist_holder/c)**2 \
		+1
		assert (first_poly>0).all()
		scaling[first_poly_mask]=first_poly
		return scaling

	def scale_cov(self):

		def check_evals(mat):
			evals,evecs = scipy.sparse.linalg.eigs(scipy.sparse.csc_matrix(mat),k=1, sigma=-.5)
			assert(evals.min()>-10**(-10))			

		def check_symmetric(mat):
			check_mat = mat-mat.T
			assert(abs(check_mat).max()<10**-10)	

		def plot_evals(mat):
			fig = plt.figure()
			ax = fig.add_subplot(1, 1, 1)
			evals = np.linalg.eigvals(mat)
			evals = np.sort(evals)
			ax.plot(-evals[evals<0])
			ax.set_yscale('log')
			plt.show()


		submeso_cov,global_cov = self.subtract_e_vecs_return_space_space() #get the global and submeso covariances
		assert (submeso_cov.diagonal().min()>-10**-8)
		assert (global_cov.diagonal().min()>-10**-8)
		holder = self.calculate_scaling(l=self.trans_geo.l) # calculate the gaspari and cohn localization for the submeso lengthscale
		submeso_scaling = self.make_scaling(holder) #make the localization that reduces cross covariances
		self.submeso_cov = scipy.sparse.csc_matrix(submeso_cov).multiply(submeso_scaling)
		del submeso_scaling

		holder = self.calculate_scaling(l=self.trans_geo.l*2) # calculate the gaspari and cohn localization for the global lengthscale
		global_scaling = self.make_scaling(holder) 
		del holder
		self.global_cov = scipy.sparse.csc_matrix(global_cov).multiply(global_scaling)
		assert (self.global_cov.diagonal()>=0).all()

	def save(self):
		trans_geo = self.trans_geo.set_l_mult(2)
		mat_obj = InverseInstance(self.global_cov,shape = self.global_cov.shape,trans_geo=trans_geo)
		mat_obj.save()
		trans_geo = self.trans_geo.set_l_mult(1)
		mat_obj = InverseInstance(self.submeso_cov,shape = self.submeso_cov.shape,trans_geo=trans_geo)
		mat_obj.save()

	def get_cov(self,row_var,col_var):
		return CovElement.load(self.trans_geo,row_var,col_var)

	def get_dist(self):

		def calculate_distance_degree():
			self.trans_geo.get_direction_matrix()
			dist = np.sqrt((self.trans_geo.north_south*self.trans_geo.lon_sep)**2 + (self.trans_geo.east_west*self.trans_geo.lat_sep)**2)
			assert (dist>=0).all()&(dist<=360).all()
			return dist

		filename = self.trans_geo.make_dist_filename()
		try:
			dist = np.load(filename+'.npy')
		except IOError:
			dist = calculate_distance_degree()
			np.save(filename,dist)
		return dist