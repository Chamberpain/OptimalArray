import numpy as np
from netCDF4 import Dataset
from GeneralUtilities.Filepath.search import find_files
import os
import time
import matplotlib.pyplot as plt
from GeneralUtilities.Compute.list import GeoList, VariableList
import scipy.sparse.linalg
from scipy.sparse import csc_matrix
from scipy.sparse import _sparsetools
from scipy.sparse.sputils import (get_index_dtype,upcast)
from itertools import combinations
from GeneralUtilities.Filepath.instance import FilePathHandler
import geopy
import geopandas as gp
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

		self.trans_geo = self.trans_geo_class(depth_idx = depth_idx,variable_list=variable_list,model_type=self.label)
		truth_array,index_list = self.dimensions_and_masks()
		self.trans_geo.set_total_list(index_list)
		self.trans_geo.set_truth_array(truth_array)
		self.dist = self.get_dist()
		self.variable_list = variable_list

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

	def normalize_data(self,data,label=None,percent=0.4,scale=1):
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
		
	def subtract_e_vecs_return_space_space(self,e_vec_num=4):

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
		space_space_submeso = out_mat # assemble full covariance matrix from individual elements
		assert (space_space_submeso.diagonal()>=0).all()
		space_space_global = InverseInstance(space_space_submeso.shape,trans_geo=self.trans_geo)
		eig_vals, eig_vecs = scipy.sparse.linalg.eigs(space_space_submeso,k=4)
		eig_vecs = eig_vecs.real
		eig_vals = eig_vals.real
		dist_filter = self.dist<5*self.trans_geo.l # maximum distance we will have non zero values with gaspari and cohn scaling
		row_val,col_val = np.where(np.block([[dist_filter]*len(self.trans_geo.variable_list)]*len(self.trans_geo.variable_list)))
		submeso_data = space_space_submeso[(row_val,col_val)]
		space_space_submeso = InverseInstance((submeso_data,(row_val,col_val)),shape=space_space_submeso.shape,trans_geo = self.trans_geo)
		del dist_filter
		del submeso_data

		for k in range(len(eig_vals)):
			print('calculating eigen vector '+str(k))
			e_val = eig_vals[k]
			e_vec = eig_vecs[:,k]
			remove_e_vec = e_val*np.outer(e_vec,e_vec)/(e_vec**2).sum()
			e_vec_data = remove_e_vec[(row_val,col_val)]
			remove_e_vec = InverseInstance((e_vec_data,(row_val,col_val)),shape=space_space_submeso.shape,trans_geo = self.trans_geo)
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
			for k in range(len(self.trans_geo.variable_list)):
				temp_list = [cov_scale*holder]*len(self.trans_geo.variable_list)
				temp_list[k] = holder # reduce the covariance of cross variables by 30%
				total_list.append(temp_list)
			return scipy.sparse.csc_matrix(np.block(total_list))

		def calculate_scaling(dist, lat_sep=None,lon_sep=None,l=300):
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
		holder = calculate_scaling(self.dist,lat_sep=self.trans_geo.lat_sep,lon_sep=self.trans_geo.lon_sep,l=self.trans_geo.l) # calculate the gaspari and cohn localization for the submeso lengthscale
		submeso_scaling = make_scaling(holder) #make the localization that reduces cross covariances
		self.submeso_cov = submeso_cov.multiply(submeso_scaling)
		assert (self.submeso_cov.diagonal()>=0).all()
		del submeso_scaling
		holder = calculate_scaling(self.dist,lat_sep=self.trans_geo.lat_sep,lon_sep=self.trans_geo.lon_sep,l=self.trans_geo.l*5) # calculate the gaspari and cohn localization for the global lengthscale
		global_scaling = make_scaling(holder)
		del holder
		self.global_cov = global_cov.multiply(global_scaling)
		assert (self.global_cov.diagonal()>=0).all()

	def save(self):
		trans_geo = self.trans_geo.set_l_mult(5)
		mat_obj = InverseInstance(self.global_cov,shape = self.global_cov.shape,trans_geo=trans_geo)
		mat_obj.save()
		trans_geo = self.trans_geo.set_l_mult(1)
		mat_obj = InverseInstance(self.submeso_cov,shape = self.submeso_cov.shape,trans_geo=trans_geo)
		mat_obj.save()

	def get_cov(self,row_var,col_var):
		return CovElement.load(self.trans_geo,row_var,col_var)

	def get_dist(self):
		def calculate_distance():
			import geopy.distance
			dist = np.zeros([len(self.trans_geo.total_list),len(self.trans_geo.total_list)])
			for ii,coord1 in enumerate(self.trans_geo.total_list):
				print(ii)
				for jj,coord2 in enumerate(self.trans_geo.total_list):
					dist[ii,jj] = geopy.distance.great_circle(coord1,coord2).km 
			assert (dist>=0).all()&(dist<=40000).all()
			return dist

		filename = self.trans_geo.make_dist_filename()
		try:
			dist = np.load(filename+'.npy')
		except IOError:
			dist = calculate_distance()
			np.save(filename,dist)
		return dist

	def p_hat_calculate(self,H,index_list,noise_factor=2):
		noise = scipy.sparse.csc_matrix.diagonal(self.cov)[index_list]
		denom = H.dot(self.cov).dot(H.T)+scipy.sparse.diags(noise*noise_factor)
		inv_denom = scipy.sparse.linalg.inv(denom)
		if not type(inv_denom)==scipy.sparse.csc.csc_matrix:
			inv_denom = scipy.sparse.csc.csc_matrix(inv_denom)  # this is for the case of 1x1 which returns as array for some reason
		cov_subtract = self.cov.dot(H.T.dot(inv_denom).dot(H).dot(self.cov))
		# diag = scipy.sparse.csc_matrix.diagonal(self.cov)
		# for idx in np.where(diag<0)[0]:
		# 	print(idx)
		# 	cov[idx,idx]=0
		p_hat = self.cov-cov_subtract
		return p_hat,cov_subtract

