import unittest
from OptimalArray.Utilities.CM4Mat import CovCM4GlobalSubsample
from OptimalArray.Utilities.ComputeOptimal import make_noise_matrix
import scipy.sparse
import numpy as np 

class TestNoiseMatrix(unittest.TestCase):

	def setUp(self):
		depth_idx = 2
		self.cov_holder = CovCM4GlobalSubsample.load(depth_idx = depth_idx)

	def test_random(self):
		a = make_noise_matrix(self.cov_holder,1)
		b = make_noise_matrix(self.cov_holder,1)
		check_mat = a-b
		self.assertGreater(abs(check_mat).max(),10**-10)

	def test_positive_definite(self):
		a = make_noise_matrix(self.cov_holder,1)
		evals,evecs = scipy.sparse.linalg.eigs(a, 1, sigma=-1)
		self.assertGreater(evals[0],-10**-10)

	def test_symetric(self):
		mat = make_noise_matrix(self.cov_holder,1)
		check_mat = mat-mat.T
		self.assertGreater(10**-10,abs(check_mat).max())