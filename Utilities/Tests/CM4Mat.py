import unittest
from OptimalArray.Utilities.CM4Mat import CovCM4Global
import scipy.sparse
import numpy as np 

class TestCM4MatPropertiesBase(unittest.TestCase):
	depth_idx = 2
	def setUp(self):
		self.cov_holder = CovCM4Global.load(depth_idx = self.depth_idx)

	def last_e_val_test(self):
		evals,evecs = scipy.sparse.linalg.eigs(self.cov_holder.cov, 1, sigma=-1)
		self.assertTrue(evals[0]>=0)

	def first_e_val_test(self):
		evals,evecs = scipy.sparse.linalg.eigs(self.cov_holder.cov, 1)
		self.assertTrue(evals[0]>=0)		

	def test_min_variance(self):
		var = self.cov_holder.cov.diagonal()
		self.assertTrue(all(var>=1))

	def test_max_variance(self):
		var = self.cov_holder.cov.diagonal()
		self.assertTrue(all(var<=(4.1*var.min())))

	def test_correlation(self):
		cor_list = []
		row_idx, col_idx, data = scipy.sparse.find(self.cov_holder.cov)
		var = self.cov_holder.cov.diagonal()
		for row,col,cov in zip(row_idx,col_idx,data):
			row_std = np.sqrt(var[row])
			col_std = np.sqrt(var[col])
			cor_list.append(cov/(row_std*col_std))
		self.assertTrue(max(cor_list)-1<10**(-6))
		self.assertTrue(min(cor_list)+1>10**(-6))


class TestCM4MatPropertiesBase4(TestCM4MatPropertiesBase):
	depth_idx = 4

class TestCM4MatPropertiesBase6(TestCM4MatPropertiesBase):
	depth_idx = 6

class TestCM4MatPropertiesBase8(TestCM4MatPropertiesBase):
	depth_idx = 8

class TestCM4MatPropertiesBase10(TestCM4MatPropertiesBase):
	depth_idx = 10

class TestCM4MatPropertiesBase12(TestCM4MatPropertiesBase):
	depth_idx = 12

class TestCM4MatPropertiesBase14(TestCM4MatPropertiesBase):
	depth_idx = 14

class TestCM4MatPropertiesBase16(TestCM4MatPropertiesBase):
	depth_idx = 16

class TestCM4MatPropertiesBase18(TestCM4MatPropertiesBase):
	depth_idx = 18

class TestCM4MatPropertiesBase20(TestCM4MatPropertiesBase):
	depth_idx = 20

class TestCM4MatPropertiesBase22(TestCM4MatPropertiesBase):
	depth_idx = 22

class TestCM4MatPropertiesBase24(TestCM4MatPropertiesBase):
	depth_idx = 24

class TestCM4MatPropertiesBase26(TestCM4MatPropertiesBase):
	depth_idx = 26


if __name__ == '__main__':
    unittest.main()