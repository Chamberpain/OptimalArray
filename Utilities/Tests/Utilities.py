import unittest
from OptimalArray.Utilities.H import Float, HInstance
from OptimalArray.Utilities.CM4Mat import CovCM4Global,CovCM4GOM
from OptimalArray.Utilities.Utilities import make_R,make_Gain,make_GainFactor,make_CovSubtract,make_P_hat
import matplotlib.pyplot as plt

import numpy as np

depth_idx = 6
cov_holder = CovCM4GOM.load(depth_idx = depth_idx)

class TestMakeR1(unittest.TestCase):
	def setUp(self):
		self.noise_factor = 1.4
		self.array_size = 10
		H = HInstance(trans_geo = cov_holder.trans_geo)
		for x in cov_holder.trans_geo.total_list[:self.array_size]:
			H.add_float(Float(x,cov_holder.trans_geo.variable_list))
		H.add_float(Float(cov_holder.trans_geo.total_list[0],cov_holder.trans_geo.variable_list[:1]))
		H.add_float(Float(cov_holder.trans_geo.total_list[1],cov_holder.trans_geo.variable_list[:1]))
		H.add_float(Float(cov_holder.trans_geo.total_list[1],cov_holder.trans_geo.variable_list[:1]))
		self.R = make_R(H,cov_holder,self.noise_factor)

	def test_R_shape(self):
		dim = self.array_size*len(cov_holder.trans_geo.variable_list)
		self.assertEqual(self.R.shape,(dim,dim),'Incorrect R matrix shape')

	def test_R_data(self):
		idx_list = []
		for k in range(len(cov_holder.trans_geo.variable_list)):
			for ii in range(self.array_size):
				idx_list.append(ii+k*len(cov_holder.trans_geo.total_list))
		data = np.ravel(self.R.data)
		for x in data:
			self.assertGreater(x,0)
		var_list = [cov_holder.cov[ii,ii] for ii in idx_list]
		noise_scale_list = data/np.array(var_list)
		print(noise_scale_list[2:])
		self.assertAlmostEqual(noise_scale_list[0],self.noise_factor/2)
		self.assertAlmostEqual(noise_scale_list[1],self.noise_factor/3)
		for x in noise_scale_list[2:]:
			self.assertAlmostEqual(x,self.noise_factor)

class TestMakeGainFactor1(unittest.TestCase):
	def setUp(self):
		noise_factor = 0.2
		array_size = 10
		H = HInstance(trans_geo = cov_holder.trans_geo)
		for x in cov_holder.trans_geo.total_list[:array_size]:
			H.add_float(Float(x,cov_holder.trans_geo.variable_list))
		H.add_float(Float(cov_holder.trans_geo.total_list[0],cov_holder.trans_geo.variable_list[:1]))
		H.add_float(Float(cov_holder.trans_geo.total_list[1],cov_holder.trans_geo.variable_list[:1]))
		H.add_float(Float(cov_holder.trans_geo.total_list[1],cov_holder.trans_geo.variable_list[:1]))
		R = make_R(H,cov_holder,noise_factor)
		factor = make_GainFactor(H.return_H(),cov_holder.cov,R)

class TestMakeCovSubtract1(unittest.TestCase):
	def setUp(self):
		self.noise_factor = 1.4
		self.array_size = 10
		H = HInstance(trans_geo = cov_holder.trans_geo)
		for x in cov_holder.trans_geo.total_list[:self.array_size]:
			H.add_float(Float(x,cov_holder.trans_geo.variable_list))
		H.add_float(Float(cov_holder.trans_geo.total_list[0],cov_holder.trans_geo.variable_list[:1]))
		H.add_float(Float(cov_holder.trans_geo.total_list[1],cov_holder.trans_geo.variable_list[:1]))
		H.add_float(Float(cov_holder.trans_geo.total_list[1],cov_holder.trans_geo.variable_list[:1]))
		R = make_R(H,cov_holder,self.noise_factor)
		self.Inv = make_Inverse(H.return_H(),cov_holder.cov,R)





if __name__ == '__main__':
    unittest.main()