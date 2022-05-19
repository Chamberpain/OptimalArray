import unittest
from OptimalArray.Utilities.H import Float, HInstance
from OptimalArray.Utilities.CM4Mat import CovCM4Global
from GeneralUtilities.Compute.list import VariableList
import geopy
from OptimalArray.Utilities.Plot.Figure_16 import FutureFloatTrans


depth_idx = 6
cov_holder = CovCM4Global.load(depth_idx = depth_idx)

class TestFloatMethods1(unittest.TestCase):
	def setUp(self):
		self.pos_idx = 20
		self.sensor_idx = [3,2]
		pos = cov_holder.trans_geo.total_list[20]
		variable_list = VariableList(cov_holder.trans_geo.variable_list[x] for x in self.sensor_idx)
		self.Float = Float(pos,variable_list)

	def test_position_index(self):
		self.assertEqual(self.Float.return_position_index(cov_holder.trans_geo),self.pos_idx,'Incorrect Float Position Indexing')

	def test_sensor_index(self):
		self.assertEqual(self.Float.return_sensor_index(cov_holder.trans_geo),self.sensor_idx,'Incorrect Sensor Indexing')

class TestRandomH(unittest.TestCase):

	def TestSensorDist(self):
		holder = HInstance.random_floats(cov_holder.trans_geo, 100, [1, 1, 0, 0, 1])
		self.assertTrue(np.unique(holder._index_of_sensors).tolist()==[0, 1,4])
		holder = HInstance.random_floats(cov_holder.trans_geo, 100, [1, 1, 0, .03, 1])
		self.assertTrue([0, 1,3,4] in np.unique(holder._index_of_sensors).tolist())
		num = len([x for x in holder._index_of_sensors if x == [0, 1, 3, 4]])
		self.assertTrue(num<=4)

	def TestArraySize(self):
		size = 100
		holder = HInstance.random_floats(cov_holder.trans_geo, size, [0.5, 1, 1, 0.5, 1])
		self.assertTrue(len(holder._index_of_sensors)==size)
		self.assertTrue(len(holder._index_of_pos)==size)

	def TestArrayDistribution(self):
		size = 1000
		holder = HInstance.random_floats(cov_holder.trans_geo, size, [0.5, 0.5, 0.5, 0.5, 0.5])
		self.assertTrue(len(np.unique(holder._index_of_sensors))==2**5)
		self.assertTrue(len(np.unique(holder._index_of_pos))>=990)

class TestBaseHMethods1(unittest.TestCase):
	def setUp(self):
		self.array_size = 10
		self.H = HInstance(trans_geo = cov_holder.trans_geo)
		for x in cov_holder.trans_geo.total_list[:self.array_size]:
			self.H.add_float(Float(x,cov_holder.trans_geo.variable_list))
		self.H.add_float(Float(cov_holder.trans_geo.total_list[0],cov_holder.trans_geo.variable_list[:1]))

	def test_return_H(self):
		dim1 = len(cov_holder.trans_geo.variable_list)*self.array_size
		dim2 = len(cov_holder.trans_geo.variable_list)*len(cov_holder.trans_geo.total_list)
		H = self.H.return_H()
		self.assertEqual(H.shape,(dim1,dim2),'Incorrect Dimension Matching')
		for k in range(len(cov_holder.trans_geo.variable_list)):
			self.assertTrue(all([H[x+(k*self.array_size),x+(k*len(cov_holder.trans_geo.total_list))]==1 for x in range(self.array_size)]),'H Matrix does not pass diagonal tests')

	def test_base_return(self):
		idx,data = self.H.base_return()
		self.assertEqual(len(idx),self.array_size*len(cov_holder.trans_geo.variable_list))
		self.assertEqual(len(data),self.array_size*len(cov_holder.trans_geo.variable_list))
		self.assertEqual(data[0],2)
		self.assertTrue(all([x==1 for x in data[1:]]))

	def test_return_noise_divider(self):
		noise_divider = self.H.return_noise_divider()
		self.assertEqual(noise_divider[0],2,'Test the values of noise_divider_output')
		self.assertEqual(len(noise_divider),self.array_size*len(cov_holder.trans_geo.variable_list),'Test the shape of noise_divider_output')
		self.assertTrue(all([x==1 for x in noise_divider[1:]]),'Test the values of noise_divider_output')


if __name__ == '__main__':
    unittest.main()