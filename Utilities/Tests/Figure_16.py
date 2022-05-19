import unittest
from OptimalArray.Utilities.H import Float, HInstance
from OptimalArray.Utilities.CM4Mat import CovCM4Global
from GeneralUtilities.Compute.list import VariableList
import geopy
from OptimalArray.Utilities.Plot.Figure_16 import FutureFloatTrans


depth_idx = 6
cov_holder = CovCM4Global.load(depth_idx = depth_idx)

class TestAdvanceHMethods1(unittest.TestCase):
	def setUp(self):
		self.H = HInstance(trans_geo = cov_holder.trans_geo)
		self.H.add_float(Float(geopy.Point(42,176),cov_holder.trans_geo.variable_list,datetime.datetime(2021,5,10)))
		self.H.add_float(Float(geopy.Point(42,176),cov_holder.trans_geo.variable_list,datetime.datetime(2019,5,10)))
		trans_mat = FutureFloatTrans.load_from_type(lat_spacing=2,lon_spacing=2,time_step=90)
		trans_list = []
		for x in range(16):
			trans_holder = trans_mat.multiply(x,value=0.00001)
			trans_holder.setup(days = 90*(1+x))
			trans_list.append(trans_holder)
		self.trans_list = trans_list
		self.future_H_list = [x.advance_H(H_array) for x in trans_list]

	def test_360_wrap(self):
		self.assertGreater(170,self.future_H_list[0]._list_of_floats[0].longitude)
		self.assertLess(-170,self.future_H_list[-1]._list_of_floats[0].longitude)

	def test_float_death(self):
		self.assertEqual(2,len(self.future_H_list[0]._list_of_floats))
		self.assertEqual(1,len(self.future_H_list[0]._list_of_floats))