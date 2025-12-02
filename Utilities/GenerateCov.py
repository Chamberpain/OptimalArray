from OptimalArray.Utilities.CM4Mat import CovLowCM4Global
# CovLowCM4Indian,CovLowCM4SO,CovLowCM4NAtlantic,CovLowCM4TropicalAtlantic,CovLowCM4SAtlantic,CovLowCM4NPacific,CovLowCM4TropicalPacific,CovLowCM4SPacific,CovLowCM4GOM,CovLowCM4CCS
# from OptimalArray.Utilities.MOM6Mat import CovMOM6CCS

import gc
import os
import shutil

def calculate_cov():
	for covclass in [CovLowCM4Global]:
		# for depth in [8,26]:
		for depth in [4]:
			print('depth idx is '+str(depth))
			dummy = covclass(depth_idx = depth)
			if os.path.isfile(dummy.trans_geo.make_inverse_filename()):
				continue
			try:
				# dummy.stack_data()
				dummy.calculate_cov()
				dummy.scale_cov()
			except FileNotFoundError:
				# dummy.stack_data()
				dummy.calculate_cov()
				dummy.scale_cov()
			dummy.save()
			del dummy
			gc.collect(generation=2)


def move_files():
	for covclass in [CovCM4Global]:
		for depth in [8,26]:
			dummy = covclass(depth_idx = depth)
			base_filepath = dummy.trans_geo.file_handler.tmp_file('')
			new_filepath = base_filepath.replace('Data','Pipeline')
			shutil.rmtree(new_filepath)
			new_filepath = os.path.dirname(os.path.dirname(new_filepath))
			try:
				shutil.move(base_filepath,new_filepath)
			except OSError:
				pass

calculate_cov()