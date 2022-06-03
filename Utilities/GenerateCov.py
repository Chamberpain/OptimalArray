from OptimalArray.Utilities.CM4Mat import CovCM4Global,CovCM4GlobalSubsample,CovCM4Indian,CovCM4SO,CovCM4NAtlantic,CovCM4TropicalAtlantic,CovCM4SAtlantic,CovCM4NPacific,CovCM4TropicalPacific,CovCM4SPacific,CovCM4GOM,CovCM4CCS
from OptimalArray.Utilities.MOM6Mat import CovMOM6CCS

import gc
import os
import shutil

def calculate_cov():
	for covclass in [CovCM4Global]:
		for depth in [8,26]:
		# for depth in [2,4,6,8,10,12,14,16,18,20,22,24]:
			print('depth idx is '+str(depth))
			dummy = covclass(depth_idx = depth)
			if os.path.isfile(dummy.trans_geo.make_inverse_filename()):
				continue
			try:
				dummy.calculate_cov()
				dummy.scale_cov()
			except FileNotFoundError:
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