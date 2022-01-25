from TransitionMatrix.Utilities.Data.CorCalc import CovCM4Global,CovCM4GOM,CovCM4CCS,CovCM4SO
import gc

def return_depth():
	dummy = CovCM4Global(depth_idx = depth)

for depth in [2,4,6,8,10,12,14,16,18,20]:
	for covclass in [CovCM4Global,CovCM4GOM,CovCM4CCS,CovCM4SO]:
	# goship_line_plot(depth_level=depth)
		dummy = covclass(depth_idx = depth)
		try:
			dummy.scale_cov()
		except FileNotFoundError:
			dummy.calculate_cov()
			dummy.scale_cov()
		dummy.save()
		del dummy
		gc.collect(generation=2)
