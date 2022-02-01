from OptimalArray.Utilities.CorMat import CovCM4Global,CovCM4Indian,CovCM4SO,CovCM4NAtlantic,CovCM4TropicalAtlantic,CovCM4SAtlantic,CovCM4NPacific,CovCM4TropicalPacific,CovCM4SPacific,CovCM4GOM,CovCM4CCS
import gc


for depth in [2,4,6,8,10,12,14,16,18,20]:
	for covclass in [CovCM4Global,CovCM4Indian,CovCM4SO,CovCM4NAtlantic,CovCM4TropicalAtlantic,CovCM4SAtlantic,CovCM4NPacific,CovCM4TropicalPacific,CovCM4SPacific,CovCM4GOM,CovCM4CCS]:
		dummy = covclass(depth_idx = depth)
		try:
			dummy.scale_cov()
		except FileNotFoundError:
			dummy.calculate_cov()
			dummy.scale_cov()
		dummy.save()
		del dummy
		gc.collect(generation=2)
