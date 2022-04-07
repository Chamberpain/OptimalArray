from OptimalArray.Utilities.CorMat import CovCM4Global,CovCM4Indian,CovCM4SO,CovCM4NAtlantic,CovCM4TropicalAtlantic,CovCM4SAtlantic,CovCM4NPacific,CovCM4TropicalPacific,CovCM4SPacific,CovCM4GOM,CovCM4CCS
import matplotlib.pyplot as plt
import cartopy.crs as ccrs



def plot_cov(self):
	plottable = self.trans_geo.transition_vector_to_plottable(self.diagonal())
	XX,YY,ax = self.trans_geo.plot_setup()
	ax.pcolormesh(XX,YY,plottable,vmax = (plottable.mean()+(3*plottable.std())))


for covclass in [CovCM4Global,CovCM4Indian,CovCM4SO,CovCM4NAtlantic,CovCM4TropicalAtlantic,CovCM4SAtlantic,CovCM4NPacific,CovCM4TropicalPacific,CovCM4SPacific,CovCM4GOM,CovCM4CCS]:
	for depth in [2,4,6,8,10,12,14,16,18,20]:
		dummy = covclass(depth_idx = depth)
		for var_1 in dummy.trans_geo.variable_list:
			for var_2 in dummy.trans_geo.variable_list:
				try:
					cov = dummy.get_cov(var_1,var_2)
					idx_1 = dummy.trans_geo.variable_list.index(var_1)
					idx_2 = dummy.trans_geo.variable_list.index(var_2)
				except FileNotFoundError:
					cov = dummy.get_cov(var_2,var_1)
					idx_1 = dummy.trans_geo.variable_list.index(var_2)
					idx_2 = dummy.trans_geo.variable_list.index(var_1)
				cov.plot_cov = plot_cov
				cov.plot_cov(cov)
				plt.savefig(dummy.trans_geo.file_handler.out_file("cov_" + var_1 + "_" + var_2))
				plt.close()