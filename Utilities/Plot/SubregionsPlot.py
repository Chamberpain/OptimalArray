import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from OptimalArray.Utilities.CorGeo import InverseGlobal,InverseIndian,InverseSO,InverseNAtlantic,InverseTropicalAtlantic,InverseSAtlantic,InverseNPacific,InverseTropicalPacific,InverseSPacific,InverseGOM,InverseCCS
from GeneralUtilities.Plot.Cartopy.eulerian_plot import BaseCartopy

class RobinsonCartopy(BaseCartopy):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,ax = plt.axes(projection=ccrs.Robinson(central_longitude=180)),**kwargs)      
		print('I am plotting global region')
		self.finish_map()

# fig = plt.figure(figsize=(18,12))
# ax = fig.add_subplot(1,1,1)

def make_plot():
	XX, YY, ax = RobinsonCartopy().get_map()
	transform = ccrs.PlateCarree()._as_mpl_transform(ax)
	for geo in [InverseIndian,InverseSO,InverseNAtlantic,InverseTropicalAtlantic,InverseSAtlantic,InverseNPacific,InverseTropicalPacific,InverseSPacific,InverseGOM,InverseCCS]:
		geo = geo()
		for k,shape in enumerate(geo.ocean_shape):
			ax.add_geometries([shape], crs=ccrs.PlateCarree(), facecolor=geo.facecolor,edgecolor='black',alpha=0.5)
			if k ==0:
				ax.annotate(geo.facename,xy=(shape.centroid.x,shape.centroid.y),fontsize=12,xycoords=transform, zorder=12,bbox=dict(boxstyle="round,pad=0.3", fc=geo.facecolor,alpha=0.75))