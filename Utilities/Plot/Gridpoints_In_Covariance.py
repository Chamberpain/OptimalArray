from OptimalArray.Utilities.CM4Mat import CovCM4Global
import numpy as np 
import matplotlib.pyplot as plt

depth = 2
cov_holder = CovCM4Global.load(depth_idx = depth)
data = np.split(cov_holder.cov.todense(),len(cov_holder.trans_geo.variable_list))[0]
data = np.split(data,len(cov_holder.trans_geo.variable_list),axis=1)[0]

data = np.array((data != 0).sum(axis=0)).ravel()
plottable = cov_holder.trans_geo.transition_vector_to_plottable(data)
XX,YY,ax = cov_holder.trans_geo.plot_setup()
pcm = ax.pcolormesh(XX,YY,plottable)
plt.colorbar(pcm,label='gridpoints in covariance')
plt.show()