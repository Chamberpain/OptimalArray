from GeneralUtilities.Data.Mapped.cm4 import CM4DIC,CM4O2,CM4PO4,CM4ThetaO,CM4Sal,CM4PH,CM4CHL,CM4PC02
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
from OptimalArray.Utilities.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
plot_handler = FilePathHandler(ROOT_DIR,'FutureArgoMixedLayer/plots')


for data_class in CM4DIC,CM4O2,CM4PO4,CM4ThetaO,CM4Sal,CM4PH,CM4CHL:
	data, lons  = data_class().return_int()
	lons, lats = data_class().return_dimensions()
	subsampled_data = data[:,::10,::10]
	sub_lats = lats[::10]
	sub_lons = lons[::10]
	for y,lat in enumerate(sub_lats):
		for x, lon in enumerate(sub_lons):
			time_series = subsampled_data[:,y,x]
			if (time_series==0).all():
				continue
			N = len(time_series)
			T = 1
			x = np.linspace(0.0, N*T, N)
			xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
			yf = scipy.fftpack.fft(time_series-time_series.mean())
			_ = plt.plot(xf, 2.0/N * np.abs(yf[:N//2]),alpha=0.2)
			tick_locations = [1/(100*12),1/(50*12),1/(20*12),1/(10*12),1/(2*12),1/(1*12),1/(9),1/(6),1/(3)]
			tick_labels = ['100 Years','50 Years','20 Years','10 Years','2 Years','1 Year','9 Months','6 Months','3 Months']
			plt.xscale('log')
			plt.xticks(ticks=tick_locations, labels=tick_labels, rotation=45, fontsize=10)
	plt.title(data_class.variable+' Spectrum')
	plt.savefig(plot_handler.store_file(data_class.variable+'_spectrum'))
	plt.close()