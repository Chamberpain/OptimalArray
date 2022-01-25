from transition_matrix.makeplots.inversion.target_load import CM2p6VectorSpatialGradient,CM2p6VectorTemporalVariance,CM2p6VectorMean,InverseBase
lat_lon_spacing_list = [(2,2)]
variable_list = ['surf_o2','surf_pco2','surf_dic','100m_o2','100m_dic']
time_step = 60
for lat,lon in lat_lon_spacing_list[::-1]:
	base = InverseBase.load_from_type(traj_type='argo',lat_spacing=lat,lon_spacing=lon,time_step=60)
	for variable in variable_list:
		for target_class in [CM2p6VectorSpatialGradient,CM2p6VectorTemporalVariance,CM2p6VectorMean]:
			print variable
			target = target_class.compile(variable,lat,lon,time_step)
			target.save()