from transition_matrix.makeplots.inversion.target_load import CM2p6Correlation

lat_lon_spacing_list = [(1,2),(2,2),(2,3),(4,4),(4,6)]
variable_list = ['surf_o2','surf_pco2','surf_dic','100m_o2','100m_dic']

for lat,lon in lat_lon_spacing_list:
	base = CM2p6Correlation.load_from_type(traj_type='argo',lat_spacing=lat,lon_spacing=lon,time_step=60)
	for variable in variable_list:
		corr_class = CM2p6Correlation.compile_corr(variable,base)
		corr_class.save()