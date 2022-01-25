from transition_matrix.makeplots.transition_matrix_plot import TransPlot
from transition_matrix.makeplots.plot_utils import basemap_setup,transition_vector_to_plottable
# from transition_matrix.makeplots.argo_data import SOCCOM
import numpy as np
from transition_matrix.makeplots.inversion.target_load import CM2p6Correlation,CM2p6VectorSpatialGradient,CM2p6VectorTemporalVariance,CM2p6VectorMean,GlodapCorrelation,GlodapVector
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
from scipy import interpolate
import random

class InversionPlot():
    def __init__(self, base_class = None, target_vector_class = None, corr_matrix_class = None, float_class = None):
        self.target = target_vector_class
        self.cor = corr_matrix_class
        self.float = float_class
        self.base = base_class

    @classmethod 
    def cm2p6(cls,variable,variance):
        variance_class_dict = {'spatial':CM2p6VectorSpatialGradient,'time':CM2p6VectorTemporalVariance,'mean':CM2p6VectorMean}
        lat,lon = (2,2)
        traj = CM2p6Correlation.traj_type_gen(variable)
        corr_matrix_class = CM2p6Correlation.load_from_type(traj_type=traj,lat_spacing=lat,lon_spacing=lon,time_step=60)
        traj = variance_class_dict[variance].traj_type_gen(variable)
        target_vector_class = variance_class_dict[variance].load_from_type(traj_type=traj,lat_spacing=lat,lon_spacing=lon,time_step=60)
        base = TransPlot.load_from_type(traj_type='argo',lat_spacing=lat,lon_spacing=lon,time_step=60)
        return cls(base_class = base,target_vector_class=target_vector_class,corr_matrix_class=corr_matrix_class)

    def cov_from_cor(cls,cor,target):
        XX,YY = np.meshgrid(target.data.tolist(),target.data.tolist())
        normalized_var = np.sqrt((XX.ravel()*YY.ravel()))
        var = normalized_var.reshape(XX.shape)
        cov = scipy.sparse.csc_matrix(np.multiply(cor.todense(),var))
        idxs = np.where(cor.diagonal()==0)[0]
        for idx in idxs:
            cov[idx,idx]=target[idx]
        return scipy.sparse.csc_matrix(np.multiply(cor.todense(),var))

    # def get_optimal_float_locations(self,float_number=1000,vector_exploration_factor=0,corr_exploration_factor=0):
    #     """ accepts a target vecotr in sparse matrix form, returns the ideal float locations and weights to sample that"""
    #     # vector = self.normalize_vector(vector,vector_exploration_factor)
    #     # vector = self.remove_float_observations(vector)
    #     if corr_exploration_factor:
    #         self.normalize_correlation(corr_exploration_factor)
    #     print 'I am starting the optimization'
    #     optimize_fun = scipy.optimize.lsq_linear(self.cor.dot(self.base),np.ravel(self.target.data),bounds=(0,1.2),verbose=2,max_iter=20)
    #     desired_vector = optimize_fun.x
    #     print 'It is optimized'
    #     y,x = zip(*[x for _,x in sorted(zip(desired_vector.tolist(),self.base.total_list.tolist()))[::-1][:float_number]])
    #     bins_lat,bins_lon = self.base.bins_generator(base.degree_bins)
    #     XX,YY,m = basemap_setup(bins_lat,bins_lon,base.traj_file_type) 
    #     m.scatter(x,y,marker='*',color='g',s=34,latlon=True)
    #     return m


    # def alt_get_optimal_float_locations(self,noise):
    #     for noise in [0.0001,0.001,0.01,0.1,1,10,100]:
    #         denom = self.base.dot(self.cor.dot(self.base.T))+scipy.sparse.diags([noise]*self.base.shape[0])
    #         inv_denom = scipy.linalg.inv(np.array(denom.todense()))
    #         num = self.cor.dot(self.base.T)
    #         output = num.todense().dot(inv_denom).dot(self.target.todense())
    #         bins_lat,bins_lon = self.base.bins_generator(self.base.degree_bins)
    #         XX,YY,m = basemap_setup(bins_lat,bins_lon,self.base.traj_file_type) 
    #         out_plot = transition_vector_to_plottable(bins_lat,bins_lon,self.base.total_list,np.ravel(output))
    #         m.pcolormesh(XX,YY,np.log(out_plot),vmin=0,vmax=np.log(cov.max()))
    #         plt.show()

    def get_index_of_first_eigen_vector(self,p_hat):
        p_hat[p_hat<0]=0
        eigs = scipy.sparse.linalg.eigs(p_hat,k=1)
        e_vec = eigs[1][:,-1]
        idx = e_vec.tolist().index(e_vec.max())
        return idx,e_vec

    def generate_H(self,column_idx,N):
        data = [1]*len(column_idx)
        row_idx = range(len(column_idx))
        return scipy.sparse.csc_matrix((data,(row_idx,column_idx)),shape=(len(column_idx),N))

    def plot_p_hat(self,p_hat,output_list,title,cov,variance_scaled=False):
        bins_lat,bins_lon = self.base.bins_generator(self.base.degree_bins)

        data_to_plot = np.ravel(p_hat.diagonal())
        if variance_scaled:
            data_to_plot/= np.ravel(cov.diagonal())
        out_plot = transition_vector_to_plottable(bins_lat,bins_lon,self.base.total_list,data_to_plot)
        XX,YY,m = basemap_setup(bins_lat,bins_lon,self.base.traj_file_type) 
        try:
            y,x = zip(*self.base.total_list[output_list].tolist())
            m.scatter(x,y,marker='*',color='y',latlon=True,zorder=50)
        except ValueError:
            pass
        out_plot[out_plot<1]=1
        cm = self.target.cm_dict[self.target.base_variable_generator()]
        if variance_scaled:
            m.pcolormesh(XX,YY,out_plot,cmap=cm,vmin=0,vmax=1)
            plt.colorbar(label='Variance Scaled')            
        else:            
            m.pcolormesh(XX,YY,out_plot,cmap=cm,vmin=0,vmax=(cov.diagonal().std()*3 + cov.diagonal().mean()))
            plt.colorbar(label=self.target.unit)
        plt.savefig(title+str(len(output_list)))
        plt.close()

    def p_hat_calculate(self,output_list,cov,noise):
        H = self.generate_H(output_list,cov.shape[0])
        denom = H.dot(cov).dot(H.T)+scipy.sparse.diags([noise]*H.shape[0])
        inv_denom = scipy.sparse.linalg.inv(denom)
        if not type(inv_denom)==scipy.sparse.csc.csc_matrix:
            inv_denom = scipy.sparse.csc.csc_matrix(inv_denom)  # this is for the case of 1x1 which returns as array for some reason
        cov_subtract = cov.dot(H.T).dot(inv_denom).dot(H).dot(cov)
        p_hat = cov-cov_subtract
        return p_hat

    def alt_get_optimal_float_locations(self,n,noise=100):
        output_list = []
        out_data = []
        cov = self.cov_from_cor(self.cor,self.target)
        # self.plot_p_hat(cov,output_list,'')
        idx,e_vec = self.get_index_of_first_eigen_vector(cov)
        output_list.append(idx)
        out_data.append(e_vec) 
        for float_ in range(n):
            print float_
            p_hat = self.p_hat_calculate(output_list,cov,noise)
            # self.plot_p_hat(p_hat,output_list,'',cov)
            idx,e_vec = self.get_index_of_first_eigen_vector(p_hat)
            output_list.append(idx)
            out_data.append(e_vec)            
        return (p_hat,output_list,out_data)

    def random_float_locations(self,cov,noise,n):
        output_list = []
        out_data = []
        output_list = random.sample(range(self.base.shape[0]),n)
        p_hat = self.p_hat_calculate(output_list,cov,noise)
        # self.plot_p_hat(p_hat,output_list,'',cov)
        return (p_hat,output_list,out_data)

def calculate_p_hat_error(p_hat):
    diag = p_hat.diagonal()
    diag[diag<0]=0
    return np.linalg.norm(diag)

def ideal_float_plot():
    for variable in ['surf_dic','surf_pco2','surf_o2','100m_dic','100m_o2']:

        inv = InversionPlot.cm2p6(variable,'time')
        # try:
        #     ideal_array = np.load('ideal_'+variable+'.npy')
        # except IOError:
        p_hat,ideal_array,out_data = inv.alt_get_optimal_float_locations(1000)
        np.save('ideal_'+variable,ideal_array)    
        cov = inv.cov_from_cor(inv.cor,inv.target)
        p_hat_ideal = inv.p_hat_calculate(ideal_array,cov,100)
        inv.plot_p_hat(p_hat_ideal,ideal_array,(variable+' ideal_array'),cov)

def signal_to_noise_plot():
    for variable in ['surf_dic','surf_pco2','surf_o2','100m_dic','100m_o2']:

        inv = InversionPlot.cm2p6(variable,'time')
        try:
            ideal_array = np.load('ideal_'+variable+'.npy')
        except IOError:
            p_hat,ideal_array,out_data = inv.alt_get_optimal_float_locations(1000)
            np.save('ideal_'+variable,ideal_array)
        SNR_list = []
        for SNR in [10000,5000,1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01]:
            for _ in range(10):
                print 'SNR is ',SNR
                inv = InversionPlot.cm2p6('surf_pco2','time')
                inv.cor.return_signal_to_noise(SNR,10)
                cov = inv.cov_from_cor(inv.cor,inv.target)
                p_hat_random,random_array,dum1 = inv.random_float_locations(cov,100,1000)
                p_hat_ideal = inv.p_hat_calculate(ideal_array,cov,100)
                SNR_list.append((SNR,calculate_p_hat_error(p_hat_random),calculate_p_hat_error(p_hat_ideal)))

        snr,random_array,ideal_array = zip(*SNR_list)
        snr = np.array(snr)
        random_array = np.array(random_array)
        ideal_array = np.array(ideal_array)
        random_y1 = []
        random_y2 = []
        ideal_y1 = []
        ideal_y2 = []
        snr_list = []
        for dummy_snr in np.unique(snr):

            snr_list.append(dummy_snr)
            random_mean = random_array[snr==dummy_snr].mean()
            random_std = random_array[snr==dummy_snr].std()
            
            random_y1.append(random_mean-random_std)
            random_y2.append(random_mean+random_std)
            ideal_mean = ideal_array[snr==dummy_snr].mean()
            ideal_std = ideal_array[snr==dummy_snr].std()
            ideal_y1.append(ideal_mean-ideal_std)
            ideal_y2.append(ideal_mean+ideal_std)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.fill_between(snr_list, random_y1,random_y2, label='Uniform',alpha=0.2)
        ax.plot(snr_list,np.average(np.array([random_y2,random_y1]),axis=0))
        ax.fill_between(snr_list, ideal_y1, ideal_y2, label='Targeted',alpha=0.2)
        ax.plot(snr_list,np.average(np.array([ideal_y2,ideal_y1]),axis=0))
        plt.legend()
        plt.xlabel('SNR')
        plt.ylabel('Unobserved $L_2$ Normed Variance')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.title('Unobserved Surface '+variable+' Variance')
        plt.savefig('unobservd_'+variable)
        plt.close()


def increased_float_plot():
    for variable in ['surf_dic','surf_pco2','surf_o2','100m_dic','100m_o2']:

        inv = InversionPlot.cm2p6(variable,'time')
        try:
            ideal_array = np.load('ideal_'+variable+'.npy')
        except IOError:
            p_hat,ideal_array,out_data = inv.alt_get_optimal_float_locations(1000)
            np.save('ideal_'+variable,ideal_array)
        cov = inv.cov_from_cor(inv.cor,inv.target)
        p_hat_ideal = inv.p_hat_calculate(ideal_array,cov,100)
        ideal_error = calculate_p_hat_error(p_hat_ideal)

        random_error = []
        num_list = [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]
        for num in num_list:
            for _ in range(5):
                p_hat_random,random_array,dum1 = inv.random_float_locations(cov,100,num)
                random_error.append((calculate_p_hat_error(p_hat_random),num))

        error_list,float_number_list = zip(*random_error)
        plottable = np.array(error_list)/ideal_error
        random_y1 = []
        random_y2 = []
        x_axis = []
        for float_ in np.unique(float_number_list):

            x_axis.append(float_)
            random_mean = plottable[np.array(float_number_list==float_)].mean()
            random_std = plottable[np.array(float_number_list==float_)].std()

            random_y1.append(random_mean-random_std)
            random_y2.append(random_mean+random_std)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.fill_between(x_axis, random_y1,random_y2, label='Uniform',alpha=0.2)
        ax.plot(x_axis,np.average(np.array([random_y2,random_y1]),axis=0))
        plt.xlabel('Number of Deployed Floats')
        plt.ylabel('Scaled Variance')
        ax.set_xscale('log')
        plt.title('Scaled '+variable+' Variance')
        plt.savefig('unobservd_float'+variable)
        plt.close()
#  def normalize_vector(self,vector,exploration_factor):
# #this needs some extensive unit testing, the normalization is very ad hock
#         target_vector = vector-vector.min()
#         target_vector = target_vector/target_vector.max()+1
#         # target_vector = np.log(abs(target_vector))
#         target_vector = target_vector+exploration_factor*target_vector.mean()*np.random.random(target_vector.shape)
#         target_vector = target_vector/target_vector.max()
#         print 'I have normalized the target vector'
#         return target_vector




#     def instrument_to_observation(self,vector):
#         return (self.correlation.matrix.dot(self.transition_matrix)).dot(vector)

#     def remove_float_observations(self,vector):
#         float_result = self.instrument_to_observation(self.float_class.vector)
#         float_result = float_result/float_result.max()
#         vector = vector.flatten()-float_result
#         vector = np.array(vector)
#         vector[vector<0] = 0
#         print 'I have subtracted off the SOCCOM vector'        
#         return vector

#     def loc_plot(self,variance=False,floats=500,corr_exploration_factor=0,vector_exploration_factor=0):
#         x,y,desired_vector = self.get_optimal_float_locations(self.target.vector,float_number=floats,corr_exploration_factor=corr_exploration_factor,vector_exploration_factor=vector_exploration_factor)
#         bins_lat,bins_lon = base.bins_generator(base.degree_bins)
#         XX,YY,m = basemap_setup(bins_lat,bins_lon,base.traj_file_type)    
#         dummy,dummy,m = self.target.plot(XX=XX,YY=YY,m=m)
#         m.scatter(x,y,marker='*',color='g',s=34,latlon=True)
#         m = self.float_class.plot(m=m)
#         return (m,desired_vector)

#     def objective_map(self):
#         x,y,desired_vector = self.get_optimal_float_locations(self.target.vector,float_number=floats,corr_exploration_factor=corr_exploration_factor,vector_exploration_factor=vector_exploration_factor)
#         data_model = self.data_model_cov_mat(x,y)
#         data_data = self.data_data_cov_mat(x,y)
#         error = self.objective_error_compute(data_data,data_model)

#         XX,YY,m = basemap_setup(self.bins_lat,self.bins_lon,self.traj_file_type)
#         plot_vector = transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.list,error)
#         m.pcolormesh(XX,YY,plot_vector,vmax=1,vmin=0)
#         plt.colorbar()
#         m.scatter(x,y,marker='*',color='g',s=34,latlon=True)

#     def objective_error_compute(self,data_data,data_model):
#         compute = data_model.dot(np.linalg.inv(data_data).dot(data_model.T)).diagonal()
#         error = np.ones(compute.shape)-compute
#         return error

#     def data_model_cov_mat(self,x,y):
#         data_model= np.zeros([len(self.list),len(x)])
#         index_list = [list(_) for _ in zip(y,x)]
#         for k,_ in enumerate(index_list):
#             idx = self.list.index(_)
#             vector = self.correlation.matrix[:,idx].todense()
#             data_model[:,k] = np.array(vector).flatten()
#         return data_model

#     def data_data_cov_mat(self,x,y):        
#         index_list = []
#         for dummy in zip(y,x):
#             index_list.append(self.list.index(list(dummy)))
#         row,col = np.meshgrid(index_list,index_list)
#         data_data = self.correlation.matrix[row.flatten(),col.flatten()]
#         data_data = data_data.reshape(len(x),len(x))
#         data_data = np.array(data_data)
#         k = range(data_data.shape[0])
#         data_data[k,k]=1
#         return data_data

#     @classmethod 
#     def landschutzer(cls,transition_plot,var):
#         return cls(transition_plot,LandschutzerCO2Flux(var),LandschutzerCorr())

#     @classmethod 
#     def modis(cls,var):
#         return cls(MODISVector(var),MOIDSCorr())

#     @classmethod 
#     def glodap(cls,transition_plot,flux):
#         transition_plot.get_direction_matrix()
#         corr_matrix_class = GlodapCorrelation(transition_plot=transition_plot)
#         target_vector_class = GlodapVector(transition_plot=transition_plot,flux=flux)
#         float_class = SOCCOM(transition_plot=transition_plot)
#         return cls(correlation_matrix_class=corr_matrix_class,target_vector_class=target_vector_class,float_class=float_class)


# def individual_cruise_plot():
#     trans_plot = TransitionPlot()
#     trans_plot.get_direction_matrix()
#     factor_list = [#0,2,10,
#     50]
#     for corr_factor in factor_list:
#         for variance in [#'spatial',
#         'time',
#         #'mean'
#         ]:
#             for n,(variable,name) in enumerate([('100m_o2','$o_2$'),('100m_dic','dic')]):
#                 plt.subplot(3,1,(n+1))
#                 ip = InversionPlot.cm2p6(transition_plot=trans_plot,variable=variable,variance=variance)

#     #             filename = '../../plots/cm2p6_'+variable+'_'+variance
#                 m, desired_vector = ip.loc_plot(corr_exploration_factor=corr_factor)
#                 ZZ = transition_vector_to_plottable(trans_plot.bins_lat,trans_plot.bins_lon,trans_plot.list,desired_vector)  
#                 f = interpolate.interp2d(trans_plot.bins_lon, trans_plot.bins_lat, ZZ, kind='cubic')
#                 a13p5_start = (-54,-3)
#                 a13p5_end = (5,1)
#                 m.plot(zip(a13p5_start,a13p5_end)[1],zip(a13p5_start,a13p5_end)[0],latlon=True,linewidth=7)
#                 xnew = np.linspace(a13p5_start[1],a13p5_end[1],100)
#                 ynew = np.linspace(a13p5_start[0],a13p5_end[0],100)
#                 znew = [f(xnew[_], ynew[_])[0] for _ in range(len(xnew))]
#                 dist = np.sqrt((xnew-xnew[0])**2+(ynew-ynew[0])**2)*111
#                 plt.subplot(3,1,3)
#                 plt.plot(dist,znew,label=name)
#                 plt.xlabel('Distance Along Trackline')
#                 plt.ylabel('Relative Variance Constrained')
#             plt.legend()
#             plt.show()

