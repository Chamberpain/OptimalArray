from transition_matrix.makeplots.transition_matrix_plot import TransPlot
from transition_matrix.makeplots.plot_utils import basemap_setup,transition_vector_to_plottable
import numpy as np
import matplotlib.pyplot as plt
import scipy
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