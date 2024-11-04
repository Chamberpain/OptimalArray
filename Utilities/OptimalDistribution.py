import os
import matplotlib.pyplot as plt
from GeneralUtilities.Data.pickle_utilities import load,save
import pickle
import numpy as np
import statistics
from scipy.stats import binned_statistic
import itertools
from OptimalArray.Utilities.MakeRandom import make_filename, load_array
import plotly.graph_objects as go
real_depth_list = [15, 40, 87.5, 137.5, 225, 350, 550, 750, 950, 1150, 1350, 1625, 2250]
depth_list = [2,4,6,8,10,12,14,16,18,20,22,24,26] 

region_dict = { #used for plot titles
            'indian': 'Indian Ocean',
            'southern_ocean': 'Southern Ocean',
            'north_atlantic': 'North Atlantic',
            'tropical_atlantic': 'Tropical Atlantic',
            'south_atlantic': 'South Atlantic',
            'north_pacific': 'North Pacific',
            'ccs': 'California Current Stream',
            'tropical_pacific': 'Tropical Pacific',
            'south_pacific': 'South Pacific',
            'gom': 'Gulf of Mexico',
    }

#Represents one hypothetical float array in a region at a set depth, with a certain number of floats, and given percent of these floats that have ph, o2, and chl sensors
class IndividualExperiment():

    price_dict = { #cost of each BGC sensor
        "oxygen": 7000,
        "nitrate": 24000, #not included in data
        "ph": 10000,
        "chlorophyll": 17000, #biooptics sensor includes chlorophyll, backscatter, downwelling irradiance
        #chlorophyll sensor alone costs ~1/3 of this price, may need to adjust later for more accurate results
    }

    base_cost = 22000 #cost of Core Argo float (float body, temperature and salinity sensors)

    #Initialize an individual experiment (one data point with monte carlo iterations averaged together)
    def __init__(self, region, ph, o2, po4, float_num, depth=2, chl=None):
        self.region, self.ph, self.o2, self.po4, self.chl, self.float_num, self.depth = region, ph, o2, chl, po4, float_num, depth
        self.filenames = self.get_filenames()
        self.cost = self.calculate_cost()
        self.unconstrained_variance_dict = self.calculate_unconstrained_variance()

    @classmethod
    def get_self_filepath(cls, region, ph, po4, float_num, depth=2, chl=None):
        if depth > 8:
            label = 'cm4/'+region+'/'+'ph'+str(ph)+'_o2'+str(o2)+'_po4'+str(po4)+'_num'+str(float_num)
            filepath = make_filename('instrument',label,depth,'class')
        else: 
            label = 'cm4/'+region+'/'+'ph'+str(ph)+'_o2'+str(o2)+'_po4'+str(po4)+'_chl'+str(chl)+'_num'+str(float_num)
            filepath = make_filename('instrument',label,depth,'class')            
        return filepath

    def return_var(self,variable):
            if variable == 'ph':
                return self.ph
            elif variable == 'o2':
                return self.o2
            elif variable == 'po4':
                return self.po4
            elif variable == 'chl':
                return self.chl

    def get_filenames(self):
        label = 'cm4/'+self.region+'/'+'ph'+str(self.ph)+'_o2'+str(self.o2)+'_po4'+str(self.po4)+'_num'+str(self.float_num)
        filepath = make_filename('instrument',label,self.depth,1)
        if os.path.exists(filepath):
            return [make_filename('instrument',label,self.depth,x) for x in range(10)]
        else: 
            label = 'cm4/'+self.region+'/'+'ph'+str(self.ph)+'_o2'+str(self.o2)+'_po4'+str(self.po4)+'_chl'+str(self.chl)+'_num'+str(self.float_num)
            return [make_filename('instrument',label,self.depth,x) for x in range(10)]

    def calculate_cost(self):
        base_expense = self.float_num * self.base_cost
        o2_expense = self.o2 * self.float_num * self.price_dict["oxygen"]
        ph_expense = self.ph * self.float_num * self.price_dict["ph"]
        po4_expense = self.po4 * self.float_num * self.price_dict["nitrate"]

        if self.chl:
            chl_expense = self.chl * self.float_num * self.price_dict["chlorophyll"]
        else:
            chl_expense=0
        total_cost = base_expense + o2_expense + ph_expense + chl_expense + po4_expense
        return total_cost

    def calculate_unconstrained_variance(self):
        data_list = []
        for filepath in self.filenames:
            print(filepath)
            H_index, p_hat_diag = load(filepath) #from general utilities.data.pickle utilities
            data_list.append(np.sum(p_hat_diag))
        data_dict = dict(zip(range(len(data_list)),data_list))
        return data_dict

    def return_monte_carlo_avg(self):
        return np.array(list(self.unconstrained_variance_dict.values())).mean()

class AllDepthsIndividualExperiment(IndividualExperiment):
    def __init__(self, region, ph, o2, float_num, chl=None, depth=None):
        self.region, self.ph, self.o2, self.chl, self.po4, self.float_num = region, ph, o2, chl, float_num
        self.filenames = self.get_filenames()
        self.cost = self.calculate_cost()
        self.unconstrained_variance_dict = self.calculate_unconstrained_variance()

    @classmethod
    def get_self_filepath(cls, region, ph, o2, po4, float_num, chl=None,depth=None):
        label = 'cm4/'+region+'/'+'ph'+str(ph)+'_o2'+str(o2)+'_po4'+str(po4)+'_chl'+str(chl)+'_num'+str(float_num)
        filepath = make_filename('instrument',label,'','alldepthclass')            
        return filepath

    def get_filenames(self):
        filenames = []
        for depth in depth_list:
            label = 'cm4/'+self.region+'/'+'ph'+str(self.ph)+'_o2'+str(self.o2)+'_po4'+str(po4)+'_num'+str(self.float_num)
            filepath = make_filename('instrument',label,depth,1)
            if os.path.exists(filepath):
                filenames.append([make_filename('instrument',label,depth,x) for x in range(10)])
            else: 
                label = 'cm4/'+self.region+'/'+'ph'+str(self.ph)+'_o2'+str(self.o2)+'_po4'+str(po4)+'_chl'+str(self.chl)+'_num'+str(self.float_num)
                filenames.append([make_filename('instrument',label,depth,x) for x in range(10)])
        return filenames

    def calculate_unconstrained_variance(self):
        data_list = []
        for filepath_list in self.filenames:
            data_holder = []
            for filepath in filepath_list:
                print(filepath)
                H_index, p_hat_diag = load(filepath) #from general utilities.data.pickle utilities
                data_holder.append(np.sum(p_hat_diag))
            data_list.append(np.mean(data_holder))
        data_dict = dict(zip(depth_list,data_list))
        return data_dict

    def return_monte_carlo_avg(self):
        return np.array(list(self.unconstrained_variance_dict.values())).sum()


#Given a region, depth, and variable, load in and plot all individual experiments (hypothetical float arrays)
class OmniBus():

    #Initialize a set of experiments for a given region, depth, and variable to focus on
    def __init__(self, region, depth=2,ExClass = IndividualExperiment):
        self.region = region
        self.depth = depth
        self.ex_list = []
        percent_list = [0,0.2,0.4,0.6,0.8,1]
        for float_num in range(0,80,10):
            for ph_percent in percent_list:
                for o2_percent in percent_list:
                    for po4_percent in percent_list
                        if self.depth > 8:
                            class_filepath = ExClass.get_self_filepath(self.region,ph_percent, o2_percent,po4_percent, float_num, depth=depth)
                            if os.path.isfile(class_filepath):
                                self.ex_list.append(load(class_filepath))
                            else:
                                ex = ExClass(self.region, ph_percent, o2_percent,po4_percent, float_num, depth=depth)
                                save(class_filepath,ex)
                                self.ex_list.append(ex)

                        else:
                            for chl_percent in percent_list:
                                class_filepath = ExClass.get_self_filepath(self.region,ph_percent, o2_percent,po4_percent, float_num, depth=depth,chl=chl_percent)
                                if os.path.isfile(class_filepath):
                                    self.ex_list.append(load(class_filepath))
                                else:
                                    ex = ExClass(self.region, ph_percent, o2_percent,po4_percent, float_num, depth=depth,chl=chl_percent)
                                    save(class_filepath,ex)
                                    self.ex_list.append(ex)
                
    def plot(self,variable):
        cost_list = [] #x-axis values
        unconstrained_variance_list = [] #y-axis values
        desired_variable_list = [] #color values
        size_list = [] #size of point values (corresponding to float_num)
        for dummy in self.ex_list:
            cost_list.append(dummy.cost)
            unconstrained_variance_list.append(dummy.return_monte_carlo_avg())
            if variable == 'ph':
                desired_variable_list.append(dummy.ph)
            elif variable == 'o2':
                desired_variable_list.append(dummy.o2)
            elif variable == 'po4':
                desired_variable_list.append(dummy.po4)
            elif variable == 'chl':
                desired_variable_list.append(dummy.chl)
            size_list.append(dummy.float_num)
        plt.scatter(cost_list, unconstrained_variance_list, s=size_list, c=desired_variable_list, cmap='viridis')
        plt.title("Cost vs. Unconstrained Variance (" + region_dict[self.region] + ", depth = " + str(self.depth) + ", variable = " + variable + ")")
        plt.xlabel("Cost")
        plt.ylabel("Unconstrained Variance")
        plt.colorbar()
        plt.show()
        plt.close()
        #Use following line to save the plot (instead of showing):
        #plt.savefig(self.region + '_' + str(self.desired_depth) + '_' + self.desired_variable + '.png', bbox_inches='tight')

    #Find and return all valuable data regarding the pareto frontier 
    def return_cost_list(self):
        return [dummy.cost for dummy in self.ex_list]

    def return_variance_list(self):
        if self.depth>8:
            return [dummy.return_monte_carlo_avg() for dummy in self.ex_list]*6
        else:
            return [dummy.return_monte_carlo_avg() for dummy in self.ex_list]

    def return_variable_list(self,variable):
        if variable == 'ph':
            return [dummy.ph for dummy in self.ex_list]
        elif variable == 'o2':
            return [dummy.o2 for dummy in self.ex_list]
        elif variable == 'chl':
            return [dummy.chl for dummy in self.ex_list]

    def return_float_num_list(self):
        return [dummy.float_num for dummy in self.ex_list]

    def return_pareto_experiments(self,type='min'):
        cost_list, unconstrained_variance_list, desired_variable_list, bins_edges, x_list, min_y_values, color_list, pareto_size_list,return_ex_list = self.pareto_engine(8,'o2')
        return return_ex_list

    def pareto_engine(self, num_bins,variable,type='min'):
        cost_list = self.return_cost_list() #all x-axis values (not just pareto frontier values)
        unconstrained_variance_list = self.return_variance_list() #all y-axis values (not just pareto frontier values)
        desired_variable_list = self.return_variable_list(variable) #all color values (not just pareto frontier values)
        size_list = self.return_float_num_list() #all size values (not just pareto frontier values)

        #Bin the data based on num_bins parameter:
        min_y_values, bins_edges, bin_numbers = binned_statistic(cost_list, unconstrained_variance_list, type, num_bins)
        x_list = [] #x-axis of values along the pareto frontier
        color_list = [] #color values along the pareto frontier
        pareto_size_list = [] #size values along the pareto frontier
        counter = 0 #variable used in deletion of bins that don't contain a value (if user enters a num_bins value that is too high)
        return_ex_list = []
        for y_val in min_y_values: #min_y_values are y-axis values along pareto frontier
            if np.isnan(y_val): #if bin doesn't have a value, skip
                min_y_values = np.delete(min_y_values, counter) #delete "not a number" value from array
                continue
            index = unconstrained_variance_list.index(y_val) #find corresponding x, color, and size values
            x_list.append(cost_list[index])
            color_list.append(desired_variable_list[index])
            pareto_size_list.append(size_list[index])
            return_ex_list.append(self.ex_list[index])
            counter += 1
        return cost_list, unconstrained_variance_list, desired_variable_list, bins_edges, x_list, min_y_values, color_list, pareto_size_list,return_ex_list
    
    #Plot the pareto frontier, default frontier contains 40 bins (and therefore 40 points) but can be changed with a different argument
    def plot_individual_pareto(self, variable, num_bins = 8):
        cost_list, unconstrained_variance_list, desired_variable_list, bin_edges, x_list, min_y_values, color_list, pareto_size_list,return_ex_list = self.pareto_engine(num_bins,variable)
        #If desired, amplify magnitude of size_list here (to make differences in float_num more apparent between points)
        plt.scatter(cost_list, unconstrained_variance_list, c=desired_variable_list, cmap='viridis',alpha=0.2) #all data in greyscale to see spread
        plt.scatter(x_list, min_y_values, s=pareto_size_list, c=color_list, cmap='viridis',zorder=10) #pareto data in color
        # plt.title("Pareto Frontier (" + region_dict[self.region] + ", depth = " + str(self.depth) + ", variable = " + variable + ")")
        plt.xlabel("Cost ($)")
        plt.ylabel("Unconstrained Variance")
        plt.colorbar()
        plt.show()
        plt.close()
        #Use following line to save the plot (instead of showing):
        #plt.savefig(self.region + '_' + str(self.desired_depth) + '_' + self.desired_variable + '.png', bbox_inches='tight')

#Subclasses of OmniBus by region
class OmniBusIndian(OmniBus):
    region = 'indian' #has to match up with region as shown in file names of data (excluding underscore)
    def __init__(self, *args,**kwargs):
        super().__init__(self.region, *args,**kwargs)

class OmniBusSO(OmniBus):
    region = 'southern_ocean'
    def __init__(self, *args,**kwargs):
        super().__init__(self.region, *args,**kwargs)

class OmniBusNAtlantic(OmniBus):
    region = 'north_atlantic'
    def __init__(self, *args,**kwargs):
        super().__init__(self.region, *args,**kwargs)

class OmniBusTropicalAtlantic(OmniBus):
    region = 'tropical_atlantic'
    def __init__(self, *args,**kwargs):
        super().__init__(self.region, *args,**kwargs)

class OmniBusSAtlantic(OmniBus):
    region = 'south_atlantic'
    def __init__(self, *args,**kwargs):
        super().__init__(self.region, *args,**kwargs)

class OmniBusNPacific(OmniBus):
    region = 'north_pacific'
    def __init__(self, *args,**kwargs):
        super().__init__(self.region, *args,**kwargs)

class OmniBusCCS(OmniBus):
    region = 'ccs'
    def __init__(self, *args,**kwargs):
        super().__init__(self.region, *args,**kwargs)

class OmniBusTropicalPacific(OmniBus):
    region = 'tropical_pacific'
    def __init__(self, *args,**kwargs):
        super().__init__(self.region, *args,**kwargs)

class OmniBusSPacific(OmniBus):
    region = 'south_pacific'
    def __init__(self, *args,**kwargs):
        super().__init__(self.region, *args,**kwargs)

class OmniBusGOM(OmniBus):
    region = 'gom'
    def __init__(self, *args,**kwargs):
        super().__init__(self.region, *args,**kwargs)

#Create plots for every region for every depth level for every variable
def plot_all():
    bus_list = []
    for bus in [OmniBusIndian, OmniBusSO, OmniBusNAtlantic, OmniBusTropicalAtlantic, OmniBusSAtlantic, OmniBusNPacific]:
        dummy = bus(ExClass = AllDepthsIndividualExperiment)
        bus_list.append(dummy)
            # if depth > 8: #ignore chl above depth index 8
            #     for variable in ['ph', 'o2']:
            #         dummy = bus(depth, variable)
            #         dummy.plot()
            # else:
            #     for variable in ['ph', 'o2', 'chl']:
            #         dummy = bus(depth, variable)
            #         dummy.plot()
def figure_3():
    ie = IndividualExperiment('north_atlantic',0.4,0.8,40,2,chl=1)
    fig_2_variance = ie.return_monte_carlo_avg()
    fig_2_cost = ie.cost

    bus = OmniBusNAtlantic(2)
    cost_list, unconstrained_variance_list, desired_variable_list, bin_edges, x_list, min_y_values, color_list, pareto_size_list = bus.pareto_engine(40,'o2')
    #If desired, amplify magnitude of size_list here (to make differences in float_num more apparent between points)
    plt.scatter(cost_list, unconstrained_variance_list, c=[x*100 for x in desired_variable_list], cmap='viridis',alpha=0.2) #all data in greyscale to see spread
    plt.scatter(x_list, min_y_values, s=pareto_size_list, c=[x*100 for x in color_list], cmap='viridis',zorder=10) #pareto data in color
    # plt.title("Pareto Frontier (" + region_dict[self.region] + ", depth = " + str(self.depth) + ", variable = " + variable + ")")
    plt.annotate('Fig. 2 Example',color='red',xy=(fig_2_cost,fig_2_variance),xytext=(fig_2_cost+500000,fig_2_variance+2000),
            arrowprops=dict(arrowstyle='->', color='red'))
    plt.xlabel("Cost ($)")
    plt.ylabel("Unconstrained Variance")
    plt.colorbar(label = 'Oxygen Sensor Distribution (%)')                    

class GlobalClass():
    def __init__(self):
        bus_list = []
        for bus in [OmniBusIndian, OmniBusSO, OmniBusNAtlantic, OmniBusTropicalAtlantic, OmniBusSAtlantic, OmniBusNPacific]:
            dummy = bus(ExClass = AllDepthsIndividualExperiment)
            bus_list.append(dummy)
        pareto_lists = [dummy.return_pareto_experiments() for dummy in bus_list]
        combined_list = list(itertools.product(*pareto_lists))
        uncon_var_list = []
        cost_list = []
        var_list = []
        var = 'o2'
        num_list = []
        for holder in combined_list:
            uncon_var_list.append(sum([dummy.return_monte_carlo_avg() for dummy in holder]))
            cost_list.append(sum([dummy.cost for dummy in holder]))
            var_list.append(np.mean([dummy.return_var(var) for dummy in holder]))
            num_list.append(np.sum([dummy.float_num for dummy in holder]))

        plt.scatter(cost_list,uncon_var_list,c=num_list)
        plt.colorbar(label='num')
        plt.show()


        type = 'min'
        min_y_values, bins_edges, bin_numbers = binned_statistic(cost_list, uncon_var_list, type, 8)
        df_list = []
        y_value = min_y_values[3]
        idx = uncon_var_list.index(y_value)
        cost = cost_list[idx]
        ex_list = combined_list[idx]
        print(type)
        print('bin idx is ',3)
        print('cost is ',cost)
        region_list = []
        TS_list = []
        o2_list = []
        ph_list = []
        chl_list = []
        for ex in ex_list:
            region_list.append(region_dict[ex.region])
            TS_list.append(ex.float_num)
            o2_list.append(ex.float_num*ex.o2)
            ph_list.append(ex.float_num*ex.ph)
            chl_list.append(ex.float_num*ex.chl)
            print('region is ',ex.region)
            print('float_num is ',ex.float_num)
            print('o2 level is ',ex.o2)
            print('ph level is ',ex.ph)
            print('chl level is ',ex.chl)
        df = pd.DataFrame({'region':region_list,'floats':TS_list,
            'o2':o2_list,'ph':ph_list,'chl':chl_list})
        fig = go.Figure(
            data=[
                go.Bar(x=df.region, y=df.floats, name="Total Floats"),
                go.Bar(x=df.region, y=df.o2, name="Oxygen"),
                go.Bar(x=df.region, y=df.ph, name="pH"),
                go.Bar(x=df.region, y=df.chl, name="Chlorophyll"),
            ],
            layout=dict(
                barcornerradius=15,
            ),
)



#One level above OmniBus, used to load in and plot OmniBus objects for every depth level given a variable
class SuperClass():
    
    def __init__(self, OmniBusClass):
        self.OmniBusClass = OmniBusClass
        self.OmniBus_list = []
        for depth in depth_list:
            self.OmniBus_list.append(OmniBusClass(depth=depth))

    def all_depths_pareto_engine(self, num_bins,variable,type='min'):
        cost_list = self.OmniBus_list[0].return_cost_list() #all x-axis values (not just pareto frontier values)
        unconstrained_variance_list = [dummy.return_variance_list() for dummy in self.OmniBus_list] #all y-axis values (not just pareto frontier values)
        unconstrained_variance_list = list(np.sum(unconstrained_variance_list,axis=0))
        desired_variable_list = self.OmniBus_list[0].return_variable_list(variable) #all color values (not just pareto frontier values)
        size_list = self.OmniBus_list[0].return_float_num_list() #all size values (not just pareto frontier values)

        #Bin the data based on num_bins parameter:
        min_y_values, bins_edges, bin_numbers = binned_statistic(cost_list, unconstrained_variance_list, type, num_bins)
        x_list = [] #x-axis of values along the pareto frontier
        color_list = [] #color values along the pareto frontier
        pareto_size_list = [] #size values along the pareto frontier
        class_list = []
        counter = 0 #variable used in deletion of bins that don't contain a value (if user enters a num_bins value that is too high)
        for y_val in min_y_values: #min_y_values are y-axis values along pareto frontier
            if np.isnan(y_val): #if bin doesn't have a value, skip
                min_y_values = np.delete(min_y_values, counter) #delete "not a number" value from array
                continue
            index = unconstrained_variance_list.index(y_val) #find corresponding x, color, and size values
            x_list.append(cost_list[index])
            color_list.append(desired_variable_list[index])
            pareto_size_list.append(size_list[index])
            class_list.append(self.)
            counter += 1
        return cost_list, unconstrained_variance_list, desired_variable_list, bins_edges, x_list, min_y_values, color_list, pareto_size_list
    

    #Plot all pareto frontiers for a given region and variable at every depth level
    def plot_2d_pareto(self, variable, num_bins = 40):
        plot_values = []
        for bus in self.OmniBus_list:
            cost_list, unconstrained_variance_list, desired_variable_list, bin_edges, x_list, min_y_values, color_list, pareto_size_list = bus.pareto_engine(num_bins,variable)
            plot_values.append(np.array(color_list[:38]))
        plot_values = np.stack(plot_values)
        XX,YY = np.meshgrid(x_list,real_depth_list)
        plt.pcolormesh(XX,YY,plot_values*100)
        plt.title("Pareto Frontier (" + region_dict[self.OmniBusClass.region] + ", " + self.variable + ")")
        plt.xlabel("Cost ($)")
        plt.ylabel("Depth")
        plt.colorbar(label=variable+' sensor distribution (%)')
        plt.gca().invert_yaxis()
        plt.yscale('log')
        plt.show()
        plt.close()

    def plot_individual_pareto(self, variable, num_bins = 40,type='min'):
        cost_list, unconstrained_variance_list, desired_variable_list, bin_edges, x_list, min_y_values, color_list, pareto_size_list = self.all_depths_pareto_engine(num_bins,variable,type=type)
        #If desired, amplify magnitude of size_list here (to make differences in float_num more apparent between points)
        plt.scatter(cost_list, unconstrained_variance_list, c=desired_variable_list, cmap='viridis',alpha=0.2) #all data in greyscale to see spread
        plt.scatter(x_list, min_y_values, s=pareto_size_list, c=color_list, cmap='viridis',zorder=10) #pareto data in color
        # plt.title("Pareto Frontier (" + region_dict[self.region] + ", depth = " + str(self.depth) + ", variable = " + variable + ")")
        plt.xlabel("Cost ($)")
        plt.ylabel("Unconstrained Variance")
        plt.colorbar()
        plt.show()

def figure_4():
    bus_holder = SuperClass(OmniBusNAtlantic)
    plot_values = []
    for bus in bus_holder.OmniBus_list:
        cost_list, unconstrained_variance_list, desired_variable_list, bin_edges, x_list, min_y_values, color_list, pareto_size_list = bus.pareto_engine(num_bins,variable)
        plot_values.append(np.array(color_list[:38]))
    plot_values = np.stack(plot_values)
    XX,YY = np.meshgrid(x_list,real_depth_list)
    plt.pcolormesh(XX,YY,plot_values*100)
    plt.xlabel("Cost ($)")
    plt.ylabel("Depth (m)")
    plt.colorbar(label='Oxygen Sensor Distribution (%)')
    plt.gca().invert_yaxis()
    plt.yscale('log')
    plt.show() 