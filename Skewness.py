# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 21:39:09 2021
@author: rayan & ewan
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import skew
from scipy.stats import kurtosis
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from scipy.stats import skewnorm
from scipy.interpolate import InterpolatedUnivariateSpline
# Returns the mean energy and variance based on Ballabio (Code from Aiden)
# Tion in keV
def DTprimspecmoments(Tion):
# Mean calculation
    a1 = 5.30509
    a2 = 2.4736e-3
    a3 = 1.84
    a4 = 1.3818
      
    mean_shift = a1*Tion**(0.6666666666)/(1.0+a2*Tion**a3)+a4*Tion
    
    # keV to MeV
    mean_shift /= 1e3
    
    mean = 14.021 + mean_shift
    
    # Variance calculation
    omega0 = 177.259
    a1 = 5.1068e-4
    a2 = 7.6223e-3
    a3 = 1.78
    a4 = 8.7691e-5
    
    delta = a1*Tion**(0.6666666666)/(1.0+a2*Tion**a3)+a4*Tion
    
    C = omega0*(1+delta)
    FWHM2 = C**2*Tion
    variance = FWHM2/(2.35482)**2
    # keV^2 to MeV^2
    variance /= 1e6
    
    return mean, variance #note: this is returned in MeV!!!

def DDprimspecmoments(Tion):
# Mean calculation
    a1 = 4.69515
    a2 = -0.040729
    a3 = 0.4
    a4 = 1.3818
      
      
    mean_shift = a1*Tion**(0.6666666666)/(1.0+a2*Tion**a3)+a4*Tion
    
    # keV to MeV
    mean_shift /= 1e3
    
    mean = 2.4495 + mean_shift
    
    # Variance calculation
    omega0 = 82.542
    a1 = 1.7013e-3
    a2 = 0.16888
    a3 = 0.49
    a4 = 7.946e-4
    
    delta = a1*Tion**(0.6666666666)/(1.0+a2*Tion**a3)+a4*Tion
    
    C = omega0*(1+delta)
    FWHM2 = C**2*Tion
    variance = FWHM2/(2.35482)**2
    # keV^2 to MeV^2
    variance /= 1e6
    
    return mean, variance #note: this is returned in MeV!!!


#### DD neutrons ####
# Mean calculation




#%%

'''
Define the different temperature profiles centered around bang time:
(need to figure out what to set temperature outside burn time range)
'''
t_0 = 200 #bang time (in picoseconds)
burn_time = 100 #choose burn time of 100ps, take this to be equal to FWHM for now 
t_std = burn_time / 2.35482 #converting FWHM to sigma
temp = 20


global_list_skews_DD = []
global_list_kurts_DD = []
global_list_skews_DT = []
global_list_kurts_DT = []



max_min = []

DT_values=[]
all_values=[]

tmin = 10
tmax = 30



detectors=np.linspace(0,0.5,50)
#detectors=[0.05, 0.3]
number_of_tests=1





t_mid_list=[]


skew_kurt_list=[]

#All_temperatures=np.linspace(25,35,11)

for i in range(number_of_tests):
#for i in All_temperatures:
    #tmin=np.random.choice(np.linspace(5,15,11)) 

    
    
    #tmin=np.random.choice(np.linspace(5,15,500))
    #tmin_1=np.random.choice(np.linspace(5,15,500))
    tmin_1=5
    tmax_1=30
    #tmax_1 = np.random.choice(np.linspace(15,30,16))
    #tmin_2=np.random.choice(np.linspace(5,15,500))    
    tmin_2=5
    #t_random = np.random.choice([150,160,170,180, 190, 210, 220, 230, 240, 250])
    t_random=200

    #tmax=i
    #max_min.append([tmin,tmax])
    #print(tmin, tmax)
    
    max_min.append([tmin_1, tmax_1, tmin_2])
    print(tmin_1, tmax_1, tmin_2)
    
    
    t_mid_list.append(t_random)
    
    
    #linearly increasing temperature from 4.3keV to 15keV over burn time = 100ps
    def lininc(t, Tmin = tmin, Tmax = tmax):    
        
        #temperatures constant outside burn
        if t < (t_0 - burn_time/2):
            return Tmin
        
        elif t > (t_0 + burn_time/2):
            return Tmax
        
        else: #during burn
            grad = (Tmax - Tmin) / burn_time
            y_midpoint = (Tmax + Tmin) / 2
            c = y_midpoint - grad * t_0
            return grad * t + c #returns temperature in keV
        
        
    #linearly decreasing temperature from 10keV to 1keV over burn time = 100ps.
    def lindec(t, Tmin = tmin, Tmax = tmax):
        
        #temperatures constant outside burn
        if t < (t_0 - burn_time/2):
            return Tmax
        
        elif t > (t_0 + burn_time/2):
            return Tmin
        
        #during burn
        else:
            grad = (Tmin - Tmax) / burn_time
            y_midpoint = (Tmax + Tmin) / 2
            c = y_midpoint - grad * t_0
            return grad * t + c #returns temperature in keV
        
    def incdec(t, Tmin_1 = tmin_1, Tmax = tmax_1, Tmin_2 = tmin_2, tmid=t_random):

        #temperatures constant outside burn
        min_time = t_0 - burn_time/2
        max_time = t_0 + burn_time/2

        if t < min_time:
            return Tmin_1
        
        elif t > max_time:
            return Tmin_2
        
        #linear increase of temperature at start of burn
        elif t > min_time and t < tmid:
            grad = (Tmax - Tmin_1) / (tmid-min_time)
            c = Tmin_1 - grad * min_time
            return grad * t + c #returns temperature in keV
        
        #linear decrease of temperature in second half of burn
        elif t < max_time and t > tmid:
            grad = (Tmin_2 - Tmax) / (max_time-tmid)
            c = Tmin_2 - grad * max_time
            return grad * t + c #returns temperature in keV    
        
        
        
    #constant temperature profile 
    def const_temp(t, T = temp):
        return T #in keV
    
    
    def exp(t, tmin, tmax):
        if t < (t_0 - burn_time/2):
            return tmin
        
        elif t > (t_0 + burn_time/2):
            return tmax
        
        #linear increase of temperature at start of burn
        else:
            ex = (tmax/tmin)/(t_0 + burn_time/2)
            return tmin*np.exp(ex)
        
        
        
        
        
        
    #Define Source function S(E,t)

    def S(E, t, T_prof, source_type='DT'):
        
        if source_type == 'DD':
            E_0, E_var = DDprimspecmoments(T_prof(t)) 
        else:
            E_0, E_var = DTprimspecmoments(T_prof(t)) 
        E_std = np.sqrt(E_var)
        
        #gaussian in energy (taken in units of MeV)
        energy_gauss = np.exp(-(E - E_0)**2 / (2 * E_std**2))
        
        #gaussian in time
        #time_gauss = np.exp(-(t - t_0)**2 / (2 * t_std**2))

        time_gauss = skewnorm.pdf(((t-t_0)/t_std), 0)*np.sqrt(2*np.pi)
        
        norm = 1 / (2 * np.pi * t_std* E_std)
        
        return  norm*energy_gauss * time_gauss


    def generate_source(T_prof, source_type):
        

        #define grid parameters
        n_energy, n_time = (200, 300) #number of grid points
        if source_type =='DD':
            energies = np.linspace(1.5, 3.5, n_energy) #in MeV
            
        else:
            energies = np.linspace(13, 15, n_energy) #in MeV
        times = np.linspace(1, 399, n_time) #t=100 to t=300

        #generate grid
        E_grid, t_grid = np.meshgrid(energies, times) 
        Z = np.zeros([n_time, n_energy])

        #creating data, Z are the values of the pdf
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                Z[i][j] = S(E_grid[i][j], t_grid[i][j], T_prof, source_type)           
        
        #normalise the source
        #Z = norm * Z
        '''
        #plot surface
        fig = plt.figure()
        ax = Axes3D(fig)
        surf = ax.plot_surface(E_grid, t_grid, Z, cmap=cm.coolwarm)

        #customise plot
        ax.set_ylabel('time (ps)')
        ax.set_xlabel('energy (Mev)')
        #ax.set_zlabel('pdf')
        #ax.set_yticks(np.arange(0,0.125,0.025))
        ax.azim =- 80
        ax.elev = 40
        fig.colorbar(surf, shrink=0.5, aspect=15)
        #plt.title("Linearly Increasing Temperature")

        plt.show()
        '''
        return Z, E_grid, t_grid
        

    particles_num = 5000
        


    '''Testing generate source: should plot source function and return the source data'''
    Z_DD, E_grid_DD, t_grid_DD = generate_source(lininc, 'DD')


    
    #Empty arrays to record data:
    time_emitted_DD = np.array([]) #time particle was emitted
    unnorm_velocities_DD = np.array([]) #particle velocities
    number_of_particles_DD = np.array([]) #number of respective particles with above values
    energies_DD = [] #turn into array!!!
    
    #Just a dataframe
    particle_df_DD = pd.DataFrame(columns = ['time emitted', 'energy', ' number of particles'])
    
    
    #Goes thorugh the 2D arrays containing each energy and time and finds the number of particles for each of those values
    #i.e. saving the data of grid points if values > 1
    for i in range(len(Z_DD)):
        for j in range(len(Z_DD[0])):
            if particles_num * Z_DD[i][j]>1:
                
                #time in picoseconds
                time_emitted_DD = np.append(time_emitted_DD, t_grid_DD[i][j])
                
                #energies in MeV
                energies_DD.append(E_grid_DD[i][j])
                
                #velocities in ms^-1
                unnorm_velocities_DD = np.append(unnorm_velocities_DD, np.sqrt(E_grid_DD[i][j] * 1.6e-13 * 2
                                          / 1.67e-27))
                
                #velocities_DD = unnorm_velocities_DD/np.mean(unnorm_velocities_DD)*10**7
                velocities_DD = unnorm_velocities_DD

                #save integer number of particles
                num = np.round(particles_num  * Z_DD[i][j])
                number_of_particles_DD = np.append(number_of_particles_DD, num)
    
    
    skews_DD=[]
    kurts_DD=[]

    
    for detector in detectors:
        time_arrive_DD = []
            
        for i in range(len(number_of_particles_DD)):
            time_arrive_DD.append(time_emitted_DD[i]+detector/velocities_DD[i]*1e12)
    
        skewness_DD = np.array([])
        for i in range(len(number_of_particles_DD)):
            particles_DD = int(number_of_particles_DD[i])
            #print(i)
            for j in range(particles_DD):
                skewness_DD = np.append(skewness_DD,time_arrive_DD[i])
        
        '''
        plt.figure()
        #scatter = plt.scatter(time_arrive,number_of_particles/max(number_of_particles), c = energies, cmap = cm.plasma)


        bins=20
        dist_bins=(max(time_arrive)-min(time_arrive))/bins

        binned_time_arrive=[]
        binned_number_of_particles=[]

        for i in range(bins):
            min_val = min(time_arrive)+i*dist_bins
            max_val = min(time_arrive)+(i+1)*dist_bins

            
            indexes = np.where(np.logical_and(time_arrive>min_val, time_arrive<max_val))
            
            binned_time_arrive.append(min(time_arrive)+i*dist_bins/2)
            
            binned_number_of_particles.append(sum(number_of_particles[indexes]))

        
        scatter = plt.scatter(binned_time_arrive,binned_number_of_particles/max(binned_number_of_particles))

        plt.colorbar(scatter, shrink=1, aspect=15, label = 'Energies (MeV)')
        plt.title('detector at ' + np.str(np.around(detector,3))+ 'm', fontsize = 10)
        plt.xlabel('Time of Arrival (ps)')
        plt.ylabel('Normalised Flux')
        plt.grid()
        '''
        
        
        skews_DD.append(skew(skewness_DD))
        kurts_DD.append(kurtosis(skewness_DD))

        
        

    
    
    '''Testing generate source: should plot source function and return the source data'''
    Z_DT, E_grid_DT, t_grid_DT = generate_source(lininc, 'DT')


    
    #Empty arrays to record data:
    time_emitted_DT = np.array([]) #time particle was emitted
    
    unnorm_velocities_DT = np.array([]) #particle velocities
    
    number_of_particles_DT = np.array([]) #number of respective particles with above values
    energies_DT = [] #turn into array!!!
    
    #Just a dataframe
    particle_df_DT = pd.DataFrame(columns = ['time emitted', 'energy', ' number of particles'])
    
    
    #Goes thorugh the 2D arrays containing each energy and time and finds the number of particles for each of those values
    #i.e. saving the data of grid points if values > 1
    for i in range(len(Z_DT)):
        for j in range(len(Z_DT[0])):
            if particles_num * Z_DT[i][j]>1:
                
                #time in picoseconds
                time_emitted_DT = np.append(time_emitted_DT, t_grid_DT[i][j])
                
                #energies in MeV
                energies_DT.append(E_grid_DT[i][j])
                
                #velocities in ms^-1
                unnorm_velocities_DT = np.append(unnorm_velocities_DT, np.sqrt(E_grid_DT[i][j] * 1.6e-13 * 2
                                          / 1.67e-27))
                
                #velocities_DT = unnorm_velocities_DT/np.mean(unnorm_velocities_DT)*10**7
                velocities_DT = unnorm_velocities_DT
                
                
                #save integer number of particles
                num = np.round(particles_num  * Z_DT[i][j])
                number_of_particles_DT = np.append(number_of_particles_DT, num)
    

    
    skews_DT=[]
    kurts_DT=[]

    
    for detector in detectors:
        time_arrive_DT = []
            
        for i in range(len(number_of_particles_DT)):
            time_arrive_DT.append(time_emitted_DT[i]+detector/velocities_DT[i]*1e12)
    
        skewness_DT = np.array([])
        for i in range(len(number_of_particles_DT)):
            particles_DT = int(number_of_particles_DT[i])
            #print(i)
            for j in range(particles_DT):
                skewness_DT = np.append(skewness_DT,time_arrive_DT[i])
        
        '''
        plt.figure()
        #scatter = plt.scatter(time_arrive,number_of_particles/max(number_of_particles), c = energies, cmap = cm.plasma)


        bins=20
        dist_bins=(max(time_arrive)-min(time_arrive))/bins

        binned_time_arrive=[]
        binned_number_of_particles=[]

        for i in range(bins):
            min_val = min(time_arrive)+i*dist_bins
            max_val = min(time_arrive)+(i+1)*dist_bins

            
            indexes = np.where(np.logical_and(time_arrive>min_val, time_arrive<max_val))
            
            binned_time_arrive.append(min(time_arrive)+i*dist_bins/2)
            
            binned_number_of_particles.append(sum(number_of_particles[indexes]))

        
        scatter = plt.scatter(binned_time_arrive,binned_number_of_particles/max(binned_number_of_particles))

        plt.colorbar(scatter, shrink=1, aspect=15, label = 'Energies (MeV)')
        plt.title('detector at ' + np.str(np.around(detector,3))+ 'm', fontsize = 10)
        plt.xlabel('Time of Arrival (ps)')
        plt.ylabel('Normalised Flux')
        plt.grid()
        '''
        
        
        skews_DT.append(skew(skewness_DT))
        kurts_DT.append(kurtosis(skewness_DT))

    print(velocities_DT.mean())
    print(velocities_DT.std())
    print(velocities_DD.mean())
    print(velocities_DD.std())        
        
    DT_values.append([skews_DT, kurts_DT])
    all_values.append([skews_DT, kurts_DT, skews_DD, kurts_DD])
    
    global_list_skews_DD.append(skews_DD)
    global_list_kurts_DD.append(kurts_DD)
    global_list_skews_DT.append(skews_DT)
    global_list_kurts_DT.append(kurts_DT)
    

    skew_kurt_list.append([skews_DD, skews_DT, kurts_DD, kurts_DT])
    

    plt.figure()
    #plt.plot(detectors,skews_DD, 'x', label ='DD Skewness')
    #plt.plot(detectors, kurts_DD, 'x', label = 'DD Kurtosis')
    plt.plot(detectors,skews_DT, 'x', label ='DT Skewness')
    plt.plot(detectors, kurts_DT, 'x', label = 'DD Kurtosis')
    plt.xlabel('detector placement (m)')
    plt.ylabel('Skewness/Kurtosis')
    plt.grid()
    plt.legend(loc='lower right')
    #plt.title('Constant Temperature (20 keV)')
    plt.title(f'tmin: {tmin}, tmax: {tmax}')
    plt.ylim(ymin = -1.2)
    plt.savefig(r'C:\Users\rayan\OneDrive\Documents\Y4\MSci Project\lininc_{}_{}.png'.format(tmax, tmin), dpi=100)
    


#%%
''' Validating temperature profiles and seeing how std varies with profile '''

t_max_1=20
t_random=225


t = np.linspace(100,300,1000)
T = np.zeros(len(t))
for i in range(len(t)):
    T[i] = incdec(t[i], 10, 25, 5, 225)

sigma = np.sqrt(DTprimspecmoments(T)[1])
fif,ax=plt.subplots()

plt.plot(t, T)    
m=np.linspace(100,225,100)
plt.plot(m, [25] * 100, color='black',linestyle='--')
n=np.linspace(0,25,100)
plt.plot([225] * 100,n, color='black',linestyle='--')

#plt.title('Temperature against Time')
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Temperature (keV)')
ax.grid()
ax.annotate('$T_{final}$', xy=(252,6),fontsize=14)
ax.annotate('$T_{initial}$', xy=(150,8),fontsize=14)
ax.annotate('$T_{max}$', xy=(101,27),fontsize=14)
ax.annotate('$t_{peak}$', xy=(227,1), fontsize=14)
#ax.annotate('t_peak', xy=(227,1))
#ax.annotate('t_peak', xy=(227,1))
ax.set_ylim(0,40)
ax.set_xlim(100,300)
plt.show

#%%


#%%
for i in [0,1,2]:
    plt.plot(detectors, global_list_skews_DT[i],'x', label = str(All_temperatures[i])+('K'))

plt.legend()
plt.grid()
plt.xlabel('detector placement (m)')
plt.ylabel('Skewness')

#%%
from scipy.interpolate import interp1d

x = detectors[::5]
y = skews_DT[::5]
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')


xnew = np.linspace(min(x), max(x), num=4100, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.xlabel('detector placement (m)')
plt.ylabel('Skewness')
plt.grid()
plt.ylabel()
plt.show()

#%%



x = detectors
y = skew_kurt_list[0][0]



f = InterpolatedUnivariateSpline(x, y, k=4)
cr_pts = f.derivative().roots()
cr_pts = np.append(cr_pts, (x[0], x[-1]))  # also check the endpoints of the interval
cr_vals = f(cr_pts)
min_index = np.argmin(cr_vals)
max_index = np.argmax(cr_vals)
print("Maximum value {} at {}\nMinimum value {} at {}".format(cr_vals[max_index], cr_pts[max_index], cr_vals[min_index], cr_pts[min_index]))




#%%

max_skews=[]
ratios=[]
x = detectors
for i in range(len(skew_kurt_list)):
    temp_list=[]
    for j in range(2):

        y = skew_kurt_list[i][j]



        f = InterpolatedUnivariateSpline(x, y, k=4)
        cr_pts = f.derivative().roots()
        cr_pts = np.append(cr_pts, (x[0], x[-1]))  # also check the endpoints of the interval
        cr_vals = f(cr_pts)
        min_index = np.argmin(cr_vals)
        max_index = np.argmax(cr_vals)
        
        temp_list.append(cr_pts[max_index])
    ratios.append((temp_list[1]/temp_list[0]))
    max_skews.append(temp_list)
        

plt.plot(np.linspace(25,35,11),ratios)
plt.ylabel('Ratio')
plt.xlabel('Temperature (keV)')
plt.grid()

#%%
array_1=[]
array_2=[]
for i in range(len(check[0][0])):
    if check[0][0][i]<150:
        array_1.append(check[0][1][i])
    else:
        array_2.append(check[0][1][i])
        
        
#%%

cols= list(range(0,len(detectors)))
column_string = [str(i) for i in cols]


up=np.asarray(global_list_skews_DD).reshape(number_of_tests,len(detectors))


df = pd.DataFrame(up[:number_of_tests],columns= column_string)


X_train, X_test, y_train, y_test = train_test_split(df[column_string], max_min, test_size=0.2, random_state=4)


regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)





pred = regr.predict(X_test)




xxx = np.abs(np.array(pred)-np.array(y_test))



df_1= pd.DataFrame(xxx,columns = ['1', '2','3'])

error1=np.round(df_1['1'].sum()/len(df_1),2)
error2=np.round(df_1['2'].sum()/len(df_1),2)
#error3=np.round(df_1['3'].sum()/len(df_1),2)

std_1 = np.round(np.std(df_1['1']),2)
std_2 = np.round(np.std(df_1['2']),2)
#std_3 = np.round(np.std(df_1['3']),2)



print(f'absolute error is {error1} +- {std_1} for tmin {error2} +- {std_2} for tmax')

#%%
cols= list(range(0,len(detectors)*2))
column_string = [str(i) for i in cols]
up=np.asarray(DT_values).reshape(number_of_tests,len(detectors)*2)


df = pd.DataFrame(up[:number_of_tests],columns= column_string)

X_train, X_test, y_train, y_test = train_test_split(df[column_string], max_min, test_size=0.2, random_state=4)


regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)





pred = regr.predict(X_test)




xxx = np.abs(np.array(pred)-np.array(y_test))



df_1= pd.DataFrame(xxx,columns = ['1', '2','3'])


error1=np.round(df_1['1'].sum()/len(df_1),2)
error2=np.round(df_1['2'].sum()/len(df_1),2)
#error3=np.round(df_1['3'].sum()/len(df_1),2)

std_1 = np.round(np.std(df_1['1']),2)
std_2 = np.round(np.std(df_1['2']),2)
#std_3 = np.round(np.std(df_1['3']),2)



print(f'absolute error is {error1} +- {std_1} for tmin {error2} +- {std_2} for tmax')

#%%

cols= list(range(0,len(detectors)*4))
column_string = [str(i) for i in cols]
up=np.asarray(all_values).reshape(number_of_tests,len(detectors)*4)


df = pd.DataFrame(up[:number_of_tests],columns= column_string)


X_train, X_test, y_train, y_test = train_test_split(df[column_string], max_min, test_size=0.2, random_state=4)


regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)





pred = regr.predict(X_test)




xxx = np.abs(np.array(pred)-np.array(y_test))



df_1= pd.DataFrame(xxx,columns = ['1', '2','3'])



error1=np.round(df_1['1'].sum()/len(df_1),2)
error2=np.round(df_1['2'].sum()/len(df_1),2)
#error3=np.round(df_1['3'].sum()/len(df_1),2)

std_1 = np.round(np.std(df_1['1']),2)
std_2 = np.round(np.std(df_1['2']),2)
#std_3 = np.round(np.std(df_1['3']),2)



print(f'absolute error is {error1} +- {std_1} for tmin {error2} +- {std_2} for tmax')



#%%
max_only=[]
for i in range(len(y_train)):
    max_only.append(y_train[i][1])
               


#Random min and max Temperatures, 1 detector at 0.3
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_white
import statsmodels.api as sm
import pandas as pd


N_train = sm.add_constant(X_train['1'])

#fit regression model
model = sm.OLS(max_only, N_train).fit()


white_test = het_white(model.resid,  model.model.exog)

#define labels to use for output of White's test
labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']

#print results of White's test
print(dict(zip(labels, white_test)))

{'Test Statistic': 7.076620330416624, 'Test Statistic p-value': 0.21500404394263936,
 'F-Statistic': 1.4764621093131864, 'F-Test p-value': 0.23147065943879694}
x=np.linspace(25,35,10)
y=df['129']
plt.plot(x, y, 'x', color = 'darkviolet')
#plt.plot(x,df['129'], 'x')
#plt.plot(x,df['204'], 'x')
#plt.plot(x,df['304'], 'x')
plt.ylabel('Kurtosis')
plt.xlabel('Temperature (keV)')
plt.grid()
import scipy
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
print(r_value**2)
#%%
bins=100
dist_bins=(max(time_arrive)-min(time_arrive))/bins

binned_time_arrive=[]
binned_number_of_particles=[]

for i in range(bins):
    min_val = min(time_arrive)+i*dist_bins
    max_val = min(time_arrive)+(i+1)*dist_bins

    
    indexes = np.where(np.logical_and(time_arrive>min_val, time_arrive<max_val))
    
    binned_time_arrive.append(min(time_arrive)+i*dist_bins/2)
    
    binned_number_of_particles.append(sum(number_of_particles[indexes]))


#%%

DD_skews = global_list_skews_DD
DD_kurts = global_list_kurts_DD


DT_skews = global_list_skews_DT
DT_kurts = global_list_kurts_DT


plt.scatter(detectors, DT_skews[0], label = 'dt Skewness', marker = 'x')
#plt.scatter(detectors, DT_kurts[0], label = 'dt Kurtosis', marker='x')
plt.scatter(detectors, DD_skews[0], label = 'dd Skewness', marker = 'x')
#plt.scatter(detectors, DD_kurts[0], label = 'dd Kurtosis', marker= 'x')
plt.ylabel('Skewness')
plt.xlabel('detector placement (m)')
plt.grid()
plt.legend(loc='lower right')



#%%

T_list = np.linspace(0,35, 36)

DT_means = [DTprimspecmoments(T)[0] for T in T_list]
DT_vars = [DTprimspecmoments(T)[1] for T in T_list]
DD_means = [DDprimspecmoments(T)[0] for T in T_list]
DD_vars = [DDprimspecmoments(T)[1] for T in T_list]


plt.plot(T_list, np.asarray(DT_vars)/np.asarray(DT_means), label='dt')
plt.plot(T_list, np.asarray(DD_vars)/np.asarray(DD_means), label='dd')

plt.xlabel('Temperature (keV)')
plt.ylabel('Ratio')
plt.legend()
plt.grid()


#%%
plt.plot(T_list, np.asarray(DT_means), label = 'Mean Energy')
plt.plot(T_list, np.asarray(DT_means)+np.sqrt(np.asarray(DT_vars)), label = 'Mean Energy ± 1σ', linestyle ='--', color='r')
plt.plot(T_list, np.asarray(DT_means)-np.sqrt(np.asarray(DT_vars)), linestyle ='--', color='r')

#plt.fill_between(T_list, np.asarray(DT_means)+np.asarray(DT_vars), np.asarray(DT_means)-np.asarray(DT_vars))
#plt.fill_between(T_list, np.asarray(DD_means)+np.asarray(DD_vars), np.asarray(DD_means)-np.asarray(DD_vars), linestyle='--', edgecolor='r', color='w')


plt.xlabel('Ion Temperature (keV)')
plt.ylabel('Neutron Energy (MeV)')
plt.legend()
plt.grid()
#%%
new_sum_DT=[]
for i in range(len(number_of_particles_DT)):
    new_sum_DT.append(number_of_particles_DT[i]*velocities_DT[i])
    
new_sum_DD=[]
for i in range(len(number_of_particles_DD)):
    new_sum_DD.append(number_of_particles_DD[i]*velocities_DD[i])    
    
    
    
#%%
#mean=51895820.5620231
#diff=447456.75966333295
#mean=51912354.16222052
#diff=519870.307146645
##mean=51924423.488738336
#diff=575049.279719555

mean=velocities_DD.mean()
diff = velocities_DD.std()

slow = (mean-diff)*1.009
fast=mean+diff

t = slow*10**-10/(fast-slow)

print(t*fast)
print(slow**2/(2*1.6*10**-13)*1.67e-27)
    