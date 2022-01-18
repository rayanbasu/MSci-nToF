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
#%%
'''
Define the different temperature profiles centered around bang time:
(need to figure out what to set temperature outside burn time range)
'''
t_0 = 200 #bang time (in picoseconds)
burn_time = 100 #choose burn time of 100ps, take this to be equal to FWHM for now 
t_std = burn_time / 2.35482 #converting FWHM to sigma
temp = 20
global_list = []


tmin = 10
tmax = 10
for tmax in [20,23,31,34,35]:
    tmin=np.random.choice([8,10,12,13])
    print(tmin, tmax)
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
        
    def incdec(t, Tmin_1 = tmin, Tmax = tmax, Tmin_2 = tmin):
        
        #temperatures constant outside burn
        if t < (t_0 - burn_time/2):
            return Tmin_1
        
        elif t > (t_0 + burn_time/2):
            return Tmin_2
        
        #linear increase of temperature at start of burn
        elif t > (t_0 - burn_time/2) and t < t_0:
            grad = (Tmax - Tmin_1) / (burn_time/2)
            c = Tmax - grad * t_0
            return grad * t + c #returns temperature in keV
        
        #linear decrease of temperature in second half of burn
        elif t < (t_0 + burn_time/2) and t > t_0:
            grad = (Tmin_2 - Tmax) / (burn_time/2)
            c = Tmax - grad * t_0
            return grad * t + c #returns temperature in keV    
    #constant temperature profile 
    def const_temp(t, T = temp):
        return T #in keV
    
    #Define Source function S(E,t)
    def S(E, t, T_prof):
        
        E_0, E_var = DTprimspecmoments(T_prof(t)) 
        E_std = np.sqrt(E_var)
        
        #gaussian in energy (taken in units of MeV)
        energy_gauss = np.exp(-(E - E_0)**2 / (2 * E_std**2))
        
        #gaussian in time
        time_gauss = np.exp(-(t - t_0)**2 / (2 * t_std**2))
        
        return energy_gauss * time_gauss


    def generate_source(T_prof):
        
        #first calculate the normalisation constant by numerically integrating 
        #over energy and time    
        norm_integral = sp.integrate.nquad(lambda E, t: S(E, t, T_prof), [[0, 100]
                                                                 ,[-np.inf, np.inf]])[0]
        norm = 1 / (norm_integral)

        #define grid parameters
        n_energy, n_time = (200, 100) #number of grid points
        energies = np.linspace(13, 15, n_energy) #in MeV
        times = np.linspace(100, 300, n_time) #t=100 to t=300

        #generate grid
        E_grid, t_grid = np.meshgrid(energies, times) 
        Z = np.zeros([n_time, n_energy])

        #creating data, Z are the values of the pdf
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                Z[i][j] = S(E_grid[i][j], t_grid[i][j], T_prof)           
        
        #normalise the source
        Z = norm * Z
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
        


    '''Testing generate source: should plot source function and return the source data'''
    Z, E_grid, t_grid = generate_source(lininc)

    particles_num = 1000
    
    
    #Empty arrays to record data:
    time_emitted = np.array([]) #time particle was emitted
    velocities = np.array([]) #particle velocities
    number_of_particles = np.array([]) #number of respective particles with above values
    energies = [] #turn into array!!!
    
    #Just a dataframe
    particle_df = pd.DataFrame(columns = ['time emitted', 'energy', ' number of particles'])
    
    
    #Goes thorugh the 2D arrays containing each energy and time and finds the number of particles for each of those values
    #i.e. saving the data of grid points if values > 1
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            if particles_num * Z[i][j]>1:
                
                #time in picoseconds
                time_emitted = np.append(time_emitted, t_grid[i][j])
                
                #energies in MeV
                energies.append(E_grid[i][j])
                
                #velocities in ms^-1
                velocities = np.append(velocities, np.sqrt(E_grid[i][j] * 1.6e-13 * 2
                                          / 1.67e-27))
                
                #save integer number of particles
                num = np.round(particles_num  * Z[i][j])
                number_of_particles = np.append(number_of_particles, num)
    
    
    skews=[]
    kurts=[]
    detectors=np.linspace(0.1,1,8)
    
    for detector in detectors:
        time_arrive = []
            
        for i in range(len(number_of_particles)):
            time_arrive.append(time_emitted[i]+detector/velocities[i]*1e12)
    
        skewness = np.array([])
        for i in range(len(number_of_particles)):
            particles = int(number_of_particles[i])
            #print(i)
            for j in range(particles):
                skewness = np.append(skewness,time_arrive[i])
        
        '''plt.figure()
        scatter = plt.scatter(time_arrive,number_of_particles/max(number_of_particles),
                           c = energies, cmap = cm.plasma)
    
    
        plt.colorbar(scatter, shrink=1, aspect=15, label = 'Energies (MeV)')
        plt.title('detector at ' + np.str(np.around(detector,3))+ 'm', fontsize = 10)
        plt.xlabel('Time of Arrival (ps)')
        plt.ylabel('Normalised Flux')'''
        
        
        skews.append(skew(skewness))
        kurts.append(kurtosis(skewness))
        
    global_list.append(skews)
    
r'''
    plt.figure()
    plt.plot(detectors,skews, 'x')
    plt.plot(detectors, kurts, 'x')
    plt.xlabel('detector placement (m)')
    plt.ylabel('Skewness')
    plt.grid()
    #plt.title('Constant Temperature (20 keV)')
    plt.title(f'Constant Temperature ({temp} keV)')
    #plt.ylim(ymax = 0.5, ymin = 0)
    plt.savefig(r'C:\Users\rayan\OneDrive\Documents\Y4\MSci Project\lininc_{}_{}.png'.format(tmax, tmin), dpi=100)
''' 


#%%
from sklearn.datasets import make_regression


cols= list(range(0,len(detectors)))
column_string = [str(i) for i in cols]


df = pd.DataFrame(global_list[:4],columns= column_string)



from sklearn import linear_model

X= df[column_string]
y=np.array([[13,  20], [10,  23],[10,  31], [12,  34]])

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.predict([global_list[4]]))

#%%

[2,3,5,7,10]

[11.05]

[35.76]
#%%




co=[2,3,5,7,10]


err_1=[34.33, 34.44,34.45, 34.46,34.47, 34.31, 34.45]     

err_2=[40.76, 34.64, 34.63,  ]

plt.plot(co,err)            