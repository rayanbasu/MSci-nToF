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

'''
Define the different temperature profiles centered around bang time:
(need to figure out what to set temperature outside burn time range)
'''
t_0 = 200 #bang time (in picoseconds)
burn_time = 100 #choose burn time of 100ps, take this to be equal to FWHM for now 
t_std = burn_time / 2.35482 #converting FWHM to sigma


#linearly increasing temperature from 4.3keV to 15keV over burn time = 100ps
def lininc(t):    
    
    #temperatures constant outside burn
    if t < (t_0 - burn_time/2):
        return 4.3
    
    elif t > (t_0 + burn_time/2):
        return 15
    
    else: #during burn
        grad = (15 - 4.3) / burn_time
        y_midpoint = (15 + 4.3) / 2
        c = y_midpoint - grad * t_0
        return grad * t + c #returns temperature in keV
    
    
#linearly decreasing temperature from 10keV to 1keV over burn time = 100ps.
def lindec(t):
    
    #temperatures constant outside burn
    if t < (t_0 - burn_time/2):
        return 10
    
    elif t > (t_0 + burn_time/2):
        return 1 #to avoid dividing by zero (?)
    
    #during burn
    else:
        grad = (1 - 10) / burn_time
        y_midpoint = 5.5
        c = y_midpoint - grad * t_0
        
        return grad * t + c #returns temperature in keV
    
#constant temperature profile 
def const_temp(t):
    return 10 #in keV

#Define Source function S(E,t)
def S(E, t):
    
    #normalisation constant (still need to calculate this!! use this for now)
    norm = 0.021249110488318672
    
    #make the temperature profile modifiable in function argument!!
    E_0, E_var = DTprimspecmoments(lindec(t)) #chosen lininc() for now
    E_std = np.sqrt(E_var)
    
    #gaussian in energy (taken in units of MeV)
    energy_gauss  = np.exp(-(E - E_0)**2 / (2 * E_std**2))
    
    #gaussian in time
    time_gauss = np.exp(-(t - t_0)**2 / (2 * t_std**2))
    
    return norm * energy_gauss * time_gauss

#%%
'''
Validating temperature profiles and seeing how std varies with profile
'''
t = np.linspace(0,1000,1000)
T = np.zeros(len(t))
for i in range(len(t)):
    T[i] = lindec(t[i])

sigma = np.sqrt(DTprimspecmoments(T)[1])

plt.plot(t, T)    
plt.plot(t, sigma)


#%%
'''
Plot 3 different cases, lin increasing, decreasing, and constant - see differences 
need to wrap this all in a function !!
'''
#make a grid
n_energy, n_time = (100, 100) #number of grid points
energies = np.linspace(13, 15, n_energy) #in MeV
times = np.linspace(100, 300, n_time) #t=100 to t=300

#create grid
E_grid, t_grid = np.meshgrid(energies, times) 
Z = np.zeros([n_time, n_energy])

#creating data
for i in range(len(Z)):
    for j in range(len(Z[0])):
        Z[i][j] = S(E_grid[i][j], t_grid[i][j])
        
        
#plot surface
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(E_grid, t_grid, Z, cmap=cm.coolwarm)

#customise plot
ax.set_ylabel('time (ps)')
ax.set_xlabel('energy (Mev)')
#ax.set_zlabel('pdf')
#ax.set_yticks(np.arange(0,0.125,0.025))
ax.azim =+80
fig.colorbar(surf, shrink=0.5, aspect=15)
plt.title("Linearly Increasing Temperature")

plt.show()

#%%
#This section is for finding normalisation factors

#Creating a lambda function for lininc
def g(x):
    return np.sqrt(DTprimspecmoments(lininc(x))[1])*np.exp(-(x-t_0)**2/(2*t_std**2))*np.sqrt(2*np.pi)


lininc_lambda= lambda x:g(x)

#Creating a lambda function for lindec
def h(x):
    return np.sqrt(DTprimspecmoments(lindec(x))[1])*np.exp(-(x-t_0)**2/(2*t_std**2))*np.sqrt(2*np.pi)
lindec_lambda= lambda x:h(x)


#Integrating both functions

#First integrating lininc
#i and j are the value and the error of the integral
i , j = sp.integrate.quad(lininc_lambda, 0, np.inf)

#This is A
print(1/i, j)


#Now integrating lindec
k , l = sp.integrate.quad(lindec_lambda, 0, np.inf)

#This is A
print(1/k, l)


#%%

#This is multiplied by the pdf distribution to give the number of particles for each time and energy
particles_num = 1000


#Thre arrays to record the time a particle was emitted, its velocity, and the number of particles witht those time and velocity values
time_emitted = []
velocities = []
energies = []
number_of_particles = []

#Just a dataffa
particle_df = pd.DataFrame(columns = ['time emitted', 'energy', ' number of particles'])


#Goes thorugh the 2D arrays containing each energy and time and finds the number of particles for each of those values
for i in range(len(Z)):
    for j in range(len(Z[0])):
        if particles_num * Z[i][j]>1:
            
            #time in picoseconds
            time_emitted = np.append(time_emitted, t_grid[i][j])
            
            #velocities in ms^-1
            velocities = np.append(velocities, np.sqrt(E_grid[i][j] * 1.6e-13 * 2
                                      / 1.67e-27))
            
            #save integer number of particles
            num = np.round(particles_num  * Z[i][j])
            number_of_particles = np.append(number_of_particles, num)

#%%
#Detector length
detector_placements=  np.linspace(0,1,11)
#Time it arrives at the detector is recorded in this array



fig, ax = plt.subplots(nrows=11, ncols=1)
fig.set_size_inches(18, 100)
fig.suptitle('Decreasing Temperature', fontsize = 90)
ax[10].set_xlabel('time of arrival (ps)', fontsize = 70)
ax[5].set_ylabel('flux', fontsize = 70)





for detector in detector_placements[:]:
    time_arrive = []
    
    for i in range(len(number_of_particles)):
        time_arrive.append(time_emitted[i]+detector/velocities[i]*1e12)
    
    
    #Plotting the number of particles arriving at each time

    scatter = ax[np.int(detector*10)].scatter(time_arrive,number_of_particles, c = energies, cmap = cm.plasma)
    #fig.colorbar(scatter, shrink=1, aspect=15)
    ax[np.int(detector*10)].set_title('detector at ' + np.str(np.around(detector,1))+ 'm', fontsize = 30)
    print(skew(time_arrive))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar_ax.tick_params(labelsize=30)

cbar = fig.colorbar(scatter, aspect=100, cax=cbar_ax)
cbar.set_label('Energies (MeV)', fontsize = 70, rotation=270)

fig.savefig(r'C:\Users\rayan\OneDrive\Documents\Y4\MSci Project\lininc.png', dpi=100)    

#%%
fig, ax = plt.subplots()

detector = 2.6


time_arrive = []
    
for i in range(len(number_of_particles)):
    time_arrive.append(time_emitted[i]+detector/velocities[i]*1e12)
    
    
#Plotting the number of particles arriving at each time

scatter = plt.scatter(time_arrive,number_of_particles/max(number_of_particles), c = energies, cmap = cm.plasma)
fig.colorbar(scatter, shrink=1, aspect=15, label = 'Energies (MeV)')
plt.title('detector at ' + np.str(np.around(detector,1))+ 'm', fontsize = 10)
plt.xlabel('Time of Arrival (ps)')
plt.ylabel('Normalised Flux')
fig.savefig(r'C:\Users\rayan\OneDrive\Documents\Y4\MSci Project\lininc2.6.png', dpi=100)    

#%%
skews=[]
detectors=np.linspace(0,10,21)
for detector in detectors:



    time_arrive = []
        
    for i in range(len(number_of_particles)):
        time_arrive.append(time_emitted[i]+detector/velocities[i]*1e12)

    skewness = np.array([])
    for i in range(len(number_of_particles)):
        particles = np.int(number_of_particles[i])
        print(i)
        for j in range(particles):
            skewness = np.append(skewness,time_arrive[i])
    
    print(scipy.stats.skew(skewness))
    skews.append(scipy.stats.skew(skewness))


plt.plot(detectors,skews)
plt.xlabel('detector placement (m)')
plt.ylabel('Skewness')
plt.grid()
plt.title('Linearly Decreasing')
