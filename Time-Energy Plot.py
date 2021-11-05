# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 21:39:09 2021
@author: rayan & ewan
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.integrate import quad

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
    E_0, E_var = DTprimspecmoments(lininc(t)) #chosen lininc() for now
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
    T[i] = lininc(t[i])

sigma = np.sqrt(DTprimspecmoments(T)[1])

plt.plot(t, T)    
plt.plot(t, sigma)


#%%
'''
Plot 3 different cases, lin increasing, decreasing, and constant - see differences 
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
import scipy
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

i , j = scipy.integrate.quad(lininc_lambda, 0, np.inf)

i , j = quad(lininc_lambda, 150, 250)

#This is A
print(1/i, j)


#Now integrating lindec

k , l = scipy.integrate.quad(lindec_lambda, 0, np.inf)

k , l = quad(lindec_lambda, 150, 250)

#This is A

print(1/k, l)


#%%
particles_num = 200
import pandas as pd


time_emitted = []
velocities = []
number_of_particles = []

particle_df = pd.DataFrame(columns = ['time emitted', 'energy', ' number of particles'])


for i in range(len(Z)):
    for j in range(len(Z)):
        if particles_num*Z[i][j]>1:
            print('y')
            time_emitted.append(t_grid[i][j])
            velocities.append(np.sqrt(E_grid[i][j]*1.6e-13*2/(1.67e-27)))
            number_of_particles.append(np.round(particles_num*Z[i][j]))



#%%

detector = 1000000


time_arrive = []

for i in range(len(number_of_particles)):
    time_arrive.append(time_emitted[i]+detector/velocities[i]*1e12)
    

plt.plot(time_arrive,number_of_particles)