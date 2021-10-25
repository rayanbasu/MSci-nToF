# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 21:39:09 2021

@author: rayan
"""
import numpy as np
import matplotlib.pyplot as plt
#%%
n=1000


t_0 = 100
t_std = 10


def time(t):
    #return np.exp(-(((t-t_0)**2)/(2*t_std**2)))/(t_std*np.sqrt(2**np.pi))
    return(t*2/200**2)



time_list = np.linspace(0,200,n)



plt.plot(time_list, time(time_list))


#%%


e_0 = 14.05
e_std = 1
def energy(x, e_0, e_std):
    return np.exp(-(((x-e_0)**2)/(2*e_std**2)))/(e_std*np.sqrt(2**np.pi))



energy_list = np.linspace(13,15,n)


plt.plot(energy_list, energy(energy_list, e_0, e_std))

#%%
# Returns the mean and variance based on Ballabio
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
    
    
    
    return mean, variance
#%%

temperature_increase = np.linspace(4,15,n)
temperature_decrease= np.linspace(10,0,n)

#%%
#4.3 to 15. 10 to 0

from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = plt.axes(projection="3d")


X, Y = np.meshgrid(time_list, energy_list)
Z1 = np.zeros([n,n])
for i in range(len(X)):
    
      
    
    e_0, e_std = DTprimspecmoments(temperature_increase[i])
    
    
    
    for j in range(len(Y)):
        Z1[i][j] = time(X[i][j]) * energy(Y[i][j], e_0,np.sqrt(e_std))  

#Z = np.exp(-(((Y-e_0)**2)/(2*e_std**2)))/(e_std*np.sqrt(2**np.pi)) * np.exp(-(((X-t_0)**2)/(2*t_std**2)))/(t_std*np.sqrt(2**np.pi))

#%%

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z1)
ax.set_xlabel('time (ps)')
ax.set_ylabel('energy (Mev)')
ax.set_zlabel('pdf')
ax.azim = -50
plt.show()

#%%
for i in range(len(X)):
    
      
    
    e_0, e_std = DTprimspecmoments(temperature_increase[i])
    print(e_0, e_std)