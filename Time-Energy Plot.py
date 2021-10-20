# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 21:39:09 2021

@author: rayan
"""
import numpy as np
import matplotlib.pyplot as plt
#%%

t_0 = 100

def time(x, t_std):
    return np.exp(-(((x-t_0)**2)/(2*t_std**2)))/(t_std*np.sqrt(2**np.pi))



time_list = np.linspace(0,200,1000)



plt.plot(time_list, time(time_list, 10))


#%%


e_0 = 14
e_std = 1
def energy(x):
    return np.exp(-(((x-e_0)**2)/(2*e_std**2)))/(e_std*np.sqrt(2**np.pi))



energy_list = np.linspace(10,18,1000)


plt.plot(energy_list, energy(energy_list))

#%%
Z1 = 1

#%%

from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = plt.axes(projection="3d")

def density(x, y):
    return x*y



X, Y = np.meshgrid(time_list, energy_list)
Z1 = np.zeros([1000,1000])
for i in range(len(X)):
    t_std = 0.1/(i*0.01+1)
    for j in range(len(Y)):
        Z1[i][j] = time(X[i][j], t_std) * energy(Y[i][j])  

#Z = np.exp(-(((Y-e_0)**2)/(2*e_std**2)))/(e_std*np.sqrt(2**np.pi)) * np.exp(-(((X-t_0)**2)/(2*t_std**2)))/(t_std*np.sqrt(2**np.pi))


fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z1)
plt.xlabel('time (ps)')
plt.ylabel('energy (Mev)')
plt.show()

#%%

Z1 = np.zeros([1000,1000])
for i in range(len(X)):
    t_std = 100/(i+1)
    for j in range(len(Y)):
        Z1[i][j] = time(X[i][j], t_std) * energy(Y[i][j])  