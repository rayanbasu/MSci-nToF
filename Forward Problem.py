# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 15:57:34 2021

@author: rayan
"""

import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt

#%%
burst_duration = 150


#Energy in MeV
energy = 14

#Energy width in MeV
energy_width = 14
np.sqrt((2*energy)/(1.67*10**-27))
#%%

def energy_dist():
    dist = normal(14, 1, 100)
    velocity_distribution = np.sqrt((2*dist*1.67*10**-19)/(1.67*10**-27))
    return velocity_distribution


data = energy_dist()

print(data)

plt.hist(data)
#%%

position_list  = []

for i in range(0, 2000):
    
    
    for k in range(len(position_list)):
        position_list[k][1] += position_list[k][0]*10**-5
        
    
    
    if i<150:
        new_vel_dist = energy_dist()
        for j in range(len(new_vel_dist)):
            position_list.append([new_vel_dist[j],0])
            
    
    
    