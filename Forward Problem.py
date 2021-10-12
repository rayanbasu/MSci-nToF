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
    dist = normal(14, 1, 10)
    velocity_distribution = np.sqrt((2*dist*1.67*10**-19)/(1.67*10**-27))
    return velocity_distribution


data = energy_dist()

print(data)

plt.hist(data)
#%%
velocity_list = np.array([])
new_position_list  = np.zeros(1500)
old_position_list  = np.zeros(1500)

detector_list=[]

for i in range(0, 4000):
    for k in range(len(velocity_list)):

        new_position_list[k] += velocity_list[k]*10**-5
        
    
    
    if i<150:
        new_vel_dist = energy_dist()
        velocity_list=np.append(velocity_list,new_vel_dist)

    
    detector_1 = 1500

    counter = 0
    for j in range(len(new_position_list)):
        if new_position_list[j]>detector_1 and old_position_list[j]<detector_1:
            counter+=1


    detector_list.append(counter)
    
    old_position_list = new_position_list.copy()
    
#%%

plt.plot(detector_list)