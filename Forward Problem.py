# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 15:57:34 2021

@author: rayan & ewan
"""
#need to generate docstring for functions!! (when happy)
#make raise errorrs!

import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors


burst_duration = 150 #picoseconds
n_mass = 1.67 * 10**-27 #neutron mass (kg)

#Reaction neutron energies in MeV:
dt_neutron = 14.03 #DT reaction neutron energy
dd_neutron = 2.45 #DD reaction neutron energy

def energy_dist(neutron = dt_neutron, width = 1, n_sample = 100):
    
    #generate normal distribution of neutrons centred at given neutron energy
    energies = normal(neutron, width, n_sample)
    
    #calculate velocities of the neutrons produced (in ms^-1)
    velocity_distribution = np.sqrt((2 * energies * 1.6*10**-19 * 10**6) / 
                                    n_mass)
    return velocity_distribution

#%%
'''
here we use a timestep of 1ps (i.e. emit neutrons every 1ps)

NEED TO CLEAN UP DETETOR LISTS AND COUNTERS: PACK NEATLY INTO ARRAYS INSTEAD
OF INDIVIDUAL LISTS
'''

#empty lists to store generated neutron velocities and positions
velocities = np.array([])
new_position_list = np.array([])
old_position_list = np.array([])

detector1_list = []
detector2_list = []
detector3_list = []
detector4_list = []

for i in range(0, 5000): #from t = 0 to t = 5000 ps

    #advance neutrons by 1ps if not in first iteration
    if i != 0:
        for j in range(len(velocities)):
            new_position_list[j] += velocities[j] * 10**-12

    #emit gaussian distribution of velocities every 1ps for first 100ps
    if i < 100:
        
        #generate newly emitted neutrons and append to velocities list
        new_vel_dist = energy_dist()
        velocities = np.append(velocities, new_vel_dist)
        
        #create empty elements to track positions of new netrons
        new_position_list = np.append(new_position_list, np.zeros(len(new_vel_dist)))
        old_position_list = np.append(old_position_list, np.zeros(len(new_vel_dist)))
        
    #positions of detectors (m)
    detector_1 = 0.025
    detector_2 = 0.050
    detector_3 = 0.075
    detector_4 = 0.100
    detector_pos = np.array([detector_1, detector_2, detector_3, detector_4])
    
    
    #reset counters
    counter1 = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0
    
    #detectors check for neutrons at current timestep
    for k in range(len(new_position_list)):
        if new_position_list[k] > detector_1 and old_position_list[k] < detector_1:
            counter1 += 1     

        if new_position_list[k] > detector_2 and old_position_list[k] < detector_2:
            counter2 += 1     
    
        if new_position_list[k] > detector_3 and old_position_list[k] < detector_3:
            counter3 += 1     
            
        if new_position_list[k] > detector_4 and old_position_list[k] < detector_4:
            counter4 += 1     

    detector1_list.append(counter1)
    detector2_list.append(counter2)
    detector3_list.append(counter3)
    detector4_list.append(counter4)
    
    old_position_list = new_position_list.copy()
    
    print(i)
    print(new_position_list[0])
    
    
#%%

detectors = np.asarray([detector1_list, detector2_list, detector3_list, detector4_list])

#%%
#plotting 3d temporal and distance spread


params = {
   'axes.labelsize': 10,
   'font.size': 10,
   'legend.fontsize': 14,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'figure.figsize': [8, 5]
   } 

plt.rcParams.update(params)

fig = plt.figure()
ax = fig.gca(projection='3d')

#for colouring plots
def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)

xs = np.arange(0, len(detector1_list)) #t variable, link to above instead of manually setting
verts = []
zs = detector_pos
for i in range(len(zs)):
    ys = detectors[i]
    verts.append(list(zip(xs, ys)))

poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'), cc('y')])


poly.set_alpha(0.8)
ax.add_collection3d(poly, zs=detector_pos, zdir='y')

ax.set_xlabel('Time (ps)')
ax.set_xlim3d(0, 2800)
ax.set_ylabel('Position (m)')
ax.set_ylim3d(0.02, 0.125)
ax.set_zlabel('Flux')
ax.set_zlim3d(0, 120)

ax.set_yticks(np.arange(0,0.125,0.025))

plt.show()


#%%
plt.plot(detector1_list)
plt.plot(detector2_list)
plt.plot(detector3_list)
plt.plot(detector4_list)




#%% testing energy_dist() function

data = energy_dist()

plt.hist(data, 'auto')