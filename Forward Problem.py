# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 15:57:34 2021

@author: rayan
"""

import scipy as sp
from scipy.signal import gaussian


#%%
burst_duration = 150*10**-15


#Energy in MeV
energy = 14

#Energy width in MeV
energy_width = 1

velocity=sp.sqrt((2*energy)/(1.67*10**-27))
#%%

def energy_dist():
    dist = gaussian(10, std = energy_width)
    return dist+energy
    
    