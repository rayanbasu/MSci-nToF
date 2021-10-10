# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 15:57:34 2021

@author: rayan
"""

import scipy as sp



#%%
burst_duration = 150*10**-15

energy = 14*1.6*10**-19

energy_width = 1*1.6*10**-19

velocity=sp.sqrt((2*energy)/(1.67*10**-27))
#%%

def energy_dist(temperature):
    dist = sp.gaussian(energy, std = energy_width)
    