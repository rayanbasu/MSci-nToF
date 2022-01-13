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
'''
t_0 = 210 #bang time (in picoseconds)
burn_time = 100 #choose burn time of 100ps, take this to be equal to FWHM for now 
t_std = burn_time / 2.35482 #converting FWHM to sigma


#linearly increasing temperature from 4.3keV to 15keV over burn time = 100ps
def lininc(t, Tmin = 4.3, Tmax = 15):    
    
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
def lindec(t, Tmin = 3, Tmax = 10):
    
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
    
    
#linearly increasing then decreasing temperature over burn time = 100ps.
def incdec(t, Tmin_1 = 2, Tmax = 15, Tmin_2 = 2):
    
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
def const_temp(t, T = 15):
    return T #in keV


''' Defining UN-NORMALISED Source function S(E,t):
Need to also define the temperature profile being used!! (T_prof argument)
Source is normalised in the generate_source() function
'''
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
    #over energy and time: (0,100) range for Energy and (0, 500) for time  
    #assumed to be approximately the entire function 
    
    norm_integral = sp.integrate.nquad(lambda E, t: S(E, t, T_prof), [[0, 100]
                                                             ,[0, 500]])[0]
    norm = 1 / (norm_integral)
    print(norm)

    #define grid parameters
    n_energy, n_time = (150, 150) #number of grid points
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
    return Z, E_grid, t_grid

#%%
'''Testing generate_source() function: 
   this should plot the source function and return the source data'''
Z, E_grid, t_grid = generate_source(lininc)


#%%
''' Validating temperature profiles and seeing how std varies with profile '''
t = np.linspace(0,1000,1000)
T = np.zeros(len(t))
for i in range(len(t)):
    T[i] = incdec(t[i])

sigma = np.sqrt(DTprimspecmoments(T)[1])

plt.plot(t, T)    
#plt.plot(t, sigma)
plt.title('Temperature against Time')
plt.xlabel('Time (ps)')
plt.ylabel('Temperature (keV)')

#%%
'''
This is multiplied by the pdf distribution to give the number of particles for
each time and energy
'''
particles_num = 5000


#Empty arrays to record data:
time_emitted = np.array([]) #time particle was emitted
velocities = np.array([]) #particle velocities
number_of_particles = np.array([]) #number of respective particles with above values
energies = [] #turn into array!!!

#Just a dataframe
particle_df = pd.DataFrame(columns = ['time emitted', 'energy', ' number of particles'])


#Goes thorugh the 2D arrays containing each energy and time and finds 
#the number of particles for each of those values
#i.e. only propagate grid points with values >= 1/1000th of max pdf value
for i in range(len(Z)):
    for j in range(len(Z[0])):
        if Z[i][j] >= np.max(Z)/1000:
            
            #time in picoseconds
            time_emitted = np.append(time_emitted, t_grid[i][j])
            
            #energies in MeV
            energies.append(E_grid[i][j])
            
            #velocities in ms^-1
            velocities = np.append(velocities, np.sqrt(E_grid[i][j] * 1.6e-13 * 2
                                      / 1.67e-27))
            
            #save integer number of particles
            num = np.round(particles_num  * Z[i][j]) # may have a problem here with rounding
            number_of_particles = np.append(number_of_particles, num)
            
          
#creating fig and ax
fig, ax = plt.subplots(nrows=11, ncols=1)
fig.set_size_inches(18, 100)
fig.suptitle('Decreasing Temperature', fontsize = 90)
ax[10].set_xlabel('time of arrival (ps)', fontsize = 70)
ax[5].set_ylabel('flux', fontsize = 70)
            
''' Ewan's attempt to plot 3d plots similar to vlad paper (ignore!!!)

#detector distances from source (in metres)
detectors = np.arange(0.02, 100, 10)

#to record times particles arrive at detectors
time_arrive = np.array([])

for i in range(len(number_of_particles)):
    print(time_emitted[i] + detectors/velocities[i]*1e12)
    time_arrive = np.append(time_arrive, 
                            time_emitted[i] + detectors/velocities[i]*1e12) #in ps

time_arrive = time_arrive.reshape((len(number_of_particles), len(detectors)))
time_arrive = time_arrive.transpose()


#%%
#plotting 3d temporal and distance spread

fig = plt.figure()
ax = fig.gca(projection='3d')

#for colouring plots
#def cc(arg):
   # return mcolors.to_rgba(arg, alpha=0.6)

xs = time_arrive #t variable
verts = []
zs = detectors #detector positions
for i in range(len(detectors)):
    ys = number_of_particles #number of particles from given squares
    verts.append(list(zip(xs[i], ys)))

poly = PolyCollection(verts) #, facecolors=[cc('r'), cc('g'), cc('b'), cc('y')])


poly.set_alpha(0.8)

ax.add_collection3d(poly, zs, zdir='y')

ax.set_xlabel('Time (ps)')
ax.set_xlim3d(0, 2000000)
ax.set_ylabel('Position (m)')
ax.set_ylim3d(0, 102)
ax.set_zlabel('Flux')
ax.set_zlim3d(0,20000)

#ax.set_yticks(np.arange(0,0.125,0.025))

plt.show()

#need to bin these results so that can show temporal spread !!! (fix)

'''

#Detector positions:
detector_placements =  np.linspace(0, 1, 11)

#Time it arrives at the detector is recorded in this array
for detector in detector_placements:
    time_arrive = []
    
    for i in range(len(number_of_particles)):
        time_arrive.append(time_emitted[i] + detector 
                           / velocities[i] * 1e12) #in ps
    
    #Plotting the number of particles arriving at each time
    scatter = ax[np.int(detector*10)].scatter(time_arrive,number_of_particles, 
                                              c = energies, cmap = cm.plasma)

    #fig.colorbar(scatter, shrink=1, aspect=15)
    ax[np.int(detector*10)].set_title('detector at ' + np.str(np.around(detector,1))+ 'm',
                                      fontsize = 30)
    print(skew(time_arrive))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar_ax.tick_params(labelsize=30)

cbar = fig.colorbar(scatter, aspect=100, cax=cbar_ax)
cbar.set_label('Energies (MeV)', fontsize = 70, rotation=270)
#fig.savefig(r'C:\Users\rayan\OneDrive\Documents\Y4\MSci Project\lininc.png', dpi=100)    

#%%
detector = 0
time_arrive = []
    
#times of arrivals at given detector distance
for i in range(len(number_of_particles)):
    time_arrive.append(time_emitted[i]+detector/velocities[i]*1e12)
    
    
#Plotting the number of particles arriving at each time
scatter = plt.scatter(time_arrive,number_of_particles/max(number_of_particles),
                       c = energies, cmap = cm.plasma)

''' NOTE: this is not what the actual detector sees. This code plots each discrete point
from our source grid as coloured points wrt energies on the graph - i.e. the 'flux'
shown is not the cumulative flux that the detector receives at the particular time, 
but rather the discrete fluxes of each particular grid point that may have been 
plotted on top of each other. 

For an accurate representation of the what the detector sees, we need to bin this data 
according to an appropriate resolution of arrival time and plot the cumulative flux.
'''

#change size of markers!!

plt.colorbar(scatter, shrink=1, aspect=15, label = 'Energies (MeV)')
plt.title('detector at ' + np.str(np.around(detector,1))+ 'm', fontsize = 10)
plt.xlabel('Time of Arrival (ps)')
plt.ylabel('Normalised Flux')
#fig.savefig(r'C:\Users\rayan\OneDrive\Documents\Y4\MSci Project\lininc2.6.png', dpi=100)    
#%%
''' plotting skewness wrt detector positions'''
skews=[]
detectors=[0] #np.linspace(0,3,20)

for detector in detectors:
    time_arrive = []
        
    for i in range(len(number_of_particles)):
        time_arrive.append(time_emitted[i] + detector 
                           / velocities[i] * 1e12) #in ps
        
    skewness = np.array([])
    for i in range(len(number_of_particles)):
        particles = np.int(number_of_particles[i]) #may have a problem here with rounding!!
        print([i,detector])
        for j in range(particles):
            skewness = np.append(skewness,time_arrive[i])
    
    print(skew(skewness))
    #think about use of bias here, does it change much in our results?
    skews.append(skew(skewness))


plt.plot(detectors,skews, 'x')
plt.xlabel('detector placement (m)')
plt.ylabel('Skewness')
plt.grid()
#plt.title('Constant Temperature (20 keV)')
#plt.title('Linearly Decreasing (35 to 1 keV)')
#plt.xlim(xmax = 2)
