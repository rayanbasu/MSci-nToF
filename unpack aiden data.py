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
from scipy.stats import skew, kurtosis
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
import os
import glob

#self-heating regime
xy00 = pd.read_csv("/Users/ewansaw/Documents/GitHub/MSci-nToF/xy00.dat"
                   ,header = 0, delimiter='  ', engine='python')
xy00 = xy00[['time','burn_av_Ti', 'yield_dtBHt/dt']]
xy00 = xy00.iloc[:,0:].values
xy00 = np.transpose(xy00)

#ignited hotspot
xy01 = pd.read_csv("/Users/ewansaw/Documents/GitHub/MSci-nToF/xy01.dat"
                   ,header = 0, delimiter='  ', engine='python')
xy01 = xy01[['time','burn_av_Ti', 'yield_dtBHt/dt']]
xy01 = xy01.iloc[:,0:].values
xy01 = np.transpose(xy01)

#propagating burn
xy03 = pd.read_csv("/Users/ewansaw/Documents/GitHub/MSci-nToF/xy03.dat"
                   ,header = 0, delimiter='  ', engine='python')
xy03 = xy03[['time','burn_av_Ti', 'yield_dtBHt/dt']]
xy03 = xy03.iloc[:,0:].values
xy03 = np.transpose(xy03)

#non-igniting dataset
no_ign = pd.read_csv("/Users/ewansaw/Documents/GitHub/MSci-nToF/NoAlphaHeating.dat"
                   ,header = 0, delimiter='  ', engine='python')
no_ign = no_ign[['time','burn_av_Ti', 'yield_dtBHt/dt']]
no_ign = no_ign.iloc[:,0:].values
no_ign = np.transpose(no_ign)

#s=0.9
s09 = pd.read_csv("/Users/ewansaw/Documents/GitHub/MSci-nToF/s=0.9.dat"
                   ,header = None, delimiter=' ', engine='python')
s09 = s09.iloc[:,0:].values

#s=1.0
s10 = pd.read_csv("/Users/ewansaw/Documents/GitHub/MSci-nToF/s=1.0.dat"
                   ,header = None, delimiter=' ', engine='python')
s10 = s10.iloc[:,0:].values

#s=1.1
s11 = pd.read_csv("/Users/ewansaw/Documents/GitHub/MSci-nToF/s=1.1.dat"
                   ,header = None, delimiter=' ', engine='python')
s11 = s11.iloc[:,0:].values

#s=1.2
s12 = pd.read_csv("/Users/ewansaw/Documents/GitHub/MSci-nToF/s=1.2.dat"
                   ,header = None, delimiter=' ', engine='python')
s12 = s12.iloc[:,0:].values

#s=1.3
s13 = pd.read_csv("/Users/ewansaw/Documents/GitHub/MSci-nToF/s=1.3.dat"
                   ,header = None, delimiter=' ', engine='python')
s13 = s13.iloc[:,0:].values

#s=1.4
s14 = pd.read_csv("/Users/ewansaw/Documents/GitHub/MSci-nToF/s=1.4.dat"
                   ,header = None, delimiter=' ', engine='python')
s14 = s14.iloc[:,0:].values

#normalise sacale factor curves (correcting with peak compression times)
s09[0] = (s09[0] - 8.14003265e-09)/0.9
s10[0] = (s10[0] - 9.10059761e-09)
s11[0] = (s11[0] - 1.01403463e-08)/1.1
s12[0] = (s12[0] - 1.11404495e-08)/1.2
s13[0] = (s13[0] - 1.21800054e-08)/1.3
s14[0] = (s14[0] - 1.3220478e-08)/1.4


datas = [xy00, xy01, xy03, no_ign, s09, s10, s11, s12, s13, s14]
norms = np.zeros(len(datas))

#rearranging the form of extra data
for data in datas[4:]:
    data[[2, 1], :] = data[[1, 2], :]
    
''' with this cell'''
for data in datas:
    data[1] = 1e-3 * data[1] #converting eV to keV
    data[0] = 1e12 * data[0] #converting to picoseconds
   
''' need to swap this cell'''
# integrate the yield against time to obtain the constants of normalisation 
for i in range(len(datas)):
    norms[i] = sp.integrate.simps(datas[i][2], datas[i][0])
    

#normalising
for i in range(len(datas)):
    datas[i][2] /= norms[i]

#%%

#chop out values for which neutron yield in data is greater than 1e-4 of max value
dataset = []
for data in datas:
    idx = np.where(data[2] > max(data[2])/100)[0]
    data = np.transpose(data)
    dataset.append(data[idx])
  
for i in range(len(datas)):
    datas[i] = np.transpose(dataset[i])

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


''' Defining UN-NORMALISED Source function S(E,t):
Need to also define the temperature profile being used!! (T_prof argument)
Source is normalised in the generate_source() function
'''

def S(E, t, regime, index):  
    #t_0 = regime[0][np.argmax(regime[2])] #bang time (in picoseconds) 
    E_0, E_var = DTprimspecmoments(regime[1][index])
    E_std = np.sqrt(E_var)
    #print(E_std)
    
    #gaussian in energy (taken in units of MeV)
    energy_gauss = np.exp(-(E - E_0)**2 / (2 * E_std**2))
    
    #time contribution 
    time_contribution = regime[2][index] #note: these are in units 1/time
    
    norm = 1 / (np.sqrt(2*np.pi) * E_std)
    
    #NOTE: have taken out normalisation here cause otherwise doesnt work with aiden data
    
    return norm * energy_gauss * time_contribution #units: probability/(time * energy)


#only for existing dataset (should already be normalised)
def generate_source(regime):
    
    #first calculate the normalisation constant by numerically integrating 
    #over energy and time: (0,100) range for Energy and (0, 500) for time  
    #assumed to be approximately the entire function 
    
    #norm_integral = sp.integrate.nquad(lambda E, t: S(E, t, T_prof), [[0, 100]
    #                                                         ,[0, 500]])[0]
    #norm = 1 / (norm_integral)
    #print(norm)

    #define grid parameters
    n_energy, n_time = (100, len(regime[0])) #number of grid points
    energies = np.linspace(13, 15, n_energy) #in MeV
    times = regime[0] 


    #generate grid
    E_grid, t_grid = np.meshgrid(energies, times) 
    Z = np.zeros([n_time, n_energy])

    #creating data, Z are the values of the pdf
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            Z[i][j] = S(E_grid[i][j], t_grid[i][j], regime, index = i)
            
            #get rid of small values
            #if Z[i][j] == 0:
             #   Z[i][j] = float('nan')
            
    #normalise the source
    #Z = norm * Z
    
    #plot surface
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(E_grid, t_grid, Z, cmap=cm.coolwarm)

    #customise plot
    ax.set_ylabel('time (ps)')
    ax.set_xlabel('energy (Mev)')
    #ax.set_ylim3d(7000,7500) #for xy00
    #ax.set_ylim3d(8000,8500) #for xy01
    #ax.set_ylim3d(8850,9100) #for xy03
    #ax.set_zlabel('pdf')
    #ax.set_yticks(np.arange(0,0.125,0.025))
    ax.azim = 0
    ax.elev = 90
    fig.colorbar(surf, shrink=0.5, aspect=15)
    #plt.title("Linearly Increasing Temperature")

    plt.show()
    return Z, E_grid, t_grid

#%% Plotting temp against time for different regimes

plt.plot(xy00[0], xy00[1], label='Self-heating')
plt.plot(xy01[0], xy01[1], label='Ignited Hotspot')
plt.plot(xy03[0], xy03[1], label='Propagating Burn')
plt.plot(no_ign[0], no_ign[1], label='Non-Igniting')
plt.axvline(x=8949.27954, linestyle='--', color = 'red', linewidth=1)
plt.axvline(x=9012.51163, linestyle='--', color = 'green', linewidth=1)
plt.axvline(x=8179.131149, linestyle='--', color = 'orange', linewidth=1)
plt.axvline(x=7319.75147, linestyle='--', color = 'blue', linewidth=1)

#plt.plot(datas[4][0], datas[4][1], label='s=0.9')
#plt.plot(datas[5][0], datas[5][1], label='s=1.0')
#plt.plot(datas[6][0], datas[6][1], label='s=1.1')
#plt.plot(datas[7][0], datas[7][1], label='s=1.2')
#plt.plot(datas[8][0], datas[8][1], label='s=1.3')
#plt.plot(datas[9][0], datas[9][1], label='s=1.4')
plt.xlabel('Time (ps)')
#plt.xlabel('Time, (t-t_0)/S (ps)')
plt.ylabel('Burn average Ti (keV)')
plt.legend()
plt.grid()
plt.xlim(xmin =6000 ,xmax = 10000)
#plt.xlim(xmin =-100 ,xmax =300)
plt.savefig('temps.png',dpi=800, transparent=True)

#%% Plotting neutron yield against time for different regimes

#plt.plot(datas[0][0], datas[0][2], label='Self-heating')
#plt.plot(datas[1][0], datas[1][2], label='Ignited Hotspot')
#plt.plot(datas[2][0], datas[2][2], label='Propagating Burn')
#plt.plot(datas[3][0], datas[3][2], label='Non-Igniting')
plt.plot(datas[4][0], datas[4][2], label='s=0.9')
plt.plot(datas[5][0], datas[5][2], label='s=1.0')
plt.plot(datas[6][0], datas[6][2], label='s=1.1')
plt.plot(datas[7][0], datas[7][2], label='s=1.2')
plt.plot(datas[8][0], datas[8][2], label='s=1.3')
plt.plot(datas[9][0], datas[9][2], label='s=1.4')
plt.xlabel('Time (ps)')
#plt.xlabel('Time, (t-t_o)/S (ps)')
plt.ylabel('Neutron Yield per unit time')
plt.legend()
#plt.yscale('log')
plt.grid()
#plt.xlim(xmin =6000 ,xmax = 9500)
#plt.ylim(ymin =1 ,ymax = 1e30)
#plt.xlim(xmin =-100 ,xmax =300)
plt.savefig('yield.png',dpi=800, transparent=True)


    #%%
Z, E_grid, t_grid = generate_source(datas[3])

source_yield = 100
Z = source_yield * Z


#Empty arrays to record data:
time_emitted = np.array([]) #time particle was emitted
velocities = np.array([]) #particle velocities
number_of_particles = np.array([]) #number of respective particles with above values
energies = [] #turn into array!!!

#Goes thorugh the 2D arrays containing each energy and time and finds 
#the number of particles for each of those values
#i.e. only propagate grid points with values >= 1/1000th of max pdf value
for i in range(len(Z)):
    for j in range(len(Z[0])):
        if Z[i][j] > 0:
            
            #time in picoseconds
            time_emitted = np.append(time_emitted, t_grid[i][j])
            
            #energies in MeV
            energies.append(E_grid[i][j])
            
            #velocities in ms^-1
            velocities = np.append(velocities, np.sqrt(E_grid[i][j] * 1.6e-13 * 2
                                      / 1.67e-27))
            
            #save integer number of particles
            num = Z[i][j]
            number_of_particles = np.append(number_of_particles, num)
                 
            
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

#creating fig and ax
nrows = 9
fig, ax = plt.subplots(nrows=nrows, ncols=1)
fig.set_size_inches(23, 70)
#fig.suptitle('Decreasing Temperature', fontsize = 90)
ax[nrows - 1].set_xlabel('Time of arrival (ps)', fontsize = 70)
ax[np.int(nrows/2)].set_ylabel('Flux', fontsize = 70)


#Detector positions:
detector_placements =  np.linspace(0, 1, nrows)

#Time it arrives at the detector is recorded in this array
for j in range(len(detector_placements)):
    detector = detector_placements[j]
    print(detector)
    time_arrive = []
    
    for i in range(len(number_of_particles)):
        time_arrive.append(time_emitted[i] + detector 
                           / velocities[i] * 1e12) #in ps
    
    #Plotting the number of particles arriving at each time
    scatter = ax[j].scatter(time_arrive,number_of_particles, 
                                              c = energies, cmap = cm.plasma)

    #fig.colorbar(scatter, shrink=1, aspect=15)
    ax[j].set_title('detector at ' + np.str(detector)+ 'm',
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
kurts=[]
detectors = np.linspace(0,2,30)
detectors = np.append(detectors, [3, 5, 10, 15, 20])

for detector in detectors:
    print(detector)
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
    kurts.append(kurtosis(skewness))

#%%
'''skewness analysis'''
plt.plot(detectors,skews, label='Self-heating')
#plt.plot(detectors,skews1,  label='Ignited Hotspot')
#plt.plot(detectors,skews3,  label='Propagating Burn')
#plt.plot(detectors,ign_skews,  label='Non-Igniting')
plt.legend()
plt.xlabel('detector placement (m)')
plt.ylabel('Skewness')
plt.grid()
#plt.title('Constant Temperature (20 keV)')
#plt.title('Linearly Decreasing (35 to 1 keV)')
plt.xlim(xmin = 0, xmax = 3)
#plt.ylim(ymin = 0, ymax = 0.05)
plt.savefig('dataskewness.png', transparent=True, dpi = 800, bbox_inches='tight')

#%%
Z, E_grid, t_grid = generate_source(datas[0])

maxindex = np.unravel_index(Z.argmax(), Z.shape)
print(maxindex)
print(t_grid[maxindex])
#%%
# Plotting temp against time for different regimes

plt.plot(xy00[0], xy00[1], label='Self-heating')
#plt.plot(xy01[0], xy01[1], label='Ignited Hotspot')
#plt.plot(xy03[0], xy03[1], label='Propagating Burn')
#plt.plot(no_ign[0], no_ign[1], label='Non-Igniting')
plt.axvline(x=t_grid[maxindex],linestyle='--', label = 'Bang Time', color = 'black')
plt.xlabel('Time (ps)')
plt.ylabel('Burn average Ti (keV)')
plt.legend()
plt.grid()
plt.xlim(xmin =7220 ,xmax = 7420)
#plt.xlim(xmin =8070 ,xmax = 8270)
#plt.xlim(xmin =8900 ,xmax = 9100)

plt.savefig('bangtime.png', transparent=True, dpi = 800, bbox_inches='tight')

'''
#%%
indexmax = xy03[1].argmax()
print(indexmax)

maxtemp_time = xy03[0][indexmax]
print(maxtemp_time)


indexmax = xy03[2].argmax()
print(indexmax)

bang = xy03[0][indexmax]
print(bang)
'''
#%%
'''kurtosis analysis'''
plt.plot(detectors,kurts, label='Self-heating')
plt.plot(detectors,kurts1,  label='Ignited Hotspot')
plt.plot(detectors,kurts3,  label='Propagating Burn')
plt.plot(detectors,ign_kurts, label='Non-Igniting')
plt.xlabel('detector placement (m)')
plt.yticks(fontsize= 6.6)
plt.ylabel('Kurtosis', fontsize = 9)
plt.grid()
plt.legend()
#plt.title('Constant Temperature (20 keV)')
#plt.title('Linearly Decreasing (35 to 1 keV)')
plt.xlim(xmin = 0, xmax = 3)
#plt.ylim(ymin = 0, ymax = 0.1)
plt.savefig('kurtosis.png', transparent=True, dpi = 800)



#%% data
detectors = np.linspace(0,2,30)
detectors = np.append(detectors, [3, 5, 10, 15, 20])

kurts = [-0.7186869687684134, -0.7325543236005241, -0.754452347071414, -0.7667867031877726, -0.7691535090567965, -0.7665576434816725, -0.7624205588046133, -0.7582600420032573, -0.7545906910168432, -0.7515116537912205, -0.7489757223349169, -0.7468969791500899, -0.7451897324839449, -0.7437803612333815, -0.7426091442359493, -0.7416287829820605, -0.740802146567455, -0.7401001209732772, -0.7394998132183948, -0.7389831315813216, -0.7385356904222315, -0.7381459741624723, -0.7378047009945776, -0.7375043380291317, -0.7372387305068546, -0.7370028168242304, -0.7367924082460311, -0.736604017565992, -0.736434724991037, -0.7362820724915169, -0.7351003470286819, -0.7344918919795953, -0.7342434960660791, -0.734201203258857, -0.7341877062670501]
kurts1 = [-0.6832849994498464, -0.718128236951229, -0.7742476272900496, -0.8028340658056297, -0.8060099415111996, -0.798016883881496, -0.7872691826144655, -0.7770707016255352, -0.7683605535429803, -0.7612009477173145, -0.7553875606363163, -0.7506708233428512, -0.7468263377153694, -0.7436708258080209, -0.7410601050160364, -0.7388823290577284, -0.7370509996226091, -0.7354990545651927, -0.7341742210304538, -0.7330354752890704, -0.7320503800235079, -0.7311930887045217, -0.7304428478136975, -0.7297828679967107, -0.7291994683763092, -0.7286814236460146, -0.7282194623987546, -0.7278058789027475, -0.7274342305282522, -0.7270991002771092, -0.7245013244593186, -0.7231534914731794, -0.7225912710787079, -0.7224911449808795, -0.7224575636433435]
kurts3 = [-0.780733595392733, -0.8149548934572008, -0.8545105284718417, -0.8594589561507213, -0.8489979678746202, -0.8366047950504196, -0.8261421539451761, -0.8180253578541947, -0.8118481028858295, -0.8071320638929413, -0.8034908441089139, -0.8006402228863747, -0.7983765615130598, -0.7965542873246938, -0.7950685496586662, -0.793842947187092, -0.7928210751234275, -0.791960731637309, -0.7912299274021679, -0.7906041110624722, -0.7900642178909543, -0.7895952805210547, -0.789185427589334, -0.7888251532172972, -0.7885067778504187, -0.7882240458934846, -0.787971822259987, -0.7877458612324677, -0.7875426287419387, -0.7873591645059284, -0.7859216350636968, -0.7851415867287619, -0.7847793715525837, -0.7847016611233513, -0.7846709139086743]
ign_kurts = [-0.9373024338606855, -0.933215743765409, -0.925868047455018, -0.91816343582492, -0.9121741358376099, -0.9085678063611757, -0.907064776279769, -0.9070520970465359, -0.9079450849429076, -0.909309589992835, -0.9108632551642919, -0.9124386205402177, -0.9139446718190727, -0.9153375591602804, -0.9166009699104922, -0.9177338370308283, -0.9187429663193121, -0.9196387597804443, -0.9204328172555472, -0.921136649435423, -0.9217610350373082, -0.9223157432067541, -0.9228094569933818, -0.9232498025294107, -0.9236434293034068, -0.9239961108879511, -0.9243128494606303, -0.92459797553339, -0.9248552389075275, -0.9250878894330405, -0.9268802634741613, -0.9277150649039427, -0.9279329157362239, -0.9279234945161341, -0.9279032059566559]

skews = [-0.11029493631167515, -0.09672447666619059, -0.060688068080714475, -0.025219917972993618, 0.000234378431042061, 0.016324442366979844, 0.026012891109832337, 0.031737206378105134, 0.035073806466709656, 0.036973100257622675, 0.03799956762694134, 0.03849056932070212, 0.038650610847432924, 0.03860570375593736, 0.03843472456755342, 0.0381877279356204, 0.03789682566323327, 0.03758277081418132, 0.03725901693864395, 0.03693426513670018, 0.03661408858772499, 0.0363019841980446, 0.03600006250739332, 0.03570950562210627, 0.03543087425578209, 0.0351643153448329, 0.034909703370911926, 0.03466673703178557, 0.03443500553964883, 0.034214034111604306, 0.03192249407065537, 0.029822229321014086, 0.028116710901880136, 0.027526922217946476, 0.02722842849578709]
skews1 = [-0.08316702104404336, -0.06416861500331637, -0.0316045446676694, -0.00454321021184291, 0.012502144619325398, 0.022102625427700594, 0.027254037381510058, 0.029915783748679953, 0.031202956770313785, 0.0317289376554077, 0.031831798326189195, 0.03170052447425181, 0.03144251867208424, 0.031119694568725632, 0.030768075333914888, 0.030408659522738826, 0.030053579308159988, 0.02970966504863592, 0.02938054859141123, 0.029067925399860497, 0.02877232269723507, 0.028493572206876962, 0.02823110337848523, 0.02798412599256692, 0.02775174378090384, 0.02753302463155987, 0.02732704326735177, 0.027132906415390658, 0.02694976681742814, 0.026776830165123455, 0.02505969165699392, 0.023573206640467432, 0.022409227808112494, 0.02201410690272443, 0.021815450413372645]
skews3 =[-0.12034415990414353, -0.07909068250403481, -0.021775709012544834, 0.012682397464515134, 0.028309777081698254, 0.03450427364528509, 0.03648881667723269, 0.036644775122606964, 0.03602723968459921, 0.03510908457578812, 0.03410493527892933, 0.03311171287973467, 0.03217128699617362, 0.03129935377830688, 0.0304991611396144, 0.029768191051074304, 0.02910146137558158, 0.02849316087923725, 0.027937445360988584, 0.027428806173714113, 0.026962220136915217, 0.026533188913740673, 0.026137724351340232, 0.025772309426199753, 0.02543385022613697, 0.025119626826467824, 0.02482724684761766, 0.024554603315194205, 0.024299837316636243, 0.024061305357125157, 0.02178067798172367, 0.01990180294314304, 0.018473220608944126, 0.017994918159099944, 0.017755540739007487]
ign_skews = [-0.03183551686760398, -0.029179368618094677, -0.026061513547704786, -0.02286885393989737, -0.01986828664000202, -0.017186397399260554, -0.014851142848450231, -0.012839904422510694, -0.011111458172040864, -0.009622168421819809, -0.008332551919198, -0.00720912858412072, -0.0062243083066891155, -0.005355612396125534, -0.004584775013736723, -0.00389692466612002, -0.003279900586560149, -0.002723703123521559, -0.002220059294510173, -0.001762081545665075, -0.0013439999144587043, -0.0009609512877941835, -0.0006088128867272622, -0.00028407001449086693, 1.628957640982593e-05, 0.0002948605545975674, 0.000553888598950265, 0.0007953261573331224, 0.0010208780276572337, 0.0012320388333505788, 0.0032544470114379757, 0.00493279711243858, 0.006220523482035248, 0.006654296968631466, 0.006871916721154883]

#%% check skew of initial neutron yield 


