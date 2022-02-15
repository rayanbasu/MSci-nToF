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

# Importing Aidan's data and organising into arrays
'''names = [                     'time', 
                              'zz_outt',
                              'current',
                              'qtmt',
                              'tohmt', 
                              'tpdvt+tpdv2t+tpdv3t+tpdv4t',
                              'csp',
                              'isub_mhd',
                              'ip34max',
                              'xyield3t',
                              'tten',
                              'tket',
                              'ttfus',
                              'ttradloss p',
                              'ower_kinetic_t ',
                              'voltage',
                              'dt',
                              'tpdvt',
                              'tpdv2t',
                              'tpdv3t',
                              'tpdv4t',
                              'ttbe',
                              'ttbe2',
                              'vload',
                              'tdtign',
                              'tdtegn',
                              'ttrne',
                              'yield_dts',
                              'mat(2)%rho(ix,iy,iz)',
                              'Te(ix,iy,iz)',
                              'Ti(ix,iy,iz)', 
                              'fusion(ix,iy,iz)',
                              'rnec(ix,iy,iz)',
                              'rad_loss(ix,iy,iz)',
                               'vr_av',
                               'dxp',
                               'dyp',
                               'dzp',
                               'tke2t',
                               'yield_dtt/dt',
                               'yield_ddt/dt',
                               'yield_dtBHt/dt', 
                               'yield_ddBHt/dt', 
                               'yield_dts',
                               'yield_dds',
                               'yield_dtBHs',
                               'yield_ddBHs',
                               'BH_ratio',
                               'burn_av_Ti',
                               'dt_mhd',
                               'dt_rad',
                               'dtAlpha',
                               'CouplingCapE',
                               'CouplingCapI']
'''

#self-heating regime
xy00 = pd.read_csv("/Users/ewansaw/Documents/GitHub/MSci-nToF/xy00.dat"
                   ,header = 0, delimiter='  ', engine='python')
xy00 = xy00[['time','burn_av_Ti', 'yield_dtBHt/dt']]
xy00 = xy00.iloc[:,0:].values
xy00 = np.transpose(xy00)
xy00[1] = 1e-3 * xy00[1] #converting eV to keV
xy00[0] = 1e12 * xy00[0] #converting to picoseconds

#ignited hotspot
xy01 = pd.read_csv("/Users/ewansaw/Documents/GitHub/MSci-nToF/xy01.dat"
                   ,header = 0, delimiter='  ', engine='python')
xy01 = xy01[['time','burn_av_Ti', 'yield_dtBHt/dt']]
xy01 = xy01.iloc[:,0:].values
xy01 = np.transpose(xy01)
xy01[1] = 1e-3 * xy01[1] #converting eV to keV
xy01[0] = 1e12 * xy01[0] #converting to picoseconds

#propagating burn
xy03 = pd.read_csv("/Users/ewansaw/Documents/GitHub/MSci-nToF/xy03.dat"
                   ,header = 0, delimiter='  ', engine='python')
xy03 = xy03[['time','burn_av_Ti', 'yield_dtBHt/dt']]
xy03 = xy03.iloc[:,0:].values
xy03 = np.transpose(xy03)
xy03[1] = 1e-3 * xy03[1] #converting eV to keV
xy03[0] = 1e12 * xy03[0] #converting to picoseconds

#chop out values for which neutron yield in data is greater than 1e-4 of max value
dataset = []
for data in ([xy00, xy01, xy03]):
    idx = np.where(data[2] > max(data[2])/50)[0]
    data = np.transpose(data)
    dataset.append(data[idx])

xy00 = np.transpose(dataset[0])
xy01 = np.transpose(dataset[1])
xy03 = np.transpose(dataset[2])


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
    time_contribution = regime[2][index]
    
    #norm = 1 / (np.sqrt(2*np.pi) * E_std)
    
    #NOTE: have taken out normalisation here cause otherwise doesnt work with aiden data
    
    return energy_gauss * time_contribution


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
    ax.azim = 40
    ax.elev = 40
    fig.colorbar(surf, shrink=0.5, aspect=15)
    #plt.title("Linearly Increasing Temperature")

    plt.show()
    return Z, E_grid, t_grid

#%% Plotting temp against time for different regimes

plt.plot(xy00[0], xy00[1], label='Self-heating')
plt.plot(xy01[0], xy01[1], label='Ignited Hotspot')
plt.plot(xy03[0], xy03[1], label='Propagating Burn')
plt.xlabel('Time (ps)')
plt.ylabel('Burn average Ti (keV)')
plt.legend()
#plt.xlim(xmin =0.5e-8 ,xmax = 1e-8)

#%% Plotting neutron yield against time for different regimes

plt.plot(xy00[0], xy00[2], label='Self-heating')
plt.plot(xy01[0], xy01[2], label='Ignited Hotspot')
plt.plot(xy03[0], xy03[2], label='Propagating Burn')
plt.xlabel('Time (ps)')
plt.ylabel('Neutron Yield (Number)')
plt.legend()
#plt.xlim(xmin =0.5e-8 ,xmax = 1e-8)


#%%

Z, E_grid, t_grid = generate_source(xy03)

#scale Z to make it easier to run THIS IS NOT PERMANENT
Z /= 5e25

#%%

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
                 
#creating fig and ax
fig, ax = plt.subplots(nrows=11, ncols=1)
fig.set_size_inches(18, 100)
#fig.suptitle('Decreasing Temperature', fontsize = 90)
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
detector_placements =  np.linspace(0, 5, 11)

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
detectors = np.linspace(0,1,20)
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
plt.plot(detectors,skews,  label='Self-heating')
plt.plot(detectors,skews1,  label='Ignited Hotspot')
plt.plot(detectors,skews3,  label='Propagating Burn')
plt.legend()
plt.xlabel('detector placement (m)')
plt.ylabel('Skewness')
plt.grid()
#plt.title('Constant Temperature (20 keV)')
#plt.title('Linearly Decreasing (35 to 1 keV)')
plt.xlim(xmin = 0, xmax = 0.7)
#plt.ylim(ymin = 0, ymax = 0.1)
#%%

maxindex = np.unravel_index(Z.argmax(), Z.shape)
print(maxindex)
print(t_grid[maxindex])
#%%
# Plotting temp against time for different regimes

#plt.plot(xy00[0], xy00[1], label='Self-heating')
#plt.plot(xy01[0], xy01[1], label='Ignited Hotspot')
plt.plot(xy03[0], xy03[1], label='Propagating Burn')
plt.axvline(x=t_grid[maxindex],linestyle='--', label = 'Bang Time')
plt.xlabel('Time (ps)')
plt.ylabel('Burn average Ti (keV)')
plt.legend()
#plt.xlim(xmin =7310 ,xmax = 7325)
#plt.xlim(xmin =8170 ,xmax = 8200)
plt.xlim(xmin =9000 ,xmax = 9025)




#%%
indexmax = xy03[1].argmax()
print(indexmax)

maxtemp_time = xy03[0][indexmax]
print(maxtemp_time)


indexmax = xy03[2].argmax()
print(indexmax)

bang = xy03[0][indexmax]
print(bang)

#%%
'''kurtosis analysis'''
plt.plot(detectors,kurts,  label='Self-heating')
plt.plot(detectors,kurts1,  label='Ignited Hotspot')
plt.plot(detectors,kurts3,  label='Propagating Burn')
plt.legend()
plt.xlabel('detector placement (m)')
plt.ylabel('Kurtosis')
plt.grid()
#plt.title('Constant Temperature (20 keV)')
#plt.title('Linearly Decreasing (35 to 1 keV)')
plt.xlim(xmin = 0, xmax = 0.5)
#plt.ylim(ymin = 0, ymax = 0.1)


np.save()


#%% data
detectors = np.linspace(0,1,20)
detectors = np.append(detectors, [3, 5, 10, 15, 20])

kurts = [0.22501401669765686, 0.1795688516613665, 0.0737452581244864, -0.03668159939530247, -0.11973655724595433, -0.1714652432075643, -0.19969058268897655, -0.21312914813496597, -0.21809207496708316, -0.2184927691770664, -0.2165896176510782, -0.2136360358091598, -0.21030823834739598, -0.20695950475885772, -0.20376500545199283, -0.20080283893951556, -0.19809906530550458, -0.195652655164547, -0.19344926005059104, -0.19146875101904026, -0.17141020617984282, -0.1691769312616187, -0.16805742981163796, -0.16779508152639844, -0.16768504260258865]
kurts1 = [0.4718010272373272, 0.3894824627426847, 0.20457070588202964, 0.0186047951131596, -0.11479073903645132, -0.19286872631655694, -0.23171364814095874, -0.24716144815306684, -0.249978738486496, -0.24651273917256322, -0.2402510865383265, -0.2330265434377563, -0.22576159115046357, -0.21889133502330926, -0.21259512372758627, -0.20692133670644886, -0.2018540109771938, -0.19734809981745594, -0.19334785873040694, -0.1897961667229051, -0.15632944211213262, -0.15317002111038525, -0.15182275086731112, -0.1515750221547889, -0.15148933000539255]
kurts3 = [0.4763944324262366, 0.33248648308999584, 0.06704932051165713, -0.1258514309282881, -0.21652551113075758, -0.24447809732297676, -0.24369984223513086, -0.23216516861915215, -0.21785698057456981, -0.20395388450412666, -0.19154018143160467, -0.18084002980182357, -0.1717509028866271, -0.1640671476000315, -0.15756987130184985, -0.15205995416863027, -0.14736736576554765, -0.14335107716109308, -0.13989567077493792, -0.13690718596511298, -0.11186944886390293, -0.10988236911533056, -0.10914729705678683, -0.1090483020622317, -0.109026303589006]


skews = [-0.4294601956670631, -0.40307986068125123, -0.34247532598225183, -0.27049035393813115, -0.20431964186667903, -0.15064720440898177, -0.10956101177189607, -0.07876677854958956, -0.05571802262625284, -0.03831467625856898, -0.024994537823670403, -0.01464092815186523, -0.006465321136206284, 8.979904862042397e-05, 0.0054218165084965325, 0.009817242657093368, 0.013485348827698241, 0.0165810608423513, 0.019220615506080743, 0.021492359683102907, 0.04250615824667899, 0.04555259991198031, 0.04766629764040206, 0.048344174015896954, 0.04867875770655405]
skews1 = [-0.4610724980609337, -0.4252863776037561, -0.346340617974957, -0.2559335438146068, -0.17622746919740187, -0.1144355027869699, -0.06932005569211142, -0.037117417076838466, -0.014194288183101715, 0.0022450509547593113, 0.014179751360336938, 0.022967725103987032, 0.029533457559052706, 0.034508752627335354, 0.038329544294876707, 0.04130034911697832, 0.043636763970317245, 0.0454935339895835, 0.04698322938003079, 0.04818880419279106, 0.05425536622321592, 0.054015736086421066, 0.05364768787853966, 0.05349791619099637, 0.0534189047066732]
skews3 =[-0.4198065000133934, -0.3612167224750987, -0.2466533750293474, -0.1391550369446025, -0.062320865756193805, -0.012890205710694748, 0.017977310125458473, 0.037303329688534614, 0.04958408402704474, 0.05752833768676158, 0.06275359469164248, 0.0662369823819045, 0.06858040774839315, 0.07016287776163788, 0.07122833980225869, 0.07193711255942398, 0.07239659910235052, 0.07268001989173334, 0.07283808189626417, 0.07290639612794242, 0.06912051938079214, 0.06758559018726464, 0.06633070993998993, 0.06589684494729478, 0.06567744137187398]

