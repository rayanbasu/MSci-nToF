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
import matplotlib

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

def DDprimspecmoments(Tion):
# Mean calculation
    a1 = 4.69515
    a2 = -0.040729
    a3 = 0.47
    a4 = 0.81844

      
    mean_shift = a1*Tion**(0.6666666666)/(1.0+a2*Tion**a3)+a4*Tion
    
    # keV to MeV
    mean_shift /= 1e3
    
    mean = 2.4495 + mean_shift
    
    # Variance calculation
    omega0 = 82.542
    a1 = 1.7013e-3
    a2 = 0.16888
    a3 = 0.49
    a4 = 7.946e-4
    
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
def lindec(t, Tmin = 1, Tmax = 10):
    
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
    
    norm = 1 / (2 * np.pi * t_std * E_std)
    
    return energy_gauss * time_gauss


def generate_source(T_prof):
    
    #first calculate the normalisation constant by numerically integrating 
    #over energy and time: (0,100) range for Energy and (0, 500) for time  
    #assumed to be approximately the entire function 
    
    #norm_integral = sp.integrate.nquad(lambda E, t: S(E, t, T_prof), [[0, 100]
    #                                                         ,[0, 500]])[0]
    #norm = 1 / (norm_integral)
    #print(norm)

    #define grid parameters
    n_energy, n_time = (200, 300) #number of grid points
    energies = np.linspace(13, 15, n_energy) #in MeV
    times = np.linspace(1, 420, n_time) #t=100 to t=300

    #generate grid
    E_grid, t_grid = np.meshgrid(energies, times) 
    Z = np.zeros([n_time, n_energy])

    #creating data, Z are the values of the pdf
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            Z[i][j] = S(E_grid[i][j], t_grid[i][j], T_prof)           
    
    #normalise the source
    #Z = norm * Z
    
    #plot surface
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(E_grid, t_grid, Z, cmap=cm.coolwarm)

    #customise plot
    ax.set_ylabel('Time (ps)')
    ax.set_xlabel('Energy (Mev)')
    ax.xaxis.labelpad=10
    #ax.set_zlabel('Probability')
    #ax.set_yticks(np.arange(0,0.125,0.025))
    ax.azim = 0
    ax.elev = 90
    fig.colorbar(surf, shrink=0.5, aspect=15, label = 'Probability')
    #plt.title("Linearly Increasing Temperature")
    
    #for getting rid of z ticks
    ax.zaxis.set_ticklabels([])
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
        
        
    plt.show()
    plt.savefig('source.png', transparent=True)#, bbox_inches='tight')
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
    T[i] = lindec(t[i])

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
particles_num = 100


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
        if Z[i][j] >= np.max(Z)/10000:
            
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


matplotlib.rc('xtick', labelsize=60) 
matplotlib.rc('ytick', labelsize=60) 


#creating fig and ax
nrows = 7
fig, ax = plt.subplots(nrows=nrows, ncols=1)
fig.set_size_inches(35, 95)
#fig.suptitle('Decreasing Temperature', fontsize = 90)
ax[nrows - 1].set_xlabel('Time of arrival (ps)', fontsize = 75)
ax[np.int(nrows/2)].set_ylabel('Flux', fontsize = 75)


#Detector positions:
detector_placements =  np.linspace(0, 2.4, nrows)

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
    ax[j].set_title('detector at ' + np.str(np.round(detector,2))+ 'm',
                                      fontsize = 70)
    print(skew(time_arrive))


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar_ax.tick_params(labelsize=50)

cbar = fig.colorbar(scatter, aspect=100, cax=cbar_ax)
cbar.set_label('Energies (MeV)', fontsize = 80)

#plt.savefig(r'C:\Users\rayan\OneDrive\Documents\Y4\MSci Project\demo.png', dpi=100)    
plt.tick_params(axis='both', which='both', labelsize=60)
fig.savefig('demo1.png', transparent=False, dpi = 200, bbox_inches='tight')
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
#lindec skew 10 to 1
skews1 = [0.10859069778700983, 0.04962548671057384, -0.08915734369166474, -0.22577878565414433, -0.3208775192370092, -0.3732685309866706, -0.39484289358665514, -0.3973285437989948, -0.3890133982379376, -0.37507736405278597, -0.3585769912391012, -0.34125480741387715, -0.3240732556933697, -0.3075403753587294, -0.2919024050836091, -0.27725641778992544, -0.26361599592755836, -0.25094950577664765, -0.23920236484187263, -0.22830992009585052, -0.21820479445962462, -0.20882096452638127, -0.2000959068590366, -0.19197160854570097, -0.18439491798486005, -0.17731752148426674, -0.1706957170081766, -0.16449008737309184, -0.1586651332868286, -0.15318890112729971, -0.09982436979820679, -0.0546836902426904, -0.019795004781466362, -0.008023507621825294, -0.0021171811555239753]
#lininc skew 4.3 to 20
skews = [-0.11712961178649432, 0.0011174611083074792, 0.21494346298230504, 0.3647713262715093, 0.43034887734485855, 0.44463015959300434, 0.4346686074116231, 0.415023401865723, 0.39255655296262637, 0.37028813781697323, 0.3494397996927609, 0.33041318285414967, 0.3132462193461709, 0.29782487160304494, 0.2839809518587611, 0.27153651091178815, 0.26032299672829523, 0.25018850523932656, 0.24099949818008895, 0.23264006301146659, 0.22501019716972545, 0.21802382033290224, 0.21160683523358995, 0.20569536932134236, 0.2002342385070992, 0.19517563183845094, 0.19047799786244918, 0.1861051073144998, 0.18202526632286925, 0.178210656358656, 0.14195348635778954, 0.1122994108845787, 0.08983423369028884, 0.08232298916378877, 0.07856519848092426]
detectors = np.linspace(0,2,30)
detectors = np.append(detectors, [3, 5, 10, 15, 20])

#%%
plt.plot(detectors,skews, label = 'Linearly Increasing')
#plt.plot(detectors,skews1, label='Linearly Decreasing')
plt.xlabel('detector placement (m)')
plt.ylabel('Skewness')
#plt.yticks(fontsize= 8)
plt.grid()
plt.legend()
#plt.title('Constant Temperature (20 keV)')
#plt.title('Linearly Decreasing (35 to 1 keV)')
plt.xlim(xmin=0, xmax = 7.6)
#plt.ylim(ymin =0)
plt.savefig('skews.png', transparent=True, dpi = 800, bbox_inches='tight')


#%%
'''kurtosis analysis'''
#lindec kurts 10 to 1
kurts1 = [0.1333081850596476, 0.06461247871198639, -0.053364661575641126, -0.13656109214398304, -0.16126683651159324, -0.14405636424152002, -0.10631706069133973, -0.062134440733621954, -0.01868866292281046, 0.021009118272287353, 0.056062381222935276, 0.0865313807225685, 0.11285282402354113, 0.1355692123084986, 0.15521167977704575, 0.17225548440959182, 0.1871084228826998, 0.20011321236322965, 0.21155502309317287, 0.2216702092731535, 0.23065458428981556, 0.23867065051535397, 0.24585366842087408, 0.2523166456103776, 0.2581543933066963, 0.26344680752034577, 0.2682615186102719, 0.2726560319153588, 0.2766794605439018, 0.2803739320047631, 0.3121477670737107, 0.33344905370048883, 0.3466363148064353, 0.3504600715285151, 0.3522610665680781]
#lininc kurts 4.3 to 20
kurts = [0.029143783765854803, -0.01683575441376428, -0.06618524132744907, -0.040670645028627384, 0.02004528243564163, 0.07951491089603202, 0.12655124806976792, 0.16116132509314918, 0.1860641657124682, 0.20393411234644443, 0.21682157282057757, 0.22618516396935107, 0.23303910906850822, 0.23808700811165506, 0.2418202027495746, 0.2445854177208635, 0.24663026430102652, 0.24813374779553943, 0.2492268276678633, 0.2500064031300693, 0.25054493339160677, 0.2508971342187869, 0.2511046954465388, 0.25119964343625556, 0.2512067648387917, 0.2511453725534061, 0.2510306055331144, 0.2508743946817833, 0.2506861871075534, 0.25047349379175676, 0.24673481479488046, 0.24146044619606366, 0.2361895135291796, 0.23418664170008663, 0.2331397547700056]
detectors = np.linspace(0,2,30)
detectors = np.append(detectors, [3, 5, 10, 15, 20])



plt.plot(detectors,kurts,  label='Linearly Increasing')
plt.plot(detectors,kurts1,  label='Linearly Decreasing')
plt.xlabel('detector placement (m)')
#plt.yticks(fontsize= 7.1)
plt.ylabel('Kurtosis')
plt.grid()
plt.legend()
#plt.title('Constant Temperature (20 keV)')
#plt.title('Linearly Decreasing (35 to 1 keV)')
plt.xlim(xmin = 0, xmax = 5)
#plt.ylim(ymin = 0, ymax = 0.1)
plt.savefig('kurtosis.png', transparent=True, dpi= 800, bbox_inches='tight')