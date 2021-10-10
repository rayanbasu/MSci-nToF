# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:42:05 2020

@author: rayan
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

#%%
#1D function to be integrated
def psi_0(x):
    return np.exp(-(x**2))/np.sqrt(np.pi)

#3D first excited state function
def h(x,y,z):
    return ((psi_0(x)*psi_0(y)*(y**2+x**2))*psi_0(z))

#3D ground state excited function
def f(x,y,z):
    return ((psi_0(x)*psi_0(y)*psi_0(z)))

#%% Newton-Coates Method. f is the function, start and end are the bounds for the integration and episilon is self-explanatory
#Takes input 'T' for Trapezium and 'S' for Simpson
def NC(f, start,end, epsilon, method):
    #Check is upper bound>lower bound
    assert end>start
    #S_1 is the initial value for Simpson's method. It will be used to compute epsilon
    #Used negative number since integrals are positive if start<end
    S_1 = -1
    
    #I_1 is the inital value for Trapezoidal
    
    I_1 = (end-start)/2*(f(end)+f(start))
    
    
    for Num in range(1,15):
        
        #N is the number of data points. It is in the form 2**x+1 since we want spacing to decrease by a factor of 2
        N=2**Num+1
        
        #h is spacing
        h=(end-start)/(N-1)

        
        #Sum is the sum of the new points to compute
        #This method increases efficiency
        Sum = 0
        for i in range(1, np.int((N+1)/2)):
                 #print(start+(2*i-1)*h)
            Sum+=f(start+(2*i-1)*h)
        
        #New integral for trapezoidal method is computed
        I_2 = 1/2*I_1 +h*Sum
        
        #If trapezoidal method is used, check if epsilon condition is met, otherwise set I_1=I_2
        if method == 'T':
            
            if np.abs((I_1-I_2)/I_1)<epsilon:

                return I_2
            
            else:

                I_1=copy.copy(I_2)
                
 
        #Same for if Simpson's method is used     
        elif method == 'S':

            S_2 = (4*I_2-I_1)/3
            I_1=copy.copy(I_2)


          

            
            if np.abs((S_2-S_1)/S_1)<epsilon:


                return S_2
            
            
            else:
                S_1=copy.copy(S_2)


#%%
from numpy.random import uniform, normal

#1D Monte-Carlo, takes f, start, end , epsilon same as MC
def MC(f,start, end, epsilon):
    assert end>start
    #List of f(x) is stored in MC
    MC=[]

    #I_1 is previous integral, same as before, used to compute epsilon
    I_1 = -1
    
    #Starts at 1 so I can divide by i and not i-1, more intuitive
    for i in range(1,100001):
        #Pick number from unifrom distribution
        number=uniform(0,2)
        
        #This is for if you want to pick from linear sampling
        #number=np.sqrt(4*uniform(0,1))
        
        #Appends the new f(x)
        MC.append(f(number))
        
        #Computes new integra;
        I_2 = (end-start)*np.array(MC).sum()/(i)


        #If epsilon condition is met, return I_2, else, I_2=I_1
        if np.abs((I_1-I_2)/(I_1))<epsilon:  

            return I_2
        else:
             I_1=I_2

#%%
#This bit is for importance sampling

# this is the weight function g(x)
def g(x, A, lamda):
    e = 2.71828
    return A*np.exp(-1*lamda*x)
#This is for sampling using transformation method
def G_inv(r, lamda):
    return (-1 * math.log((1-(1-np.exp(-(2*lamda)))*float(r))))/lamda



#Calculate the variance for any lamda

#N is the number of samples
def Var(lamda, N):
    #A is the normalisation factor. Derivation is given in the report
    A = lamda/(1-np.exp(-(2*lamda)))

    

    
    #Calculate the average squared
    Sum = 0
    #For N numbers, find a random number and then find f(x)/g(x)
    for i in range(N):
        number = uniform(0, 5)
        Sum += psi_0(number)/g(number, A, lamda)
    squared_avg = (Sum/N)**2
    # get sum of squares
    Sum = 0
    
    #For N numbers, find a random number and then find f^2(x)/g^2(x)
    for i in range(N):
        number = uniform(0, 5)
        Sum += (psi_0(number)/g(number, A, lamda))**2
    
    sum_of_squares = Sum / N
    #Find the variance
    return sum_of_squares - squared_avg

# get variance as a function of lambda by testing many
# different lambdas



#List of sample lambdas
lamda_samples = [i*0.02 for i in range(1, 100)]

#List to store variances
variances = []

#Calculate variance for each test lamda
for i in range(len(lamda_samples)):
    A = lamda_samples[i]/(1-np.exp(-(2*lamda_samples[i])))
    variances.append(Var(lamda_samples[i], 1000))

#Optimal_lamda corresponds to lambda with least variance
optimal_lamda = lamda_samples[np.argmin(np.asarray(variances))]

#Importance sampling using the optimal lamda, where you can define the number of sample sizes and epsilon
#%%
def Imp(f, lamda, N, epsilon):
    
    A = lamda/(1-np.exp(-(2*lamda)))
    #I is inital integral as before
    I_1=-1

    #Sum of f(x) is stored within Sum
    Sum = 0
    for i in range(N):
        #Take number from sample distribution
        
        number=G_inv(uniform(0,1), lamda=lamda)
        #Calculate f(x)/g(x) and add it to the sum
        Sum += f(number)/g(number, A, lamda)
        
        #Compute integral and check if epsilon condition is met
        I_2=float(Sum/(i+1))

        if np.abs((I_1-I_2)/(I_2))<epsilon:    

            return I_2
        else:
             I_1 = I_2
        

    #If epsilon condition is not met, return this statement
    return 'Epsilon too high'


#%%

#3D Monte-Carlo. Works the same as 1D
#Now takes bounds for x,y, and z. Instead of epsilon, this uses number of data points N
def MC_3d(f,x_start, x_end, y_start, y_end, z_start, z_end, N):
    #MC stores a list of f(x)
    MC=[]
    assert x_end>x_start
    assert y_end>y_start
    assert y_end>z_start
    for i in range(1,N):
        
        #Pick random number within bounds
        x_number = uniform(x_start,x_end)
        y_number = uniform(y_start,y_end)
        z_number= uniform(z_start,z_end)

        #Compute f(x,y,z) and store within M_C
        MC.append(f(x_number, y_number, z_number))
        
    #Sum the f(x) and multiply and divide by the relevant values to find integral
    guess = (x_end-x_start)*(y_end-y_start)*(z_end-z_start)*np.array(MC).sum()/(i)
    return guess


#%%
#Trapezium rule and Simpson's are done separately, because it was easier for me to understand
#These are still 1D, will be used for 3D later

#Only difference between these functions and the initial 1D one is that this uses N, previous one used epsilon

#Trapezium 3D, takes f, start, end, N, all defined before
def Trap_3D(f, start,end,N):
    assert end>start
    #y is the sum of f(x)
    y=0
    # P is a list of x-values to be computed
    P = np.linspace(start, end, N)
    for i in range(len(P)):
        #For the first and last values, add f(x)/2
        if i in [0,len(P)-1]:
            y += f(P[i])/2
        else:
            y += f(P[i])
    #Compute Integral
    I_2 = y * (end-start)/(N-1)

    return I_2


def Sim_3D(f, start,end,N):
    
    #Repeats the same thing as Trap_3D for 

    assert end>start
    y=0
    P = np.linspace(start, end, N)
    for i in range(len(P)):
        if i in [0,len(P)-1]:
            y += f(P[i])/2
        else:
            y += f(P[i])

    I_1 = y * (end-start)/(N-1)
    

    #Then fills the points in between to compute the next integral, as you would when doing this iteratively
    h=(end-start)/(N-1)
    #print(h)
    Sum = 0
    for i in range(1, np.int((N+1)/2)):
        Sum+=f(start+(2*i-1)*h)
        
    #I_2 is the second integral for half the sample spacing as I_1
    I_2 = 1/2*I_1 +h*Sum

 

    #Then calculates S_2 as a weighted sum of I_1 and I_2
    S_2 = (4*I_2-I_1)/3
          

    return S_2




#%%
#3D integrals for Simpon's and Trapezoidal methods

#Takes the same outputs as MC_3D, but you can decide how many samples you want for x, y, and z using N_x, N_y, and N_z
def Trap_int_3D(f, x_start, x_end, y_start, y_end, z_start, z_end ,N_x, N_y, N_z):
    #Integrates in the z direction
    def g(x,y):
        return Trap_3D(lambda z: f(x,y,z), z_start, z_end, N_z)
    #Integrates in the y direction
    def m(x):
        return Trap_3D(lambda y: g(x, y), y_start, y_end, N_y)
    #Integrates in the x direction
    return Trap_3D(m, x_start, x_end, N_x)

#Input odd numbers for best results
#Same as Trapezoidal
def Sim_int_3D(f, x_start, x_end, y_start, y_end, z_start, z_end ,N_x, N_y, N_z):
    def g(x,y):
        return Sim_3D(lambda z: f(x,y,z), z_start, z_end, N_z)
    
    def m(x):
        return Sim_3D(lambda y: g(x, y), y_start, y_end, N_y)

    return Sim_3D(m, x_start, x_end, N_x)








#%%
# run simulation for N=1000, with some epsilon


MC_N_3=np.array([3,9,5,1,8])

MC_N_4=np.array([8,29,28,23,8])

MC_N_5=np.array([53,95,152,135,115,132])

MC_N_6=np.array([386,331,291,320,447])

MC_N_7=np.array([989,582.611,859,764])


#%%
MC_results_10=[]
for i in range(10):
    MC_results.append(importance_sampling_MC(optimal_lamda, N,0.0))
print(np.mean(MC_results))

#%%
Epsilon=['-3','-4','-5','-6','-7','-8']
Sim_N=[4,5,7,9,10,12]
Trap_N=[3,4,5,6,6,7]
plt.plot(Epsilon, Sim_N, 'x',label='Simpson')
plt.plot(Epsilon, Trap_N, 'x',label='Trapezoidal')
plt.ylabel('N log(2)')
plt.xlabel('Epsilon log(10)')     
plt.legend()
plt.grid()    
#%%
Sim_Error = np.array([0.49765217297516645,0.4976605716004128, 0.4976610974642394, 0.49766113031942405,  0.49766113031942405,0.49766113237260284])/0.4976611325094764-1


Trap_Error=np.array([ 0.4976074524113482 ,0.49764768620101657 ,0.4976602916019077,0.49766107995074826,0.49766111936976964 ,0.49766113168824405])/0.4976611325094764-1

plt.plot(Epsilon,np.abs(Sim_Error),'x', label='Simpson' )
plt.plot(Epsilon,np.abs(Trap_Error), 'x', label='Trapezoidal')
plt.legend()
plt.grid()
plt.ylabel('% error')


#%%

T_3d_results_2=np.array([0.12115164762154534,0.1231287418243649,.12325299762374514])/0.118136-1
T_3d_results_1=np.array([0.11175611865526938,0.1178031401694185, 0.11813276367480435, ])/0.12325404203950516-1
S_3d_results_2=np.array([0.1178031401694185,0.11786611455603802,0.11813281852390375])/0.118136-1
S_3d_results_1=np.array([0.08616157845864854, 0.1233502936802829,0.12325506513251332])/0.12325404203950516-1
np.array([0.12739392805760694,0.11783244513919817,0.11813278252035396])

np.array([0.1026649722728772,0.12313971612173724, 0.12325300476434771])

#%% 3D Plots

L = np.log10(np.array([3,10,100,1000,10000]))

plt.plot(L[:3],np.abs(T_3d_results_2),'x', label='Trapezoidal')
#plt.plot(L[:3],np.abs(S_3d_results_1))
#plt.plot(L[:3],,np.abs(S_3d_results_1))
plt.plot(L[:3],np.abs(S_3d_results_2),'o', label='Simpson')
plt.errorbar(L,MC_3d_results_2 ,yerr=MC_3d_std_2, fmt='+', label='Monte-Carlo')
plt.legend()
plt.grid()
plt.xlabel('N log(10)')
plt.ylabel('Percentage Error')
plt.ylim(-0.25,0.5)
#%%

plt.plot(L[:3],np.abs(T_3d_results_1),'x', label='Trapezoidal')
#plt.plot(L[:3],np.abs(S_3d_results_1))
#plt.plot(L[:3],,np.abs(S_3d_results_1))
plt.plot(L[:3],np.abs(S_3d_results_1),'o', label='Simpson')
plt.errorbar(L,MC_3d_results_1 ,yerr=MC_3d_std_1, fmt='+', label='Monte-Carlo')
plt.legend()
plt.grid()
plt.xlabel('N log(10)')
plt.ylabel('Percentage Error')
plt.ylim(-0.25,0.5)


      #%%       

MC_N_3=np.array([74,24,75,35,21])

MC_N_4=np.array([8,29,28,23,8])

MC_N_5=np.array([268,80,82,104,158])

MC_N_6=np.array([413,550,471,435,551])

MC_N_7=np.array([1345,1290,875,983,2191])

MC_I_3=np.array([3,9,5,1,8])

MC_I_4=np.array([8,29,28,23,8])

MC_I_5=np.array([53,95,152,135,115,132])

MC_I_6=np.array([386,331,291,320,447])

MC_I_7=np.array([989,582.611,859,764])

Epsilon = ['-3','-4','-5','-6','-7']

MC_I = [np.mean(MC_I_3),np.mean(MC_I_4),np.mean(MC_I_5),np.mean(MC_I_6),np.mean(MC_I_7)]
MC_N = [np.mean(MC_N_3),np.mean(MC_N_4),np.mean(MC_N_5),np.mean(MC_N_6),np.mean(MC_N_7)]

MC_I_error  = [np.std(MC_I_3),np.std(MC_I_4),np.std(MC_I_5),np.std(MC_I_6),np.std(MC_I_7)]
MC_N_error  = [np.std(MC_N_3),np.std(MC_N_4),np.std(MC_N_5),np.std(MC_N_6),np.std(MC_N_7)]

plt.errorbar(Epsilon,MC_N,fmt='x', yerr=MC_N_error, label='Uniform Sampling')
plt.errorbar(Epsilon,MC_I,fmt='x', yerr=MC_I_error, label='Importance Sampling')

plt.ylabel('Number of steps to reach Epsilon')
plt.xlabel('Epsilon log(10)')
plt.legend()
plt.grid()

#%%
#Uni = MC(0,2,0.00000001)  
plt.plot(Uni, label = 'Uniform Dist')
plt.plot(Imp, label = 'Importance DIst')
plt.legend()         
plt.grid()
plt.xlabel('Number of samples')
plt.ylabel('Integral')
MC_L=[0.28418487913525753,0.27574128540785975,0.2810423409565578,0.27715078553501055,0.27616221093463666,0.27878095673141806,0.2743173253848582, 0.2743173253848582,0.27823835019605075,0.27823835019605075,0.27843253846508464]     
MC_U=[0.49663072879619197, 0.49576515397371645,  0.5036826616273354,  0.49953054496829363,0.4914224861049772, 0.4971508244154322, 0.49699815563884014, 0.49463060053487673, 0.5016130701943641, 0.49587773506543703]
#%%
MC_3d_results_2_3=np.array([0.12810765385002934,0.1380826541830248,0.3596057602020404,0.051865569029118555])/0.118136-1
MC_3d_results_2_10=np.array([0.19683620601244604,0.07818982289097347, 0.0934419977333925,  0.10829857735760873, .2364420294399357])/0.118136-1
MC_3d_results_2_100=np.array([0.12298914792944111,0.10083991822634027,0.10780355324884625, 0.12130799974569356, 0.141941150267714])/0.118136-1
MC_3d_results_2_1000=np.array([0.11373250677001381,0.11358945566034613, 0.12096854727260112, 0.11714838176792366, 0.1261446801562372])/0.118136-1
MC_3d_results_2_10000=np.array([0.11695058157814552, 0.11909535036799022, 0.11888612651302415, 0.11760949927690532, 0.11760949927690532])/0.118136-1


#%%

MC_3d_results_1_10=[]
MC_3d_results_1_100=[]
MC_3d_results_1_1000=[]
MC_3d_results_1_10000=[]

for i in range(5):
    MC_3d_results_1_10.append(MC_3d(0,2,0,2,0,2,10))
    MC_3d_results_1_100.append(MC_3d(0,2,0,2,0,2,100))
    MC_3d_results_1_1000.append(MC_3d(0,2,0,2,0,2,1000))
    MC_3d_results_1_10000.append(MC_3d(0,2,0,2,0,2,1000))
#%%
MC_3d_results_1_3=np.array([0.006855995089412816,0.0046444922788815655,0.07710062951821724,0.39416756252844,0.008094401867509218])/0.12325404203950516-1

#%%
MC_3d_results_1_10=np.array(MC_3d_results_1_10)/0.12325404203950516-1
MC_3d_results_1_100=np.array(MC_3d_results_1_100)/0.12325404203950516-1
MC_3d_results_1_1000=np.array(MC_3d_results_1_1000)/0.12325404203950516-1
MC_3d_results_1_10000=np.array(MC_3d_results_1_10000)/0.12325404203950516-1
#%%

MC_3d_results_1=[np.abs(np.mean(MC_3d_results_1_3)),np.abs(np.mean(MC_3d_results_1_10)),np.abs(np.mean(MC_3d_results_1_100)),np.abs(np.mean(MC_3d_results_1_1000)), np.abs(np.mean(MC_3d_results_1_10000))]
MC_3d_results_2=[np.abs(np.mean(MC_3d_results_2_3)),np.abs(np.mean(MC_3d_results_2_10)),np.abs(np.mean(MC_3d_results_2_100)),np.abs(np.mean(MC_3d_results_2_1000)), np.abs(np.mean(MC_3d_results_2_10000))]

MC_3d_std_1=[np.abs(np.std(MC_3d_results_1_3)),np.std(np.std(MC_3d_results_1_10)),np.abs(np.std(MC_3d_results_1_100)),np.abs(np.std(MC_3d_results_1_1000)), np.abs(np.std(MC_3d_results_1_10000))]
MC_3d_std_2=[np.abs(np.std(MC_3d_results_2_3)),np.std(np.std(MC_3d_results_2_10)),np.abs(np.std(MC_3d_results_2_100)),np.abs(np.std(MC_3d_results_2_1000)), np.abs(np.std(MC_3d_results_2_10000))]


plt.plot(MC_3d_results_2, label='Excited State Integral')
plt.plot(MC_3d_results_1, label='Ground State Integral')
plt.grid()
plt.legend()