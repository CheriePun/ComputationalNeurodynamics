"""
Computational Neurodynamics
Exercise 1

Simulates Hodgkin-Huxley's neuron model using the Euler method.
Parameters extracted from:

paper by Hodgkin and Huxley in 1952

"""

import numpy as np
import matplotlib.pyplot as plt
import math

# Create time points
Tmin = 0
Tmax = 100   # Simulation time
dt   = 0.01  # Step size
T    = np.arange(Tmin, Tmax+dt, dt)

# Base current
I = 10

## Parameters of Hodgkin-Huxley's model
g_Na = 120
e_Na = 115
g_K = 36
e_K = -12
g_L = 0.3
e_L = 10.6

## Functions to calculate differentials
def alpha_m(v):
    return (2.5-0.1*v) / (math.exp(2.5-0.1*v)-1)
    
def beta_m(v):
    return 4*math.exp(-v/18)
   
def alpha_n(v):
    return (0.1-0.01*v) / (math.exp(1-0.1*v)-1)

def beta_n(v):
    return 0.125*math.exp(-v/80)

def alpha_h(v):
    return 0.07*math.exp(-v/20)
   
def beta_h(v):
    return 1/(math.exp(3-0.1*v) + 1)
    
def m_diff(i):
    return alpha_m(v[i])*(1-m[i])-beta_m(v[i])*m[i]
    
def n_diff(i):
    return alpha_n(v[i])*(1-n[i])-beta_n(v[i])*n[i]
    
def h_diff(i):
    return alpha_h(v[i])*(1-h[i])-beta_h(v[i])*h[i]
    
def sum_of_currents(i):
    return g_Na*(m[i]**3)*h[i]*(v[i]-e_Na)+g_K*(n[i]**4)*(v[i]-e_K)+g_L*(v[i]-e_L)

def v_diff(i):
    return -sum_of_currents(i) + I
    
# Create arrays for each parameter
v = np.zeros(len(T))
m = np.zeros(len(T))
n = np.zeros(len(T))
h = np.zeros(len(T))

## Initial values
v[0] = -10
m[0] = 0
n[0] = 0
h[0] = 0

## SIMULATE
for t in xrange(1, len(T)):
    # Update v, m, n, h according to Hodgkin-Huxley's equations
    v[t] = v[t-1] + dt*v_diff(t-1)
    m[t] = m[t-1] + dt*m_diff(t-1)
    n[t] = n[t-1] + dt*n_diff(t-1)
    h[t] = h[t-1] + dt*h_diff(t-1)
    

# Plot the results
plt.plot(T      , v      , 'b', label='')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.legend(loc=0)
plt.show()
