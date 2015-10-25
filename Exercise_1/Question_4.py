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
    
def m_diff(v,m):
    return alpha_m(v)*(1-m)-beta_m(v)*m
    
def n_diff(v,n):
    return alpha_n(v)*(1-n)-beta_n(v)*n
    
def h_diff(v,h):
    return alpha_h(v)*(1-h)-beta_h(v)*h
    
def sum_of_currents(v,i):
    return g_Na*(m[i]**3)*h[i]*(v-e_Na)+g_K*(n[i]**4)*(v-e_K)+g_L*(v-e_L)

def v_diff(v,i):
    return -sum_of_currents(v,i) + I
    
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
for t in xrange(len(T)-1):
    # Update v, m, n, h according to Hodgkin-Huxley's equations
    vk1 = v_diff(v[t], t)
    vk2 = v_diff(v[t] + 0.5*dt*vk1, t)
    vk3 = v_diff(v[t] + 0.5*dt*vk2, t)
    vk4 = v_diff(v[t] + dt*vk3, t)
    v[t+1] = v[t] + dt*(vk1+2*vk2+2*vk3+vk4)/6
    
    mk1 = m_diff(v[t], m[t])
    mk2 = m_diff(v[t], m[t] + 0.5*dt*mk1)
    mk3 = m_diff(v[t], m[t] + 0.5*dt*mk2)
    mk4 = m_diff(v[t], m[t] + dt*mk3)
    m[t+1] = m[t] + dt*(mk1+2*mk2+2*mk3+mk4)/6
    
    nk1 = n_diff(v[t], n[t])
    nk2 = n_diff(v[t], n[t] + 0.5*dt*nk1)
    nk3 = n_diff(v[t], n[t] + 0.5*dt*nk2)
    nk4 = n_diff(v[t], n[t] + dt*nk3)
    n[t+1] = n[t] + dt*(nk1+2*nk2+2*nk3+nk4)/6
    
    hk1 = h_diff(v[t], h[t])
    hk2 = h_diff(v[t], h[t] + 0.5*dt*hk1)
    hk3 = h_diff(v[t], h[t] + 0.5*dt*hk2)
    hk4 = h_diff(v[t], h[t] + dt*hk3)
    h[t+1] = h[t] + dt*(hk1+2*hk2+2*hk3+hk4)/6
    

# Plot the results
plt.plot(T      , v      , 'b', label='')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.legend(loc=0)
plt.show()
