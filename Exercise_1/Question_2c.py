"""
Computational Neurodynamics
Exercise 1

Simulates Izhikevich's neuron model using the Runge-Kutta 4 method.
Parameters for regular spiking, fast spiking and bursting
neurons extracted from:

http://www.izhikevich.org/publications/spikes.htm

"""

import numpy as np
import matplotlib.pyplot as plt

# Create time points
Tmin = 0
Tmax = 200   # Simulation time
dt   = 0.01  # Step size
T    = np.arange(Tmin, Tmax+dt, dt)

# Base current
I = 10

## Parameters of Izhikevich's model (regular spiking)
a = 0.02
b = 0.2
c = -65
d = 8

## Parameters of Izhikevich's model (fast spiking)
# a = 0.02
# b = 0.25
# c = -65
# d = 2

## Parameters of Izhikevich's model (bursting)
# a = 0.02
# b = 0.2
# c = -50
# d = 2

v = np.zeros(len(T))
u = np.zeros(len(T))

## Initial values
v[0] = -65
u[0] = -1

def v_differentiation(v, u):
    return 0.04*v**2 + 5*v + 140 - u + I

def u_differentiation(v, u):
    return a * (b*v - u)
    
## SIMULATE
for t in xrange(len(T)-1):
  # Update v and u according to Izhikevich's equations
  vk1 = v_differentiation(v[t], u[t])
  vk2 = v_differentiation(v[t] + 0.5*dt*vk1, u[t])
  vk3 = v_differentiation(v[t] + 0.5*dt*vk2, u[t])
  vk4 = v_differentiation(v[t] + dt*vk3, u[t])
  v[t+1] = v[t] + dt*(vk1+2*vk2+2*vk3+vk4)/6
  
  uk1 = u_differentiation(v[t], u[t])
  uk2 = u_differentiation(v[t], u[t] + 0.5*dt*uk1)
  uk3 = u_differentiation(v[t], u[t] + 0.5*dt*uk2)
  uk4 = u_differentiation(v[t], u[t] + dt*uk3)
  u[t+1] = u[t] + dt*(uk1+2*uk2+2*uk3+uk4)/6
  
  
  # Reset the neuron if it has spiked
  if v[t+1] >= 30:
    v[t]   = 30          # Add a Dirac pulse for visualisation
    v[t+1] = c           # Reset to resting potential
    u[t+1] = u[t+1] + d  # Update recovery variable
    
    
## Plot the membrane potential
plt.subplot(211)
plt.plot(T, v)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential v (mV)')
plt.title('Izhikevich Neuron')

# Plot the reset variable
plt.subplot(212)
plt.plot(T, u)
plt.xlabel('Time (ms)')
plt.ylabel('Reset variable u')
plt.show()

