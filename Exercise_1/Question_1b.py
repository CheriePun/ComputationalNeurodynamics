"""
Computational Neurodynamics
Exercise 1

Solves the second-order ODE of mass-spring-damper system, by numerical
simulation using the Euler method.

"""

import numpy as np
import matplotlib.pyplot as plt

dt = 0.001     # Step size for exact solution
m = 1
c = 0.1
k = 1

# Create time points
# Initialize arrays
Tmin = 0
Tmax = 100
T = np.arange(Tmin, Tmax+dt, dt)
y = np.zeros(len(T))
dy_dt = np.zeros(len(T))
d2y_dt2 = np.zeros(len(T))

####
y[0] = 1 # Initial value
dy_dt[0] = 0 # Initial value
d2y_dt2[0] = (-1/m)*(c*dy_dt[0] + k*y[0]) #Calculate using y[0] and dy_dt[0]

# Use Euler method to calculate y and dy/dt at t using values at (t-1)
# Use the new values of y and dy/dt to calculate d2y/dt2 at t in preparation
# for next step
for t in xrange(1, len(T)):
    y[t] = y[t-1] + dt*dy_dt[t-1]
    dy_dt[t] = dy_dt[t-1] + dt*d2y_dt2[t-1]
    d2y_dt2[t] = (-1/m)*(c*dy_dt[t] + k*y[t])


# Plot the results
plt.plot(T      , y            , 'r', label='Second-order ODE')
plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc=0)
plt.show() #Show the graph at the end of the execution of the code
