#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 04:34:51 2021

@author: hasanabukassim
"""

import numpy as np
import matplotlib.pyplot as plt

# We want to fit data using polynomial of any order.

# Example of reading data in csv file with A header into an array using
# loadtext. The header which ia a row with text is skipped using the 
# parameter skiprows.

# Thermal conductivity of mercury.
with open('t4q5.csv') as f:    
    mydata = np.loadtxt(f, delimiter=',', skiprows=2)

# [:,0] = data corresponding to all rows and just the first column.
x = mydata[:,0]

# [:,1] = data corresponding to all rows and just the second column.
y = mydata[:,1]

# Print x values only.
print('x = ',x)

# Print y values only.
print('y = ',y)

# Print x and y values.
print('x = ',x,', y = ',y)

# Polynomial fitting coefficients by using polyfit:
    # First-order fitting = linear fitting, y = a0*x + a1.
z1 = np.polyfit(x, y, 1)
   # Second-order fitting = quadratic fitting, y = a0x^2 + a1*x + a2.
z2 = np.polyfit(x, y, 2)

print('Fitting coefficients of z1 =',z1)
print('Fitting coefficients of z2 =',z2)

# Create polynomial using poly1d:
    # First-order polynomial = linear equation.
p1 = np.poly1d(z1)
    # Second-order polynomial = quadratic equation.
p2 = np.poly1d(z2)

# We can predict any value of the thermal conductivity at any temperature.
# Evaluate the polynomials at any point, xpredict.
xpredict = 535
p1predict = p1(xpredict)
p2predict = p2(xpredict)

print('At x =', xpredict,', Linear prediction = ',p1predict) 
print('At x =', xpredict,', Quadratic prediction = ',p2predict)

# Create a new figure, or activate an existing figure.   
fig = plt.figure()
# Title of figure:
   # color='magenta'
fig.suptitle('Thermal Conductivity of Mercury',color='m', fontsize=16)
# Draw grid lines:
plt.grid(True)
# Original data:        
   # color='blue'
plt.plot(x,y,'o',color='b',label='Original data',markersize=10)
# Linear equation:
   # color='green'
plt.plot(x,p1(x),'r',color='g',label='Fitted data, p1')
# Quadratic equation:
   # color='red'
plt.plot(x, p2(x),'r',color='r',label='Fitted line, p2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
# Draw lines:
   # Vertical line, color='black'
plt.axvline(x=xpredict, color='k', linestyle='--')
   # Horinzontal line, color='cyan'
plt.axhline(y=p1predict, color='c', linestyle='--')
   # Horinzontal line, color='yellow'
plt.axhline(y=p2predict, color='y', linestyle='--')
plt.show()


