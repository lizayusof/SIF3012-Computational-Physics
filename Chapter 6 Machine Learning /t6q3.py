#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 06:35:06 2021

@author: hasanabukassim
"""

import numpy as np
import matplotlib.pyplot as mat

# This example is taken from numpy.linalg.lstsq website. You can visit it at:
# https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html    

# We want to fit a line, y = mx + c, through some noisy data-points:

# Example of reading data in csv file with A header into an array using
# loadtext. The header which ia a row with text is skipped using the 
# parameter skiprows.
with open('t4q3.csv') as f:
    mydata = np.loadtxt(f, delimiter=',', skiprows=1)

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

# Rewrite the linear equation as y = Ap 
# where A = [[x 1]] and p=[[m], [c]]
A = np.vstack([x,np.ones(len(x))]).T

# With A, solve y = mx + b by least square method using lstsq
m,b = np.linalg.lstsq(A,y,rcond=None)[0]

print('Gradient =', m, '; y-axis intercept =', b)

# Create plot
mat.plot(x,y,'o',label='Original data',markersize=10)
mat.plot(x, m*x +b , 'r',label='Fitted line')
mat.xlabel('x')
mat.ylabel('y')
mat.legend()
mat.show()

# Save plot as png file. Remove whitespace around the image.
mat.savefig('t6q3.png', bbox_inches='tight')
