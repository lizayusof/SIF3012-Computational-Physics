# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:32:38 2020

@author: User
"""

#-----------------------------------------------------------------------------#
# This python program is part of the activities under COIL program            #
# between University of Malaya and Prince Songkla University.                 # 
# Acnowledgement: Ameera Aswani for providing this code.                      # 
#-----------------------------------------------------------------------------#

#import library
import numpy as np
import matplotlib.pyplot as mpl
from scipy.stats import linregress

# This is an example of a python program that you keyin the data (input data)
# yourself.
# The example data is a measurement of thermal conductance with temperature.

# The number of data points is n = 10
# x is the data for x-axis (temperature).
# y is the data for y-axis (thermal conductance).
x = np.array([100,200,300,400,500,600,700,800,900,1000])
y = np.array([1.32,0.94,0.835,0.803,0.694,0.613,0.547,0.487,0.433,0.38])

# The Scipy function linregress(x,y) calculates a linear least-squares 
# regression for two sets of measurements and saves the value of the slope (m)
# and the y-intercept (b) in lslinear.
lslinear= linregress(x,y)

# m is the slope in Equation (2) in the Least Squares slides.
# b is the y-axis intercept in Equation (3) in the Least Squares slides.
m= lslinear.slope
b= lslinear.intercept
print('gradient',m)
print('intercept', b)

# The plot of the input data (original data) using individual points.
mpl.scatter(x,y,label='Original Data')

# The plot of the best straight line, Equation (1) in the Least Squares slides. 
mpl.plot(x,m*x +b,color='red',label='Least Square Method')

# Setting label of the x-axis.
mpl.xlabel('Temperature (K)')

# Setting label of the y-axis.
mpl.ylabel('Conductance (Watts/Kelvin)')

# Setting title of the graph.
mpl.title('Conductance against Temperature')

# Insert the legend in the graph.
mpl.legend()

# Calculates and print the R-squared of best straight in lslinear.
print(f"R-squared: {lslinear.rvalue**2:.6f}")

#mpl.xlim(150,1000)

# Save the graph in the png format in an external file.
mpl.savefig('least_square2.png')

# Plot the graphs.
mpl.show()
