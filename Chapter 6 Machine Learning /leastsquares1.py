#-----------------------------------------------------------------------------#
# This python program is adapted from COIL program                            #
# between University of Malaya and Prince Songkla University.                 # 
#                                                                             #
#-----------------------------------------------------------------------------#

#import library
import pandas as pd
import matplotlib.pyplot as mpl
from scipy.stats import linregress

# This is an example of a python program that reads data (input data)
# from an external file.
# In this case the external file is an excel file that you normally records
# experimental data from the lab.
# The data in the excel file is a measurement of thermal conductance with
# temperature.

# Extract input data (original data)from the excel file.
# The name of the excel file is lsdata1.xlsx.
df=pd.read_excel(r'lsdata1.xlsx')

# The number of data points is n=10
# x is the data for x-axis (temperature).
# y is the data for y-axis (thermal conductance).
x=df['Temperature']
y=df['Conductance']

# The Scipy function linregress(x,y) calculates a linear least-squares 
# regression for two sets of measurements and saves the value of the slope (m)
# and the y-intercept (b) in lslinear.
lslinear= linregress(x,y)

# m is the slope in Equation (2) in the Least Squares slides.
# b is the y-axis intercept in Equation (3) in the Least Squares slides.
m= lslinear.slope
b= lslinear.intercept

# The plot of the input data (original data) using individual points.
mpl.scatter(df['Temperature'], df['Conductance'],label='Original Data')

#calculate error bar from data and fitting
yerr = (m*x+b)-y

# The plot of the best straight line, Equation (1) in the Least Squares slides. 
mpl.plot(x,m*x +b,color='red',label='Least Square Method')

#add error bar
mpl.errorbar(x,m*x+b,yerr,fmt='.k')


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

# Save the graph in the png format in an external file.
mpl.savefig('least_square1.png')

# Plot the graphs.
mpl.show()
