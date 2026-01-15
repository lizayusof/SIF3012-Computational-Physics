#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 23:29:35 2021

@author: hasanabukassim
"""

# The steps are described in Tutorial 4 Question 6.

#----------------------------------------------------------------------------#

# Step 1: Import python packages

import numpy as np
# Imports only the linear model function from Scikit Learn package.
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------#

# Step 2: Generate random training dataset

# Generate a random dataset (input/observation) of size nobs.
# Note: change this according to the problem to solve.
nobs = 100   

# Create the random dataset for the input/observation with size nobs
# with selected minimum value (low) and maximum value (high).
# Note: change this according to the problem to solve.
xobs = np.random.uniform(low=0.0, high=200.0, size=nobs)

# The random variable noise is added to create noise to the dataset
# to make the dataset distribution looks more "random".
noise = np.random.normal(loc=0.0, scale=5.0, size=nobs)

# Set yobs (the output) as a linear relationship with xobs (input/observation).
# Note: change this according to the problem to solve. 
#yobs=0.07059*xobs + noise

yobs = 2*xobs + 5 + noise

#----------------------------------------------------------------------------#

# Step 3: Fit Linear regression model. There are different names for 
#         x and y described in Question 6.
# Follow the steps to use the sklearn package.

# 1. xobs is reshaped from a numpy array to a matrix by using reshape, 
#    which is required by the sklearn package and is the features.
#    The value -1 is the number of rows from the original xobs, 
#    The value 1 is representing 1 column in the original xobs,.
x = xobs.reshape(-1, 1)

# 2. Set yobs as the target in the supervised learning-regression analysis.
y = yobs

# 3. Creating the linear regression model.
#    LinearRegression fits a linear model with coefficients calculated/returned
#    to lr.
lr = LinearRegression()

# 4. The coefficients in lr are fitted to a linear model i.e. 
#    y = f(x) = a0#x + a1.
#    fit(x, y) fits a linear model.
lr.fit(x, y)

# 4. Print the coefficient and intercept of the linear model.
# The coefficients:
print('The coefficient = ',lr.coef_)
# The intercept:
print('The intercept = ',lr.intercept_)

# 5. Plot the training set and the supervised linear model.

# Create a new figure, or activate an existing figure.   
fig = plt.figure(figsize=(21,8))
# Title of figure:
   # color='magenta'
fig.suptitle('Machine Learning linear Regression in Supervised Learning - \
Regression Algorithm.',color='m', fontsize=16)

#    The original training set (points).
#    Plot the random sample dataset (training set) using a scatterplot
plt.plot(xobs, yobs, 'o')

#   The supervised linear model (linear line).
#   Plot the linear model using a line.
#   linspave returns num evenly spaced samples/grid, calculated over the
#   interval [start, stop]. num = (integer) number of samples to generate. 
xgrid = np.linspace(xobs.min(), xobs.max(), num=100)
plt.plot(xgrid, lr.coef_[0]*xgrid + lr.intercept_)

#----------------------------------------------------------------------------#

# Step 4: Predictions with the linear regression using 2 methods

# xpredict = the selected x values we want to predict y.
xpredict = np.array([50, 100, 150, 200])

# Method 1. Use the linear equation,  y = f(x) = a0#x + a1.
#           ypredict1 = predicted y using xpredict
ypredict1 = lr.intercept_ + xpredict*lr.coef_[0]

# Method 2. Use the predict function in scikit-learn.
#           ypredict2 = predicted y using xpredict
ypredict2=lr.predict(xpredict.reshape(-1,1))

# Print the predictions from the two methods.
# Both methods should return the same predicted values.
print('Predictions using the linear regression supervised learning algorithm')
print('Selected x values: ', xpredict)
print('Method 1: ', ypredict1)
print('Method 2: ', ypredict2)

#----------------------------------------------------------------------------#

# Plot the graphs.
plt.show()

# Save the graph in the png format in an external file.
plt.savefig('t6q6.png')
