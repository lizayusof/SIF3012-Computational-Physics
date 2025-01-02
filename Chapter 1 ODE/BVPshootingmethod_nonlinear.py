#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Non-linear BVP using shooting method
#Created by Norhasliza Yusof for SIF3012 Computational Physics
#This code can be modify for RK4 or simplify in doing iteration in calling the integration.

#Non-linear shooting method using arrays

import numpy as np
import matplotlib.pyplot as mpl

# Mesh points
n = 500

# Maximum number of iteration to find the root
m = 5
y  = np.zeros((1001,1001))

# g1f = dy1/dx
def g1f(y1,y2,x):
    return y2

# g2f = dy2/dx
def g2f(y1,y2,x):
    return (-(np.pi**2)*(y1+1.0))/4


#initialised problem

#tolerence
tol = 1.0E-06 

# xa = x(a)
# xb = x(b)
xa = 0.0 
xb = 1.0 


# step size 
h  = (xb-xa)/(n-1)


#d  = 0.1

# Boundaries at xa and xb
#ya = y(a), yb = y(b)
ya = 0.0
yb = 1.0

# Initial gradient approximation
s0 = (yb-ya)/(xb-xa)

# Step to approximate second gradient
# You can make it smaller to increase accuracy
ds = 0.01

# Second gradient approximation
s1 = s0+ds

#Secant search for the root
# RK4 embeded in the second search for root

# The first guess = known solution
y[1,n] = ya


# Iteration to improve gradient begins
for i in range(1,m):
    y[2,1]=s0  # First trial solution (gradient) using Runge-Kutta method
    for i in range(1,n):
        x = xa+h*i  # x change with step size
        y1 = y[1,i]
        y2 = y[2,i]
        dk11 = h*g1f(y1,y2,x)
        dk21 = h*g2f(y1,y2,x)
        dk12 = h*g2f((y1+dk11/2.0),(y2+dk21/2.0),(x+h/2.0))
        dk22 = h*g2f((y1+dk11/2.0),(y2+dk21/2.0),(x+h/2.0))
        dk13 = h*g1f((y1+dk12/2.0),(y2+dk22/2.0),(x+h/2.0))
        dk23 = h*g2f((y1+dk12/2.0),(y2+dk22/2.0),(x+h/2.0))
        dk14 = h*g1f((y1+dk13),(y2+dk23),(x+h))
        dk24 = h*g2f((y1+dk13),(y2+dk23),(x+h))
        y[1,i+1] = y[1,i]+(dk11+2.0*(dk12+dk13)+dk14)/6.0
        y[2,i+1] = y[2,i]+(dk21+2.0*(dk22+dk23)+dk24)/6.0

    f0 = y[1,n]-1.0  # Adjusted paramater to improve y1 using s0

    y[2,1] = s1 # Second trial solution (gradient) using Runge-Kutta method
    for i in range(1,n):
        x  = xa+h*i
        y1 = y[1,i]
        y2 = y[2,i]
        dk11 = h*g1f(y1,y2,x)
        dk21 = h*g2f(y1,y2,x)
        dk12 = h*g2f((y1+dk11/2.0),(y2+dk21/2.0),(x+h/2.0))
        dk22 = h*g2f((y1+dk11/2.0),(y2+dk21/2.0),(x+h/2.0))
        dk13 = h*g1f((y1+dk12/2.0),(y2+dk22/2.0),(x+h/2.0))
        dk23 = h*g2f((y1+dk12/2.0),(y2+dk22/2.0),(x+h/2.0))
        dk14 = h*g1f((y1+dk13),(y2+dk23),(x+h))
        dk24 = h*g2f((y1+dk13),(y2+dk23),(x+h))
        y[1,i+1] = y[1,i]+(dk11+2.0*(dk12+dk13)+dk14)/6.0
        y[2,i+1] = y[2,i]+(dk21+2.0*(dk22+dk23)+dk24)/6.0

    # Adjusted paramater to improve y1 using s1
    f1 = y[1,n]-1.0
    d  = f1-f0 # Difference between first and second approximations of solution using s0 and s1
    s2 = s1-f1*(s1-s0)/d # Improved (new) second gradient
    s0 = s1 # The previous second gradient becomes the (new) first grad
    s1 = s2 # The new second gradient replaces the old second gradient

for i in range(1,n):
    x  = xa+h*i    
    mpl.scatter(x,y[1,i-1], color='red')
    
#print the value of x and the solution    
print(x,y[1,i-1])

#plot the solution vs x
mpl.plot(x,y[1,i-1],'d', color='k')

#compare with the exact solution
t = np.arange(0.0,1.05,0.1) #t is the new x-axis
u = (np.cos(np.pi*t/2.0))+2.0*(np.sin(np.pi*t/2.0))-1.0 #the extact solution
mpl.plot(t,u,color='green',label='Analytical solution')   

#label
mpl.xlabel('x')
mpl.ylabel('u(x)')

mpl.legend()
mpl.show() 
