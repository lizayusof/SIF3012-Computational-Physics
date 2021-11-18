#Non-linear BVP using shooting method
#Created by Norhasliza Yusof and Hasan Abu Kassim
#This code can be modify for RK4 or simplify in doing iteration in calling the integration.
#This equation is to solve u"=-pi**2*(u+1)/4.0
#The exact solution/analytical solution for this problem is u(x)=cos(pi*x/2)+2.0*sin(pi*x/2)-1

import numpy as np
import matplotlib.pyplot as mpl

# f1 = dy1/dx
def f1(x,y1,y2):
    return y2

# f2 = dy2/dx
def f2(x,y1,y2):
    return -(np.pi**2*(y1+1.0))/4.0

#initialization of the problem/input

# Mesh points
n =  101

# Maximum number of iteration to find the root
kmax = 11

#Range of integration on x-axis
xa=0.0
xb=1.0

# Step to approximate second gradient
# You can make it smaller to increase accuracy
ds=0.01

# Tolerance
tol=1.e-6

# Step size
h = (xb-xa)/(n-1)

# Boundaries at xa and xb
ya=0.0
yb=1.0

# Initial gradient approximation
s0 = (yb-ya)/(xb-xa)

# Second gradient approximation
s1 = s0+ds

# The first guess = known solution
y1=ya

# Secant method to find the root

# Iteration to improve gradient begins

for i in range(1,kmax):
    
# First trial solution (gradient) using Runge-Kutta method
    
    y2=s0
    for i in range(0,n-1):
        x=xa+h*i
        K11 = h*f1(x,y1,y2)
        K12 = h*f2(x,y1,y2)
    
        K21 = h*f1(x+0.5*h, y1+0.5*K11, y2+0.5*K12)
        K22 = h*f2(x+0.5*h, y1+0.5*K11, y2+0.5*K12)

        K31 = h*f1(x+0.5*h,  y1+0.5*K21, y2+0.5*K22)
        K32 = h*f2(x+0.5*h, y1+0.5*K21, y2+0.5*K22)
    
        K41 = h*f1(x+h, y1+K31,y2+K32)
        K42 = h*f2(x+h, y1+K31, y2+K32)

        y11 = y1 + (K11+2*K21+2*K31+K41)/6.0
        y22 = y2 + (K12+2*K22+2*K32+K42)/6.0

# Solution for f1 = dy1/dx
        y1 = y11

# Solution for f2 = dy2/dx
        y2 = y22

# Adjusted paramater to improve y1 using s0
    del0 = y1-1.0

# Second trial solution (gradient) using Runge-Kutta method

    y2=s1 
    for i in range(n-1):
        x=xa+h*i
        K11 = h*f1(x,y1,y2)
        K12 = h*f2(x,y1,y2)
        
        K21 = h*f1(x+0.5*h, y1+0.5*K11, y2+0.5*K12)
        K22 = h*f2(x+0.5*h, y1+0.5*K11, y2+0.5*K12)
        
        K31 = h*f1(x+0.5*h,  y1+0.5*K21, y2+0.5*K22)
        K32 = h*f2(x+0.5*h, y1+0.5*K21, y2+0.5*K22)
        
        K41 = h*f1(x+h, y1+K31,y2+K32)
        K42 = h*f2(x+h, y1+K31, y2+K32)
        

        y11 = y1 + (K11+2*K21+2*K31+K41)/6.0
        y22 = y2 + (K12+2*K22+2*K32+K42)/6.0

# Solution for f1 = dy1/dx
        y1 = y11
        
# Solution for f2 = dy2/dx
        y2 = y22

# Adjusted paramater to improve y1 using s1
    del1=y1-1.0

# Difference between first and second approximations of solution using s0 and s1
    d=del1-del0

# If d is within tolerance then y1 is close enough to the boundary at x=b
    if abs(d) < tol:
       print ('stop')

# If not, improve the gradient by the following approximation
    else:
        s2 = s1-del1*(s1-s0)/d # Improved (new) second gradient
        s0 = s1 # The previous second gradient becomes the (new) first gradient
        s1 = s2  # The new second gradient replaces the old second gradient

# s1 --> s0
# s2 --> s1

    print(i,y1,abs(d))
    mpl.plot(x,y11,'ro',linestyle='-')


# Analytical result for comparison with numerical solution
t = np.arange(0.0,1.05,0.1)
u = (np.cos(np.pi*t/2.0))+2.0*(np.sin(np.pi*t/2.0))-1.0
mpl.plot(t,u,label='Analytical solution')


mpl.title ("$\cos (\pi x/2) + 2 \sin (\pi x/2)-1$")
mpl.xlabel('x-axis')
mpl.ylabel('y-axis')
mpl.legend()
mpl.savefig('shooting_ODE.png')


mpl.show()
