#Created by Norhasliza Yusof for SIF3012 Course
#program shooting method BVP using Runge-Kutta order 2
#This code can be modify for RK4 or simplify in doing iteration in calling the integration.
#This equation is to solve u"=-pi**2*(u+1)/4.0
#The exact solution/analytical solution for this problem is u(x)=cos(pi*x/2)+2.0*sin(pi*x/2)-1

import numpy as np
#import pandas as pd
import matplotlib.pyplot as mpl

def f1(x,y1,y2):
    return y2

def f2(x,y1,y2):
    return 4.0*y1-4.0*x

def f3(x,y3,y4):
    return y4

def f4(x,y3,y4):
    return 4.0*y3

# *** the first loop is to determine the boundaries at xn ***

# the initial boundaries at x0
y1 = 0.0
y2 = 0.0
y3 = 0.0
y4 = 1.0
beta = 2.0

n = 4
x = 0.0
h =  1/n

print('first')
    #for i in range(0,n):
    
#rk2 calculation to do the correction
for i in range(0,n):
      K11 = f1(x,y1,y2)
      K21 = f1(x+h, y1+h*f1(x,y1,y2),y1)
      y11 = (y1 + (h/2.0)*(K11+K21))
    
      K12 = f2(x,y1,y2)
      K22 = f2(x+h, y2+h*f2(x,y1,y2),y2)
      y22 = (y2 + (h/2.0)*(K12+K22))
    
      y1 = y11
      y2 = y22

      K13 = f3(x,y3,y4)
      K23 = f3(x+h, y3+h*f3(x,y3,y4),y3)
      y33 = (y3 + (h/2.0)*(K13+K23))
    
      K14 = f4(x,y3,y4)
      K24 = f4(x+h, y4+h*f4(x,y3,y4),y4)
      y44 = (y4 + (h/2.0)*(K14+K24))
    
      x = x+h
      y3 = y33
      y4 = y44


# y1n and y3n are the boundaries at xn
y1n = y1
y3n = y3


# *** the second loop is to determine the solution of the linear BVP ***

# the initial boundaries at x0
print('second')
y1 = 0.0
y2 = 0.0
y3 = 0.0
y4 = 1.0

n = 4
x = 0.0
h =  1/n

    #for i in range(0,n):
    
    #rk2 calculation to do the correction
for i in range(0,n):
    K11 = f1(x,y1,y2)
    K21 = f1(x+h, y1+h*f1(x,y1,y2),y1)
    y11 = (y1 + (h/2.0)*(K11+K21))
        
    K12 = f2(x,y1,y2)
    K22 = f2(x+h, y2+h*f2(x,y1,y2),y2)
    y22 = (y2 + (h/2.0)*(K12+K22))
        

    y1 = y11
    y2 = y22
                
    K13 = f3(x,y3,y4)
    K23 = f3(x+h, y3+h*f3(x,y3,y4),y3)
    y33 = (y3 + (h/2.0)*(K13+K23))
                
    K14 = f4(x,y3,y4)
    K24 = f4(x+h, y4+h*f4(x,y3,y4),y4)
    y44 = (y4 + (h/2.0)*(K14+K24))
                
    x = x+h
    y3 = y33
    y4 = y44

# the linear solution
    yx = y1 + ((beta-y1n)/y3n)*y3
    print(i,x,yx)
    mpl.plot(x,yx,'xk')

x = np.arange(0.0,1.05,0.01)
#Here y is the exact solution/analytical solution for u"(x).
#See the comment in the header for the full formula
y = np.exp(2)*(np.exp(4)-1.0)**(-1)*(np.exp(2*x)-np.exp(-2*x))+x
print(x,y)

mpl.plot(x,y,'-')


mpl.show()
