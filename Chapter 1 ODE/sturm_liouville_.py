#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by Norhasliza Yusof for SIF3012 Computational Physics course

#This python code adapated from Tao Pang Computational Physics textbook (1997)


# Main program for solving the Legendre equation with the simplest 
# algorithm for the Sturm-Liouville equation and the bisection method
# for the root search.  
# 

import numpy as np

n=501
u = np.zeros (1000)

# Initialization of the problem

dl = 1.0E-06
h  = 2.0/(n-1)
ak = 0.5
bk = 1.5
dk = 0.5
ek = ak
u[1]  = -1.0
u[2]  = -1.0+h
istep =  0



# The simplest algorithm for the Sturm-Liouville equation.
# Copyright (c) Tao Pang 1997.
def SMPL (n,h,ek,u):
  h2 = 2.0*h**2
  q = ek*(1.0+ek)
  i = 2
#  for i in range(2, n-1):
  while i < n : 
    x  =  (i-1)*h-1.0
    p  =  2.0*(1.0-x*x)
    p1 = -2.0*x*h
    u[i+1] = ((2.0*p-h2*q)*u[i]+(p1-p)*u[i-1])/(p1+p)
    i+=1

SMPL (n,h,ek,u)
f0 = u[n]-1.0

#print header for output
str_fmt = "{:^8} {:^3}{:^16} {:>6}"
print(str_fmt.format('istep','ek','dk','f1'))


# Bisection method for the root

while (abs(dk)>dl):
    ek = (ak+bk)/2.0
    SMPL (n,h,ek,u)
    f1 = u[n]-1.0
    if ((f0*f1)<0):
      bk = ek
      dk = bk-ak
    else:
      ak = ek
      dk = bk-ak
      f0 = f1
    istep = istep+1
    
    print("{:5d} {:10f} {:e} {:f} ".format(istep,ek,dk,f1))
  

#Note : ek = l, dk = 
      