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
#u = np.zeros (1000)
# Initialization of the problem
u = np.arange(0,502)

dl = 1.0E-06
h  = 2.0/(n-1)
ak = 0.5
bk = 1.5
dk = 0.5
ek = ak
u[1]  = -1.0
u[2]  = -1.0+h
istep =  0


def SMPL (n,h,ek,u):
#  print(ek)
# The simplest algorithm for the Sturm-Liouville equation.
# Copyright (c) Tao Pang 1997.
  h2 = 2.0*h**2
  q = ek*(1.0+ek)
  for i in range(2, n-1):
    x  =  (i-1)*h-1.0
    p  =  2.0*(1.0-x*x)
    p1 = -2.0*x*h
    u[i+1] = ((2.0*p-h2*q)*u[i]+(p1-p)*u[i-1])/(p1+p)

    
SMPL (n,h,ek,u)
f0 = u[n]-1.0


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
#    print('dk=',dk,'dl=',dl)
    
    print(istep,ek,dk,f1)
  

