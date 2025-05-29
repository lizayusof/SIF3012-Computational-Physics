#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:18:33 2024

@author: norhaslizayusof
"""
#solve linear algebra for Ax =b (2x2)

#import library
import numpy as np



#matrix A
A = np.array([[1,2],[3,5]]) #(2x2)
b = np.array([1,2]) #(1x2)

#solve Ax=b
x = np.linalg.solve(A,b)

print('x=',x)

test = np.allclose(np.dot(A,x),b)
print(test)