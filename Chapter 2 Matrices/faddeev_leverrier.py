#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Implementation of the simple Faddeev-Leverrier algorithm for
finding the coefficients of the characteristic polynomial of a
matrix, we draw only on Numpy.'''

#This algorithm is very simple, we have drawn on the paper below:

# @article{Helmberg1993219,
# title = "On Faddeev-Leverrier's method for the computation of the characteristic polynomial of a matrix and of eigenvectors ",
# journal = "Linear Algebra and its Applications ",
# volume = "185",
# number = "",
# pages = "219 - 233",
# year = "1993",
# note = "",
# issn = "0024-3795",
# doi = "http://dx.doi.org/10.1016/0024-3795(93)90214-9",
# url = "//www.sciencedirect.com/science/article/pii/0024379593902149",
# author = "Gilbert Helmberg and Peter Wagner and Gerhard Veltkamp"
# }

import numpy as np

def faddeev_leverrier(A):
    '''
    Given an n x n matrix A, we return the coefficients of it's
    characteristic polynomial

    P(x) ^= det(xI - A) = a_0 * x^n + a_1 * x^(n - 1) + ... + a_n

    It is a property of P(x) that a_0 = 1, a_n = det(A)

    We return the list a = [a_0, a_1, ..., a_n]
    '''
    A = np.array(A) #Ensure we have a numpy array
    n = A.shape[0]
    assert A.shape[1] == n, 'Array must be square!'

    a = np.array([1.])
    Ak = np.array(A)
    for k in range(1, n + 1):
        ak = -Ak.trace() / k
        a = np.append(a, ak)
        Ak = Ak + np.diag(np.repeat(ak, n))
        Ak = np.dot(A, Ak)
    return a

if __name__ == '__main__':
  A = np.array([[1., 2.], [3., 4.]])
  print (faddeev_leverrier(A))
  
  A = [[2.,1.,4.,5.,1.,-3.],
       [-1.,0.,2.,9.,4.,5.],
       [3.,4.,-1.,-1.,-1.,-1.],
       [1.,2.,3.,4.,5.,6.],
       [-6.,-5.,-4.,-3.,-2.,-1.],
       [1.,-1.,10.,-10.,100.,-100.]]
 
  print (faddeev_leverrier(A))