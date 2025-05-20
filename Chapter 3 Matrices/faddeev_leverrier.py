'''Implementation of the simple Faddeev-Leverrier algorithm for
finding the coefficients of the characteristic polynomial of a
matrix, we draw only on Numpy.'''

#The original code is taken from https://github.com/RJTK/faddeev_leverrier/blob/master/faddeev_leverrier.py
#but has been modified to run using the current python version (3.11)


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
    A = np.array(A, dtype=float)  # Ensure float type
    n = A.shape[0]
    Ak = A.copy()
    coeffs = [1.0]  # a0 = 1

    for k in range(1, n + 1):
        ak = -np.trace(Ak) / k
        coeffs.append(ak)
        if k != n:
            Ak += np.diag(np.repeat(ak, n))  # Add ak * I
            Ak = A @ Ak  # Matrix multiplication

    return np.array(coeffs)

if __name__ == '__main__':
  A = np.array([[1., 2.], [3., 4.]])
  print (faddeev_leverrier(A))
  
  A = [[2,1,4,5,1,-3],
       [-1,0,2,9,4,5],
       [3,4,-1,-1,-1,-1],
       [1,2,3,4,5,6],
       [-6,-5,-4,-3,-2,-1],
       [1,-1,10,-10,100,-100]]
  
  print (faddeev_leverrier(A))
