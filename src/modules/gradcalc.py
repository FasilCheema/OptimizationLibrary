#Author : Fasil Cheema
#Purpose: This module is used to calculate the gradient, specifically for a quadratic of the form x^T A x + b x + c
#Date   : March 20, 2024

from vecmath import vecAdd, vecProd, vecT

def compute_gradient(x_t, parameters):

    A_mat = parameters.A_mat
    b_vec = parameters.b_vec

    #Must convert the row vector that is b_vec into a column vector
    #   this is a result for the analytic result of the gradient
    b_vec = vecT(b_vec)

    grad = vecAdd(vecProd(2*A_mat,x_t),b_vec)

    return grad

