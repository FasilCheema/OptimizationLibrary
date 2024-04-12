#Author : Fasil Cheema
#Purpose: This module is used to calculate the gradient, specifically for a quadratic of the form x^T A x + b x + c
#Date   : March 20, 2024

from vecmath import vecAdd, vecProd

def compute_gradient(x_t, parameters):

    A_mat = parameters.A_mat
    b_vec = parameters.b_vec

    grad = 2 * vecAdd(vecProd(A_mat,x_t),b_vec)

    return grad

