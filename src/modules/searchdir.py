#Author : Fasil Cheema
#Purpose: This module is used to calculate the search direction for our optimizer
#Date   : March 20, 2024

import vecmath
from gradcalc import compute_gradient

def diralg(Mat, x_t, parameters):
    '''
    Design Pattern: This is a design pattern common to the Newton optimization algorithms
    in this function we compute the current search direction given a matrix (usually a Hessian 
    or inverse Hessian). This direction is simply the inverse matrix times the gradient of the 
    current search direction. 
    '''

    inv_Mat  = vecmath.inverse(Mat)
    gradient = compute_gradient(x_t,parameters)

    s_t = - vecmath.vecProd(inv_Mat,gradient)

    return s_t

def betaFR(x_t, x_prev,parameters):

    
    curr_gradient = compute_gradient(x_t, parameters)
    prev_gradient = compute_gradient(x_prev, parameters)

    numerator   = vecmath.vecNorm(curr_gradient)
    numerator   = numerator**2

    denominator = vecmath.vecNorm(prev_gradient)
    denominator = denominator**2

    beta_t = numerator/denominator

    return beta_t

def dirBFGS(H_t, x_t, parameters):

    inv_H     = vecmath.inverse(H_t)
    gradient  = compute_gradient(x_t, parameters)

    s_t = - vecmath.vecProd(inv_H, gradient)

    return s_t

def dirDFP(B_t, x_t, parameters):

    gradient = compute_gradient(x_t,parameters)

    s_t = - vecmath.vecProd(B_t,gradient)

    return s_t

def dirFRCG(x_t, x_prev, s_prev, parameters):

    beta_t   = betaFR(x_t,x_prev,parameters)
    tmp_vec  = beta_t*s_prev

    neg_gradient = -compute_gradient(x_t,parameters)


    s_t = vecmath.vecAdd(neg_gradient, tmp_vec)

    return s_t

