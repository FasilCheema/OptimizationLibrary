import numpy as np 
from gradcalc import compute_gradient
import vecmath


def exactStep(x_t,s_t, parameters):
    '''
    Function to compute the exact step size for a search algorithm given a search direction (s_t) and a point (x_t).
    For the quadratic form this value has an analytic solution (see MIS), this analytic solution for the quadratic 
    form is computed here.
    '''
    numerator_array = vecmath.vecProd(vecmath.vecT(compute_gradient(x_t,parameters)),s_t)
    denominator_array = vecmath.vecProd(vecmath.vecProd(vecmath.vecT(s_t),2*parameters.A_mat), s_t )

    numerator = numerator_array.item()
    denominator = denominator_array.item()
    
    alpha_t = - numerator/denominator

    return alpha_t