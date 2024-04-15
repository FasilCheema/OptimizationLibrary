import numpy as np 
from gradcalc import compute_gradient
import vecmath


def exactStep(x_t,s_t, parameters):
    
    numerator_array = vecmath.vecProd(vecmath.vecT(compute_gradient(x_t,parameters)),s_t)
    denominator_array = vecmath.vecProd(vecmath.vecProd(vecmath.vecT(s_t),2*parameters.A_mat), s_t )

    numerator = numerator_array.item()
    denominator = denominator_array.item()
    
    alpha_t = - numerator/denominator

    return alpha_t