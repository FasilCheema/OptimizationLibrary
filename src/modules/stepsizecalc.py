import numpy as np 
from gradcalc import gradCalc
import vecmath


def exactStep(x_t,s_t, parameters):
    
    numerator = vecmath.vecProd(vecmath.vecT(gradCalc(x_t,parameters)),s_t)
    denominator = vecmath.vecProd(vecmath.vecProd(vecmath.vecT(s_t),parameters.A_mat), s_t )

    alpha_t = - numerator/denominator

    return alpha_t