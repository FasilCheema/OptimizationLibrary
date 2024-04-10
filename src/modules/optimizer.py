#Author : Fasil Cheema
#Purpose: This module is the main module for the optimizers here the call is made and all helper modules are executed.
#Date   : March 20, 2024

import vecmath
from paramconfig import paramconfig
import searchdir
import gradcalc
from inputverify import InputVerifier
import numpy as np
from stepsizecalc import stepsizeCalc


class optimizer:

    def __init__(self, alpha_t,s_t,x_t,H_t,B_t,t):
        self.alpha_t = alpha_t
        self.s_t     = s_t
        self.x_t     = x_t
        self.H_t     = H_t
        self.B_t     = B_t
        self.t       = t

    def update(self,alpha_t,s_t,x_t,H_t,B_t,t):
        self.alpha_t = alpha_t
        self.s_t     = s_t
        self.x_t     = x_t
        self.H_t     = H_t
        self.B_t     = B_t
        self.t       = t

    def optimization_algo(self, A_mat, b_vec, c, x_0, H_0, B_0, step_size, max_s, min_err):
        '''
        Design Pattern: This is a design pattern common to the optimization algorithms in this 
        module. This function takes in parameters (mostly) common to all optimization algorithm 
        in this repository.  
        '''

        #Verify the validity of the input paramters 
        validator = InputVerifier()
        valid, err_msg = validator.inputverify(A_mat,b_vec,c,x_0,H_0,B_0,step_size,max_s,min_err)

        if not(valid is True):
            
            print(err_msg)

        else:
            #If the parameters are valid we instantiate and fill a paramconfig object, to hold our current
            #   parameter configuration. This object is used to easily pass around parameters to the rest of
            #   the code. 
            parameters = paramconfig(A_mat,b_vec,c,x_0,H_0,B_0,step_size,max_s,min_err)
        
            self.H_t = H_0
            self.x_t = x_0

            for i in range(max_s):

                s_t = searchdir.diralg(self.H_t,self.x_t, parameters)
                
                if step_size != -1:
                    step_size = stepsizeCalc(self.x_t,self.s_t,parameters)
                else: 
                    step_size == 1

                self.alpha_t = step_size
                
                x_new = vecmath.vecAdd(self.x_t,step_size*s_t)

                grad = gradcalc.gradCalc(x_new, parameters)

                if grad <= min_err:
                    
                    self.update(step_size,s_t,x_new,H_t,B_0,i)

                    return x_new
                
                H_t = self.computeApproxMatOptAlg(self.H_t,x_new,self.x_t,self.s_t,self.step_size)
                self.update(step_size,s_t,x_new,H_t,B_0, i)

            return x_new
    
    def computeApproxMatOptAlg(self,Mat,x_new,x_t,s_t,alpha_t):

        y_t = vecmath.vecAdd(vecmath.gradCalc(x_new), -vecmath.gradCalc(x_t)) 
        p_t = alpha_t*s_t

        numerator1   = vecmath.vecProd(y_t,vecmath.vecT(y_t))
        denominator1 = vecmath.vecDot(y_t, p_t)
        term1 = (1/denominator1) * numerator1

        numerator2 = vecmath.vecProd(Mat,p_t)
        numerator2 = vecmath.vecProd(numerator2,vecmath.vecT(p_t))
        numerator2 = vecmath.vecProd(numerator2,vecmath.vecT(Mat))

        denominator2 = vecmath.vecProd(vecmath.vecT(p_t),Mat)
        denominator2 = vecmath.vecProd(denominator2, p_t)

        term2 = (1/denominator2)*numerator2

        Mat_new = Mat + term1 - term2

        return Mat_new

    def BFGS(self,A_mat,b_vec,c,x_0,H_0,step_size = -1, max_s = 10000, min_err = 0.005):

        B_0 = np.zeros(A_mat.shape)

        validator = InputVerifier()
        valid, err_msg = validator.inputverify(A_mat,b_vec,c,x_0,H_0,B_0,step_size,max_s,min_err)
        
        if valid == False:
            print(err_msg)
        else:
            parameters = paramconfig(A_mat,b_vec,c,x_0,H_0,B_0,step_size,max_s,min_err)

            for i in range(max_s):

                if i == 0:
                    self.H_t = H_0
                    self.x_t = x_0

                if step_size != -1:
                    step_size = stepsizeCalc(self.x_t,self.s_t,parameters)
                else:
                    step_size == 1
                
                s_t = searchdir.dirBFGS(self.H_t,self.x_t, parameters)

                x_new = vecmath.vecAdd(self.x_t,step_size*s_t)

                grad = gradcalc.gradCalc(self.x_new)

                if grad <= min_err:
                    
                    self.update(step_size,s_t,x_t,H_t,B_t)

                    return x_new
                
                H_t = self.computeHessianBFGS(self.H_t,self.x_new,self.x_t,self.s_t,step_size)
                self.update(step_size,s_t,x_new,H_t,B_0)
                

    def DFP(self,A_mat,b_vec,c,x_0,B_0,step_size = -1, max_s = 10000, min_err = 0.005):

        H_0 = np.zeros(A_mat.shape)

    def FRCG(self,A_mat,b_vec,c,x_0,step_size = -1, max_s = 10000, min_err = 0.005):
        H_0 = np.zeros(A_mat.shape)
        B_0 = np.zeros(A_mat.shape)


    def computeHessianBFGS(self,H_t,x_new,x_t,s_t,alpha_t):

        y_t = vecmath.vecAdd(vecmath.gradCalc(x_new), -vecmath.gradCalc(x_t)) 
        p_t = alpha_t*s_t

        numerator1   = vecmath.vecProd(y_t,vecmath.vecT(y_t))
        denominator1 = vecmath.vecDot(y_t, p_t)
        term1 = (1/denominator1) * numerator1

        numerator2 = vecmath.vecProd(H_t,p_t)
        numerator2 = vecmath.vecProd(numerator2,vecmath.vecT(p_t))
        numerator2 = vecmath.vecProd(numerator2,vecmath.vecT(H_t))

        denominator2 = vecmath.vecProd(vecmath.vecT(p_t),H_t)
        denominator2 = vecmath.vecProd(denominator2, p_t)

        term2 = (1/denominator2)*numerator2

        H_new = H_t + term1 - term2

        return H_new

    def computeInvHessianDFP(self,B_t,x_new,x_t,s_t,alpha_t):
        
        y_t = vecmath.vecAdd(vecmath.gradCalc(x_new), -vecmath.gradCalc(x_t)) 
        p_t = alpha_t*s_t

        numerator1   = vecmath.vecProd(p_t,vecmath.vecT(p_t))
        denominator1 = vecmath.vecDot(p_t, y_t)
        term1 = (1/denominator1) * numerator1

        numerator2 = vecmath.vecProd(B_t,y_t)
        numerator2 = vecmath.vecProd(numerator2,vecmath.vecT(y_t))
        numerator2 = vecmath.vecProd(numerator2,vecmath.vecT(B_t))

        denominator2 = vecmath.vecProd(vecmath.vecT(y_t),B_t)
        denominator2 = vecmath.vecProd(denominator2, y_t)

        term2 = (1/denominator2)*numerator2

        B_new = B_t + term1 - term2

        return B_new

