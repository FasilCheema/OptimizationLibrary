#Author : Fasil Cheema
#Purpose: This module is the main module for the optimizers here the call is made and all helper modules are executed.
#Date   : March 20, 2024

import vecmath
from paramconfig import paramconfig
import searchdir
import gradcalc
from inputverify import InputVerifier
import numpy as np
from stepsizecalc import exactStep
from abc import ABCMeta, abstractmethod
import constants



class optimizer:
    '''
    This class is the optimizer it has common state variables for each
    iteration that is computed for a specific algorithm. From this class
    a user can choose to call one of three optimizers BFGS,DFP, or FRCG.
    The class instance will update the current solution, the current step
    size, the current Hessian/Inv Hessian matrices, search direction, 
    and current step. The output of each function call will be the final 
    solution unless invalid parameters are supplied.
    '''

    def update(self,alpha_t,s_t,x_t,H_t,B_t,t):
        '''
        After each iteration updates the state of the optimizer. 
        '''
        self.alpha_t = alpha_t
        self.s_t     = s_t
        self.x_t     = x_t
        self.H_t     = H_t
        self.B_t     = B_t
        self.t       = t

    def BFGS(self,A_mat,b_vec,c,x_0,H_0,step_size = -1, max_s = constants.MAX_STEP, min_err = constants.ERR_THRESH):
        '''
        The BFGS algorithm; one of the minimizers. It is a Quasi-Newton method.
        '''
        #The InputVerification and paramconfig record will hold relevant values. Depending on the method if a user chooses 
        #   a Quasi-Newton method (DFP or BFGS) they need to compute either a Hessian or an inverse Hessian approximate. 
        #   This line acts like a placeholder so the paramconfig record still has a value stored for the approximate method not used.
        B_0 = np.zeros(A_mat.shape)

        #Verify the validity of the input parameters 
        validator = InputVerifier()
        valid, err_msg = validator.inputverify(A_mat,b_vec,c,x_0,H_0,B_0,step_size,max_s,min_err)
        
        if not(valid is True):

            print(err_msg)
        
        else:
            
            #If the parameters are valid we instantiate and fill a paramconfig object, to hold our current
            #   parameter configuration. This object is used to easily pass around parameters to the rest of
            #   the project. 
            
            parameters = paramconfig(A_mat,b_vec,c,x_0,H_0,B_0,step_size,max_s,min_err)

            Hessian = H_0
            curr_sol = x_0
            
            if step_size == -1:
                step_size_flag = True
            else:
                step_size_flag = False

            for i in range(max_s):

                search_dir = searchdir.dirBFGS(Hessian,curr_sol, parameters)
                
                if step_size_flag:
                    step_size = exactStep(curr_sol,search_dir,parameters)
                
                self.update(step_size,search_dir,curr_sol,Hessian,B_0,i)

                curr_sol = vecmath.vecAdd(curr_sol,step_size*search_dir)
                
                Hessian = self.computeHessianBFGS(curr_sol,parameters)

                grad = gradcalc.compute_gradient(curr_sol,parameters)
                
                grad_norm = vecmath.vecNorm(grad) 
                
                if grad_norm  <= min_err:
                    
                    return curr_sol
                
            return curr_sol
                

    def DFP(self,A_mat,b_vec,c,x_0,B_0,step_size = -1, max_s = constants.MAX_STEP, min_err = constants.ERR_THRESH):
        '''
        The DFP algorithm; one of the minimizers. It is a Quasi-Newton method.
        '''
        H_0 = np.zeros(A_mat.shape)
        #The InputVerification and paramconfig record will hold relevant values. Depending on the method if a user chooses 
        #   a Quasi-Newton method (DFP or BFGS) they need to compute either a Hessian or an inverse Hessian approximate. 
        #   This line acts like a placeholder so the paramconfig record still has a value stored for the approximate method not used.

        #Verify the validity of the input parameters 
        validator = InputVerifier()
        valid, err_msg = validator.inputverify(A_mat,b_vec,c,x_0,H_0,B_0,step_size,max_s,min_err)
        
        if not(valid is True):

            print(err_msg)
        
        else:
            
            #If the parameters are valid we instantiate and fill a paramconfig object, to hold our current
            #   parameter configuration. This object is used to easily pass around parameters to the rest of
            #   the project. 
            
            parameters = paramconfig(A_mat,b_vec,c,x_0,H_0,B_0,step_size,max_s,min_err)
            
            InvHessian = B_0
            curr_sol = x_0
            
            if step_size == -1:
                step_size_flag = True
            else:
                step_size_flag = False

            for i in range(max_s):

                search_dir = searchdir.dirDFP(InvHessian,curr_sol, parameters)
                
                if step_size_flag:
                    step_size = exactStep(curr_sol,search_dir,parameters)
                
                self.update(step_size,search_dir,curr_sol,H_0,InvHessian,i)
                
                curr_sol = vecmath.vecAdd(curr_sol,step_size*search_dir)   
                InvHessian = self.computeInvHessianDFP(curr_sol,parameters)
                
                grad = gradcalc.compute_gradient(curr_sol,parameters)
                
                grad_norm = vecmath.vecNorm(grad) 
                
                if grad_norm  <= min_err:
                    
                    return curr_sol
                
            return curr_sol
                
    def FRCG(self,A_mat,b_vec,c,x_0,step_size = -1, max_s = constants.MAX_STEP, min_err = constants.ERR_THRESH):
        '''
        The Fletcher Reeves algorithm; one of the minimizers. It is a 
        Conjugate Gradient Descent method.
        '''
        H_0 = np.zeros(A_mat.shape)
        B_0 = np.zeros(A_mat.shape)

        #Verify the validity of the input parameters 
        validator = InputVerifier()
        valid, err_msg = validator.inputverify(A_mat,b_vec,c,x_0,H_0,B_0,step_size,max_s,min_err)
        
        if not(valid is True):

            print(err_msg)
        
        else:
            
            #If the parameters are valid we instantiate and fill a paramconfig object, to hold our current
            #   parameter configuration. This object is used to easily pass around parameters to the rest of
            #   the project. 
            parameters = paramconfig(A_mat,b_vec,c,x_0,H_0,B_0,step_size,max_s,min_err)

            curr_sol = x_0

            
            #Use exact step 
            if step_size == -1:
                step_size_flag = True
            else:
                step_size_flag = False

            for i in range(max_s):

                if i == 0:
                    search_dir = -gradcalc.compute_gradient(curr_sol, parameters)
                else:
                    search_dir = searchdir.dirFRCG(curr_sol,self.x_t,self.s_t,parameters)
                    
                if step_size_flag:
                    step_size = exactStep(curr_sol,search_dir,parameters)

                self.update(step_size,search_dir,curr_sol,H_0,B_0,i)
                    
                curr_sol = vecmath.vecAdd(curr_sol,search_dir*step_size)

                grad = gradcalc.compute_gradient(curr_sol,parameters)

                grad_norm = vecmath.vecNorm(grad)
                
                
                if grad_norm  <= min_err:
                    
                    return curr_sol
                
            return curr_sol


    def computeHessianBFGS(self,x_new,parameters):

        y_t = vecmath.vecAdd(gradcalc.compute_gradient(x_new,parameters), -gradcalc.compute_gradient(self.x_t,parameters)) 
        p_t = self.alpha_t * self.s_t

        numerator1   = vecmath.vecProd(y_t,vecmath.vecT(y_t))
        denominator1 = vecmath.vecDot(y_t, p_t)
        term1 = (1/denominator1) * numerator1

        numerator2 = vecmath.vecProd(self.H_t,p_t)
        numerator2 = vecmath.vecProd(numerator2,vecmath.vecT(p_t))
        numerator2 = vecmath.vecProd(numerator2,vecmath.vecT(self.H_t))

        denominator2 = vecmath.vecProd(vecmath.vecT(p_t),self.H_t)
        denominator2 = vecmath.vecProd(denominator2, p_t)

        term2 = (1/denominator2)*numerator2

        H_new = self.H_t + term1 - term2

        return H_new

    def computeInvHessianDFP(self,x_new,parameters):
        
        y_t = vecmath.vecAdd(gradcalc.compute_gradient(x_new,parameters), -gradcalc.compute_gradient(self.x_t,parameters)) 
        p_t = self.alpha_t * self.s_t

        numerator1   = vecmath.vecProd(p_t,vecmath.vecT(p_t))
        denominator1 = vecmath.vecDot(p_t, y_t)
        term1 = (1/denominator1) * numerator1

        numerator2 = vecmath.vecProd(self.B_t,y_t)
        numerator2 = vecmath.vecProd(numerator2,vecmath.vecT(y_t))
        numerator2 = vecmath.vecProd(numerator2,vecmath.vecT(self.B_t))

        denominator2 = vecmath.vecProd(vecmath.vecT(y_t),self.B_t)
        denominator2 = vecmath.vecProd(denominator2, y_t)

        term2 = (1/denominator2)*numerator2

        B_new = self.B_t + term1 - term2

        return B_new

