#Author : Fasil Cheema
#Purpose: This module should verify the common input to the main function calls in the library.
#Date   : March 20, 2024

import numpy as np 
from paramconfig import paramconfig
import constants

class InputVerifier:
    '''
    A class designed to verify the input to the optimization library.
    '''
    def __init__(self):
        self.valid   = True
        self.dim     = 0
        self.err_msg = "Input Error! The following reasons caused an error: "
    
        self.MAX_DIM = constants.MAX_DIM
        self.ERR_THRESH = constants.ERR_THRESH

    def inputverify(self, A_mat, b_vec, c, x_0, H_0, B_0, step_size, max_s, min_err):
        '''
        The main function of the Verifier; we enumerate the possible error cases users can 
        supply to the library and verify here that the inputs are valid.
        '''
        if not(isinstance(A_mat,np.ndarray)):
            self.valid   = False
            err_str1 = "\n A_mat is not a valid numpy array "
            self.err_msg += err_str1
        else: 
            self.dim = A_mat.shape[0]

            #Checks if square matrix
            if self.dim != A_mat.shape[1]:
                self.valid = False
                err_str2 = "\n A_mat is not a square matrix "
                self.err_msg += err_str2

            #Checks if less than max dimensionality
            if self.dim > self.MAX_DIM:
                self.valid = False
                err_str3 = "\n The dimension of the problem exceeds the max dimension "
                self.err_msg += err_str3
        
        #checks if b is a numpy array
        if not(isinstance(b_vec,np.ndarray)):
            self.valid   = False
            err_str4 = "\n b_vec is not a valid numpy array "
            self.err_msg += err_str4
        else:  
        
            if (b_vec.shape[0] != 1) or (b_vec.ndim != 2):
                self.valid   = False
                err_str5 = "\n b_vec is not a row vector "
                self.err_msg += err_str5
            
            if self.dim != b_vec.shape[1]:
                self.valid   = False
                err_str6 = "\n b_vec does not have a valid shape "
                self.err_msg += err_str6

        #Note that in this check we do not use the isinstance() method
        #   This is due to python's strange behaviour where it accepts
        #   bools as integers. Using type we avoid this.  
        if not((type(c) is float) or (type(c) is int)):
            self.valid   = False
            err_str7 = "\n c should be a scalar real (float or int) "
            self.err_msg += err_str7
        
        #checks if x_0 is a numpy array
        if not(isinstance(x_0,np.ndarray)):
            self.valid   = False
            err_str8 = "\n x_0 is not a valid numpy array "
            self.err_msg += err_str8
        else: 
            if (x_0.ndim != 2) and (x_0.shape[1] == 1):
                self.valid   = False
                err_str9 = "\n x_0 is not a column vector "
                self.err_msg += err_str9
            
            if self.dim != x_0.shape[0]:
                self.valid   = False
                err_str10 = "\n x_0 does not have a valid shape "
                self.err_msg += err_str10
        
        #Verification for the H_0 parameter
        if not(isinstance(H_0,np.ndarray)):
            self.valid   = False
            err_str11 = "\n H_0 is not a valid numpy array "
            self.err_msg += err_str11
        else: 
            #Checks if appropriate dimensionality
            if self.dim != H_0.shape[0]:
                self.valid = False
                err_str12 = "\n H_0 is not of the appropriate dimension "
                self.err_msg += err_str12

            #Checks if square matrix
            if not((H_0.shape[0] == H_0.shape[1]) and (H_0.ndim == 2)) :
                self.valid = False
                err_str13 = "\n H_0 is not a square matrix "
                self.err_msg += err_str13
        
        #Verification for the B_0 parameter
        if not(isinstance(B_0,np.ndarray)):
            self.valid   = False
            err_str14 = "\n B_0 is not a valid numpy array "
            self.err_msg += err_str14
        else: 
            #Checks if appropriate dimensionality
            if self.dim != B_0.shape[0]:
                self.valid = False
                err_str15 = "\n B_0 is not of the appropriate dimension "
                self.err_msg += err_str15

            #Checks if square matrix
            if not((B_0.shape[0] == B_0.shape[1]) and (B_0.ndim == 2)) :
                self.valid = False
                err_str16 = "\n B_0 is not a square matrix "
                self.err_msg += err_str16

        #Note that in this check we do not use the isinstance() method
        #   This is due to python's strange behaviour where it accepts
        #   bools as integers. Using type we avoid this.  
        if not((type(step_size) is float) or (type(step_size) is int)):
            self.valid   = False
            err_str17 = "\n stepsize should be a scalar real (float or int) "
            self.err_msg += err_str17
        elif (step_size != -1) and (step_size <= 0):
            self.valid = False
            err_str18 = "\n stepsize should be a positive real (or -1)"
            self.err_msg += err_str18

        if not(isinstance(max_s, int)):
            self.valid   = False
            err_str19 = "\n max steps should be an integer"
            self.err_msg += err_str19
        else:

            if max_s <= 0:
                self.valid   = False
                err_str20 = "\n stepsize should be a natural number "
                self.err_msg += err_str20

        if not(isinstance(min_err, float)):
            self.valid   = False
            err_str21 = "\n min error should be a scalar real (float) "
            self.err_msg += err_str21
        else:
            
            if (min_err <= 0) or (min_err >= 1):
                self.valid   = False
                err_str22 = "\n min error should be between 0 and 1 (exclusive) "
                self.err_msg += err_str22
            elif not(min_err >= self.ERR_THRESH):
                self.valid = False
                err_str23 = "\n min error is below valid threshold "
                self.err_msg += err_str23

        
        return self.valid, self.err_msg


        

            

            

            



        

            

