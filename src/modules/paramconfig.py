#Author : Fasil Cheema
#Purpose: This module should contain the relevant parameters to the library.
#Date   : March 20, 2024

class paramconfig:
    '''
    This is an imporant class, when instantiated this gives us an instance 
    that contains the relevant information to our specific problem. This defines the 
    quadratic form and initial inputs that once provided are not changed throughout the 
    running of the library but are critical as they define the behavior of the library.
    The instance acts as a quick method 
    '''
    
    def __init__(self,A_mat,b_vec,c,x_0,H_0,B_0,step_size,max_s,min_err):
        
        self.A_mat = A_mat
        self.b_vec = b_vec
        self.c     = c 
        self.x_0   = x_0
        self.H_0   = H_0
        self.B_0   = B_0
        self.step_size = step_size
        self.max_s = max_s
        self.min_err = min_err

