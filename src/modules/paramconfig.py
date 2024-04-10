#Author : Fasil Cheema
#Purpose: This module should contain the relevant parameters to the library.
#Date   : March 20, 2024


class paramconfig:
    def __init__(self,A_mat,b_vec,c,x_0,H_0,B_0,step_size,max_s,min_err):
        self.MAX_DIM  = 6
        self.MAX_STEP = 10000
        self.MIN_ERR  = 0.005

        self.A_mat = A_mat
        self.b_vec = b_vec
        self.c     = c 
        self.x_0   = x_0
        self.H_0   = H_0
        self.B_0   = B_0
        self.step_size = step_size
        self.max_s = max_s
        self.min_err = min_err

