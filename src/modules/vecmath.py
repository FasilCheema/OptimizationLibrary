#Author : Fasil Cheema
#Purpose: This module contains all vector math functions relevant to the library.
#Date   : March 20, 2024


import numpy as np

def vecAdd(mat1, mat2):
    '''
    Function to add matrices/vectors of the same dimensions
    '''
    sum_result = np.add(mat1,mat2)

    return sum_result 

def vecProd(mat1, mat2):
    '''
    Function to multiply matrices/vectors of the same dimensions
    '''
    prod = np.matmul(mat1, mat2)

    return prod

def inverse(mat1):
    '''
    Function to take the inverse of a square matrix
    '''
    inv = np.linalg.inv(mat1)

    return inv

def vecDot(vec1, vec2):
    '''
    Function to take the dot product of 2 vectors of the same dimension
    '''
    vec1T   = vecT(vec1)

    dotprod_arr = np.matmul(vec1T,vec2)

    dot_prod = dotprod_arr.item() 

    return dot_prod

def vecT(mat1):
    '''
    Function to take the transpose of a vector/matrix
    '''
    tpose = np.transpose(mat1)

    return tpose

def vecNorm(vec1):
    '''
    Function to take the l2 norm of a vector
    '''
    norm = np.linalg.norm(vec1)

    return norm


