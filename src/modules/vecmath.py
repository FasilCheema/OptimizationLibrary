#Author : Fasil Cheema
#Purpose: This module contains all vector math functions relevant to the library.
#Date   : March 20, 2024


import numpy as np

def vecAdd(mat1, mat2):

    sum = np.add(mat1,mat2)

    return sum 

def vecProd(mat1, mat2):

    prod = np.multiply(mat1, mat2)

    return prod

def inverse(mat1):

    inv = np.linalg.inv(mat1)

    return inv

def vecDot(vec1, vec2):

    dotprod = np.dot(vec1,vec2)

    return dotprod

def vecT(mat1):

    tpose = np.transpose(mat1)

    return tpose

def vecNorm(vec1):

    norm = np.linalg.norm(vec1)

    return norm


