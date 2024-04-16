import numpy as np
import optlib 

A_mat = np.array([[1,0.5],[0.5,1.5]])
b_vec = np.array([[-3,-4]])
c = 0
x_0 = np.array([[0],[1]])
H_0 = np.array([[1,0],[0,1]])
B_0 = H_0
optimizer = optlib.optimizer()
result = optimizer.FRCG(A_mat,b_vec,c,x_0,1,2)
print(result)
result = optimizer.BFGS(A_mat,b_vec,c,x_0,H_0,1,2)
print(result)
result = optimizer.DFP(A_mat,b_vec,c,x_0,B_0,1,2)
print(result)