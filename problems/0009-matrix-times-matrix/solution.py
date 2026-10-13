import numpy as np

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
	
    A, B = np.asarray(a), np.asarray(b)
    if A.shape[1] != B.shape[0]: return -1 
    return A @ B