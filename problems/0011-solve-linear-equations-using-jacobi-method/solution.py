import numpy as np

TOLERANCE = 1e-8

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	x = np.zeros_like(b)
	D = np.diag(A)
	R = A - np.diag(D)

	for i in range(n): 
		x_new = (b - R @ x) / D 
		if np.linalg.norm(x_new - x) < TOLERANCE: 
			return x_new 

		x = x_new

	return x

