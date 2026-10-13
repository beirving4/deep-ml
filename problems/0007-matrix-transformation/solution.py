import numpy as np

def is_invertible(matrix: np.ndarray) -> bool: 
	if matrix.shape[0] != matrix.shape[1]: return False
	if np.isclose(np.linalg.det(matrix), 0.0): return False
	return True

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	
	A, T, S = np.asarray(A), np.asarray(T),np.asarray(S)

	if (not is_invertible(T)) or (not is_invertible(S)):
		return -1

	return np.linalg.inv(T) @ A @ S