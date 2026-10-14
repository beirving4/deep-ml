import numpy as np

TOLERANCE = 1e-8

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here
    m, n = A.shape 

    if (m >= n): 
        raw_eig_vals, raw_eig_vecs = np.linalg.eigh(A.T @ A)
        order = np.argsort(raw_eig_vals)[::-1]
        eig_vals, V = raw_eig_vals[order], raw_eig_vecs[:, order]
        S = np.sqrt(np.maximum(eig_vals, 0.0))
        mask = S > TOLERANCE
        S, V = S[mask], V[:, mask]
        U = (A @ V) / S
    else: 
        raw_eig_vals, raw_eig_vecs = np.linalg.eigh(A @ A.T) 
        order = np.argsort(raw_eig_vals)[::-1]
        eig_vals, U = raw_eig_vals[order], raw_eig_vecs[:, order]
        S = np.sqrt(np.maximum(eig_vals, 0.0))
        mask = S > TOLERANCE
        S, U = S[mask], U[:, mask]
        V = (A.T @ U) / S

    return U, S, V.T
