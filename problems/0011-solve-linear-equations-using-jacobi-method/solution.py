import torch

TOLERANCE = 1e-8

def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    # Your implementation here
    x_t = torch.zeros_like(b_t)
    D_t = torch.diag(A_t)
    R_t = A_t - torch.diag(D_t)

    for i in range(n):

        x_new_t = (b_t - R_t @ x_t) / D_t 
        if torch.linalg.norm(x_new_t - x_t) < TOLERANCE: 
            return x_new_t 

        x_t = x_new_t 

    return x_t
 