import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    
    A, B = torch.tensor(a), torch.tensor(b)
    if A.shape[-1] != B.shape[0]: return torch.tensor(-1)
    return A @ B
