import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here
    match mode:
		case "row":
			agg_dim = 1 
		case "column":
			agg_dim = 0
		case _: 
			raise ValueError("mode not recognized! try 'row' or 'column' ")
    
    return a_t.mean(axis=agg_dim)
    
