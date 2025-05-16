import torch
import functools

def break_on_view(x: torch.Tensor, name=""):
    """
    Utility to detect if a tensor is a view of another tensor.
    
    Args:
        x: Tensor to check
        name: Name of the tensor for debugging output
        
    Returns:
        The input tensor
    """
    if x._base is not None:
        print(f"[VIEW-WARN] {name} is a view of its base -- clone before write!")
    return x

def safe_clone(x: torch.Tensor):
    """
    Safely clone a tensor, making a contiguous copy if it's a view.
    
    Args:
        x: Tensor to clone
        
    Returns:
        A new tensor that's not a view
    """
    if x._base is not None or not x.is_contiguous():
        return x.contiguous().clone()
    return x.clone()

def check_shapes(tensors, names=None):
    """
    Print shapes of tensors for debugging.
    
    Args:
        tensors: List of tensors
        names: Optional list of names for the tensors
    """
    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]
        
    for name, tensor in zip(names, tensors):
        shape_str = f"{tensor.shape}" if hasattr(tensor, 'shape') else "not a tensor"
        print(f"{name}: {shape_str}")

# Function decorator to safely clone all outputs
def safe_outputs(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        outputs = func(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            return safe_clone(outputs)
        elif isinstance(outputs, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in outputs):
            return type(outputs)(safe_clone(x) for x in outputs)
        return outputs
    return wrapper 