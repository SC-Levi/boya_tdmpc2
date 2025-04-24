import torch
from torch.nn import Buffer


class MooreRunningScale(torch.nn.Module):
    """
    Enhanced running trimmed scale estimator with improved handling for different tensor shapes.
    Extends the functionality of the original RunningScale from TDMPC2.
    """

    def __init__(self, cfg):
        """
        Initialize the scale with default values and configuration.
        
        Args:
            cfg: Configuration object containing tau parameter
        """
        super().__init__()
        self.cfg = cfg
        
        # Get device from config if available, otherwise use CUDA if available
        if hasattr(cfg, 'device'):
            self.device = cfg.device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        # Initialize values
        self.value = Buffer(torch.ones(1, dtype=torch.float32, device=self.device))
        self._percentiles = Buffer(torch.tensor([5, 95], dtype=torch.float32, device=self.device))
        self.tau = getattr(cfg, 'tau', 0.01)  # Default tau if not provided

    @property
    def device(self):
        """Get the device of the scale."""
        return self.value.device
        
    def to(self, device):
        """
        Move scale buffers to the specified device.
        
        Args:
            device: Target device
            
        Returns:
            self: Updated scale instance
        """
        self.value = self.value.to(device)
        self._percentiles = self._percentiles.to(device)
        return self

    def state_dict(self):
        """Create a state dictionary for saving the scale state."""
        return dict(value=self.value, percentiles=self._percentiles)

    def load_state_dict(self, state_dict):
        """Load scale state from a state dictionary."""
        self.value = state_dict['value'].clone()
        self._percentiles = state_dict['percentiles'].clone()

    def _positions(self, x_shape):
        """
        Calculate positions for percentile calculation.
        
        Args:
            x_shape: Shape of the input tensor
            
        Returns:
            tuple: Indices and weights for interpolation
        """
        positions = self._percentiles * (x_shape-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled = torch.where(ceiled > x_shape - 1, x_shape - 1, ceiled)
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        return floored.long(), ceiled.long(), weight_floored.unsqueeze(1), weight_ceiled.unsqueeze(1)

    def _percentile(self, x):
        """
        Calculate percentiles of the input tensor with improved shape handling.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Percentile values
        """
        # Ensure input is at least 2D for proper flattening
        if x.ndim == 0:
            # Handle scalar inputs
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 1:
            # Handle 1D inputs
            x = x.unsqueeze(1)
            
        x_dtype, x_shape = x.dtype, x.shape
            
        try:
            # Flatten all dimensions except the first
            x_flat = x.flatten(1, x.ndim-1)
            
            # Sort values along the first dimension
            in_sorted = torch.sort(x_flat, dim=0).values
            
            # Get interpolation indices and weights
            floored, ceiled, weight_floored, weight_ceiled = self._positions(x_flat.shape[0])
            
            # Calculate interpolated percentiles
            d0 = in_sorted[floored] * weight_floored
            d1 = in_sorted[ceiled] * weight_ceiled
            
            # Reshape output to match input dimensions
            return (d0+d1).reshape(-1, *x_shape[1:]).to(x_dtype)
            
        except RuntimeError as e:
            # Provide detailed error message with shape information
            raise RuntimeError(f"Failed to calculate percentiles. Input shape: {x.shape}, "
                               f"ndim: {x.ndim}, device: {x.device}. Original error: {str(e)}")

    def update(self, x):
        """
        Update the running scale value.
        
        Args:
            x: Input tensor to update the scale
        """
        try:
            # Move input to the same device if needed
            if x.device != self.device:
                x = x.to(self.device)
                
            # Calculate percentiles from the detached input
            percentiles = self._percentile(x.detach())
            
            # Calculate scale as the difference between high and low percentiles
            value = torch.clamp(percentiles[1] - percentiles[0], min=1.)
            
            # FIXED: Update running estimate using explicit non-in-place calculation
            # instead of torch.lerp which might use in-place operations internally
            self.value = self.value * (1.0 - self.tau) + value * self.tau
            
        except Exception as e:
            print(f"Warning: Failed to update scale: {e}")
            # Don't update the scale in case of errors
            pass

    def forward(self, x, update=False):
        """
        Scale the input by the current scale value.
        
        Args:
            x: Input tensor to scale
            update: Whether to update the scale with this input
            
        Returns:
            torch.Tensor: Scaled input
        """
        # Update scale if requested
        if update:
            self.update(x)
            
        # Ensure input and scale are on the same device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Scale the input
        return x / self.value

    def __repr__(self):
        """String representation of the scale."""
        return f'MooreRunningScale(value: {self.value.item():.4f}, device: {self.device})' 