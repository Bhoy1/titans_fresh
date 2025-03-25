"""
Simple MLP-based memory module for the Titans architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryMLP(nn.Module):
    """
    Simple MLP-based memory module.
    Uses a multi-layer perceptron as a memory representation.
    """
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        expansion_factor: float = 4.0,
        activation: nn.Module = nn.GELU
    ):
        """
        Initialize the MLP memory module.
        
        Args:
            dim: Input and output dimension
            depth: Number of layers in the MLP
            expansion_factor: Factor for hidden layer dimension
            activation: Activation function to use
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.expansion_factor = expansion_factor
        
        # Calculate hidden dimension
        dim_hidden = int(dim * expansion_factor)
        
        # Input and output dimensions for each layer
        dims = (dim, *((dim_hidden,) * (depth - 1)), dim)

        # Create weights for each layer
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(dim_in, dim_out)) 
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ])
        
        # Activation function
        self.activation = activation()

        # Initialize weights
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the memory MLP.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        for ind, weight in enumerate(self.weights):
            is_last = ind == len(self.weights) - 1
            
            # Apply weight matrix
            x = x @ weight
            
            # Apply activation for all except last layer
            if not is_last:
                x = self.activation(x)

        return x