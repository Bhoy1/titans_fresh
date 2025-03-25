"""
Simple neural memory module for the Titans architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any

from .memory_mlp import MemoryMLP

class SimpleNeuralMemory(nn.Module):
    """
    A simplified neural memory module compatible with Vision Transformer integration.
    Maintains state and implements momentum and forgetting mechanisms.
    """
    def __init__(
        self,
        dim: int,                    # Embedding dimension
        memory_depth: int = 2,       # Depth of memory model
        memory_dim: Optional[int] = None,  # Inner memory dimension
        momentum: bool = True,       # Use momentum for memory updates
        momentum_factor: float = 0.9,  # Momentum factor
        forget_factor: float = 0.1,  # Forget factor for memory decay
        integration_type: str = "mal"  # "mal", "mac", or "mag"
    ):
        """
        Initialize the simple neural memory module.
        
        Args:
            dim: Embedding dimension
            memory_depth: Depth of memory model
            memory_dim: Inner memory dimension (if None, same as dim)
            momentum: Whether to use momentum for memory updates
            momentum_factor: Momentum factor
            forget_factor: Forget factor for memory decay
            integration_type: Integration approach ("mal", "mac", "mag")
        """
        super().__init__()
        self.dim = dim
        self.memory_dim = memory_dim if memory_dim is not None else dim
        self.integration_type = integration_type
        self.momentum_enabled = momentum
        self.momentum_factor = momentum_factor
        self.forget_factor = forget_factor
        
        # Create dimension mapping if needed
        if self.memory_dim != dim:
            self.down_proj = nn.Linear(dim, self.memory_dim)
            self.up_proj = nn.Linear(self.memory_dim, dim)
            self.use_projection = True
        else:
            self.use_projection = False
        
        # Create memory model
        self.memory_model = MemoryMLP(
            dim=self.memory_dim,
            depth=memory_depth,
            expansion_factor=4.0
        )
        
        # Initialize memory state (will be set during first forward pass)
        self.register_buffer('memory_state', None, persistent=False)
        
        # For momentum tracking
        if momentum:
            self.register_buffer('momentum_buffer', None, persistent=False)
    
    def forward(self, x: torch.Tensor, block_outputs: Optional[list] = None) -> torch.Tensor:
        """
        Forward pass with memory updates.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            block_outputs: Optional list of transformer block outputs (not used in simple memory)
            
        Returns:
            Updated tensor after memory processing
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply projection if needed
        if self.use_projection:
            x_mem = self.down_proj(x)
        else:
            x_mem = x
        
        # Process through memory model
        memory_output = self.memory_model(x_mem)
        
        # Update internal state
        if self.memory_state is None:
            # First forward pass: initialize memory state
            self.memory_state = memory_output.detach().mean(dim=1, keepdim=True)
        else:
            # Compute current state
            current_state = memory_output.detach().mean(dim=1, keepdim=True)
            
            # Apply momentum if enabled
            if self.momentum_enabled:
                if self.momentum_buffer is None:
                    self.momentum_buffer = current_state
                else:
                    self.momentum_buffer = self.momentum_factor * self.momentum_buffer + (1 - self.momentum_factor) * current_state
                
                # Update with momentum and forgetting
                self.memory_state = (1 - self.forget_factor) * self.memory_state + self.forget_factor * self.momentum_buffer
            else:
                # Simple update with forgetting
                self.memory_state = (1 - self.forget_factor) * self.memory_state + self.forget_factor * current_state
        
        # Project back if needed
        if self.use_projection:
            output = self.up_proj(memory_output)
        else:
            output = memory_output
        
        return output
    
    def get_memory_tokens(self, num_tokens: int = 4, block_outputs: Optional[list] = None) -> torch.Tensor:
        """
        For MAC integration - get memory tokens to be used as context.
        
        Args:
            num_tokens: Number of memory tokens to return
            block_outputs: Optional list of transformer block outputs (not used in simple memory)
            
        Returns:
            Memory tokens of shape [1, num_tokens, dim]
        """
        if self.memory_state is None:
            # No memory yet, return zeros
            return torch.zeros(1, num_tokens, self.dim, device=next(self.parameters()).device)
        
        # Project memory state to full dimension if needed
        if self.use_projection:
            mem_tokens = self.up_proj(self.memory_state)
        else:
            mem_tokens = self.memory_state
            
        # Expand to requested number of tokens
        return mem_tokens.expand(-1, num_tokens, -1)