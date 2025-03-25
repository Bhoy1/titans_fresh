"""
Memory as Layer (MAL) integration approach for Vision Transformers.
Inserts memory modules between transformer blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union, Any

class ViTWithMAL(nn.Module):
    """
    Vision Transformer with Memory as Layer (MAL) integration.
    Inserts memory modules between transformer blocks.
    """
    def __init__(
        self,
        base_model: nn.Module,
        memory_module: nn.Module,
        memory_interval: int = 1,  # Insert memory after every N blocks
        enable_diff_views: bool = False
    ):
        """
        Initialize ViT with Memory as Layer integration.
        
        Args:
            base_model: Base Vision Transformer model
            memory_module: Memory module to insert
            memory_interval: Insert memory after every N blocks
            enable_diff_views: Whether to enable differentiated views for QKV
        """
        super().__init__()
        self.embed_dim = base_model.embed_dim
        self.memory_interval = memory_interval
        self.enable_diff_views = enable_diff_views
        
        # Extract components from base model
        self.patch_embed = base_model.patch_embed
        self.cls_token = base_model.cls_token
        self.pos_embed = base_model.pos_embed
        self.pos_drop = base_model.pos_drop
        self.norm = base_model.norm
        self.head = base_model.head
        
        # Create new blocks sequence with interleaved memory modules
        original_blocks = base_model.blocks
        self.blocks = nn.ModuleList()
        
        for i, block in enumerate(original_blocks):
            self.blocks.append(block)
            
            # Add memory after specified interval
            if (i + 1) % memory_interval == 0 and i < len(original_blocks) - 1:
                self.blocks.append(memory_module)
        
        # Initialize block weights for differentiated views
        if enable_diff_views and hasattr(memory_module, 'qkv_receives_diff_views') and memory_module.qkv_receives_diff_views:
            if hasattr(memory_module, 'set_transformer_blocks'):
                num_blocks = len(original_blocks)
                memory_module.set_transformer_blocks(num_blocks)
        
        # Register memory states buffer
        self.memory_states = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output logits tensor of shape [batch_size, num_classes]
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (B, 1 + n_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # For differentiated views, we need to track all transformer block outputs
        if self.enable_diff_views:
            block_outputs = []
            current = x
            memory_idx = 0
            
            for i, block in enumerate(self.blocks):
                if isinstance(block, type(self.blocks[0])):  # This is a transformer block
                    current = block(current)
                    block_outputs.append(current)
                else:  # This is a memory module
                    # For memory modules with differentiated views
                    if hasattr(block, 'qkv_receives_diff_views') and block.qkv_receives_diff_views:
                        # Get state for this memory module
                        state = self.memory_states.get(memory_idx, None)
                        
                        # Process with differentiated views
                        if hasattr(block, 'forward') and 'state' in block.forward.__code__.co_varnames:
                            # Advanced memory with state
                            memory_out, next_state = block(current, state, block_outputs)
                            self.memory_states[memory_idx] = next_state
                        else:
                            # Simple memory without state
                            memory_out = block(current, block_outputs=block_outputs)
                    else:
                        # Standard memory processing
                        if hasattr(block, 'forward') and 'state' in block.forward.__code__.co_varnames:
                            # Advanced memory with state
                            state = self.memory_states.get(memory_idx, None)
                            memory_out, next_state = block(current, state)
                            self.memory_states[memory_idx] = next_state
                        else:
                            # Simple memory without state
                            memory_out = block(current)
                    
                    current = memory_out
                    memory_idx += 1
        else:
            # Standard processing without tracking block outputs
            current = x
            memory_idx = 0
            
            for i, block in enumerate(self.blocks):
                if isinstance(block, type(self.blocks[0])):  # This is a transformer block
                    current = block(current)
                else:  # This is a memory module
                    # Process memory module
                    if hasattr(block, 'forward') and hasattr(block.forward, '__code__') and 'state' in block.forward.__code__.co_varnames:
                        # Advanced memory with state
                        state = self.memory_states.get(memory_idx, None)
                        memory_out, next_state = block(current, state)
                        self.memory_states[memory_idx] = next_state
                    else:
                        # Simple memory without state
                        memory_out = block(current)
                    
                    current = memory_out
                    memory_idx += 1
        
        # Apply final normalization
        current = self.norm(current)
        
        # Return class token for classification
        return self.head(current[:, 0])