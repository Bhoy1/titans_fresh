"""
Memory as Gate (MAG) integration approach for Vision Transformers.
Uses memory output as an alternative path, combined with main
path using a learned gating mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List, Union, Any

class ViTWithMAG(nn.Module):
    """
    Vision Transformer with Memory as Gate (MAG) integration.
    Uses memory output as an alternative path, combined with main
    path using a learned gating mechanism.
    """
    def __init__(
        self,
        base_model: nn.Module,
        memory_module: nn.Module,
        num_persist_mem_tokens: int = 4,
        enable_diff_views: bool = False
    ):
        """
        Initialize ViT with Memory as Gate integration.
        
        Args:
            base_model: Base Vision Transformer model
            memory_module: Memory module to use
            num_persist_mem_tokens: Number of persistent memory tokens
            enable_diff_views: Whether to enable differentiated views for QKV
        """
        super().__init__()
        self.embed_dim = base_model.embed_dim
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.enable_diff_views = enable_diff_views
        
        # Store base model components
        self.base_model = base_model
        
        # Create persistent memory tokens (learnable parameters)
        self.persistent_memory = nn.Parameter(
            torch.randn(num_persist_mem_tokens, self.embed_dim) * 0.02
        )
        
        # Memory module
        self.memory_module = memory_module
        
        # Initialize block weights if using differentiated views
        if enable_diff_views and hasattr(memory_module, 'qkv_receives_diff_views') and memory_module.qkv_receives_diff_views:
            if hasattr(memory_module, 'set_transformer_blocks'):
                num_blocks = len(self.base_model.blocks)
                memory_module.set_transformer_blocks(num_blocks)
        
        # Gating mechanism
        self.gate_norm1 = nn.LayerNorm(self.embed_dim)
        self.gate_norm2 = nn.LayerNorm(self.embed_dim)
        self.to_gate = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Sigmoid()
        )
        
    def interpolate_pos_encoding(self, pos_embed: torch.Tensor, x_size: int) -> torch.Tensor:
        """
        Interpolate position embeddings for sequences of different length.
        
        Args:
            pos_embed: Original position embeddings
            x_size: New sequence length (excluding class token)
            
        Returns:
            Interpolated position embeddings
        """
        # Handle class token separately
        pos_embed_cls = pos_embed[:, 0:1, :]
        pos_embed_patch = pos_embed[:, 1:, :]
        
        # Get dimensions
        dim = pos_embed.shape[-1]
        orig_patches = pos_embed_patch.shape[1]
        
        # Interpolate position embeddings to match new sequence length
        if x_size != orig_patches:
            # Calculate grid sizes
            orig_size = int(math.sqrt(orig_patches))
            new_size = int(math.sqrt(x_size))
            
            # Reshape for interpolation
            pos_embed_patch = pos_embed_patch.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
            
            # Interpolate
            pos_embed_patch = F.interpolate(
                pos_embed_patch, 
                size=(new_size, new_size), 
                mode='bicubic', 
                align_corners=False
            )
            
            # Reshape back
            pos_embed_patch = pos_embed_patch.permute(0, 2, 3, 1).reshape(1, new_size * new_size, dim)
        
        # Combine class token position embeddings with patch position embeddings
        return torch.cat([pos_embed_cls, pos_embed_patch], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output logits tensor of shape [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        
        # Process input through patch embedding
        x = self.base_model.patch_embed(x)
        
        # Add class token
        class_token = self.base_model.cls_token.expand(batch_size, -1, -1)
        
        # Add persistent memory tokens
        persist_mem = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate [class token, persistent memory, patch tokens]
        x_with_mem = torch.cat([class_token, persist_mem, x], dim=1)
        
        # Calculate new position embeddings
        extra_tokens = 1 + self.num_persist_mem_tokens
        
        # Interpolate position embeddings for patches
        num_patches = x.shape[1]
        pos_embed = self.interpolate_pos_encoding(self.base_model.pos_embed, num_patches)
        
        # Create position embeddings for memory tokens (zeros or learned)
        pos_embed_cls = pos_embed[:, 0:1, :]
        pos_embed_mem = torch.zeros(1, extra_tokens - 1, self.embed_dim, device=x.device)
        
        # Combine position embeddings
        pos_embed = torch.cat([pos_embed_cls, pos_embed_mem, pos_embed[:, 1:, :]], dim=1)
        
        # Add position embeddings
        x_with_mem = x_with_mem + pos_embed
        
        # Apply dropout
        x_with_mem = self.base_model.pos_drop(x_with_mem)
        
        # Process through transformer blocks
        if self.enable_diff_views and hasattr(self.memory_module, 'qkv_receives_diff_views') and self.memory_module.qkv_receives_diff_views:
            # Store outputs from each block for differentiated views
            block_outputs = []
            current = x_with_mem
            
            for block in self.base_model.blocks:
                current = block(current)
                block_outputs.append(current)
            
            # Get class token from final output
            x_out = current[:, 0]
        else:
            # Standard processing without storing intermediate outputs
            for block in self.base_model.blocks:
                x_with_mem = block(x_with_mem)
            
            # Get class token
            x_out = x_with_mem[:, 0]
            block_outputs = None
        
        # Process through memory path
        if self.enable_diff_views and hasattr(self.memory_module, 'qkv_receives_diff_views') and self.memory_module.qkv_receives_diff_views and block_outputs is not None:
            if hasattr(self.memory_module, 'forward') and hasattr(self.memory_module.forward, '__code__') and 'state' in self.memory_module.forward.__code__.co_varnames:
                # For stateful memory modules with differentiated views
                memory_out, _ = self.memory_module(x.mean(dim=1, keepdim=True), None, block_outputs)
                memory_out = memory_out.squeeze(1)
            else:
                # For simpler memory modules without state
                memory_out = self.memory_module(x.mean(dim=1, keepdim=True), block_outputs=block_outputs).squeeze(1)
        else:
            # Standard memory processing
            if hasattr(self.memory_module, 'forward') and hasattr(self.memory_module.forward, '__code__') and 'state' in self.memory_module.forward.__code__.co_varnames:
                # For stateful memory modules
                memory_out, _ = self.memory_module(x.mean(dim=1, keepdim=True))
                memory_out = memory_out.squeeze(1)
            else:
                # For simpler memory modules
                memory_out = self.memory_module(x.mean(dim=1, keepdim=True)).squeeze(1)
        
        # Apply gating mechanism
        x_norm = self.gate_norm1(x_out)
        memory_norm = self.gate_norm2(memory_out)
        gate = self.to_gate(x_norm)
        
        # Combine outputs with gating
        combined = gate * x_norm + (1 - gate) * memory_norm
        
        # Apply final classification
        output = self.base_model.norm(combined)
        output = self.base_model.head(output)
        
        return output