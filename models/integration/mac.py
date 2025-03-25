"""
Memory as Context (MAC) integration approach for Vision Transformers.
Adds memory tokens as context to the input sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List, Union, Any

class ViTWithMAC(nn.Module):
    """
    Vision Transformer with Memory as Context (MAC) integration.
    Adds memory tokens as context to the input sequence.
    """
    def __init__(
        self,
        base_model: nn.Module,
        memory_module: nn.Module,
        num_persist_mem_tokens: int = 4,
        num_longterm_mem_tokens: int = 4,
        enable_diff_views: bool = False
    ):
        """
        Initialize ViT with Memory as Context integration.
        
        Args:
            base_model: Base Vision Transformer model
            memory_module: Memory module to use
            num_persist_mem_tokens: Number of persistent memory tokens
            num_longterm_mem_tokens: Number of long-term memory tokens
            enable_diff_views: Whether to enable differentiated views for QKV
        """
        super().__init__()
        self.embed_dim = base_model.embed_dim
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.enable_diff_views = enable_diff_views
        
        # Create base model components
        self.base_model = base_model
        
        # Create persistent memory tokens (learnable parameters)
        self.persistent_memory = nn.Parameter(
            torch.randn(num_persist_mem_tokens, self.embed_dim) * 0.02
        )
        
        # Query projection for memory retrieval
        self.to_memory_query = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Memory module
        self.memory_module = memory_module
        
        # Initialize block weights for differentiated views
        if enable_diff_views and hasattr(memory_module, 'qkv_receives_diff_views') and memory_module.qkv_receives_diff_views:
            if hasattr(memory_module, 'set_transformer_blocks'):
                num_blocks = len(self.base_model.blocks)
                memory_module.set_transformer_blocks(num_blocks)
        
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
        
        # Generate query for memory retrieval
        query = self.to_memory_query(x.mean(dim=1, keepdim=True))
        
        # For differentiated views, collect block outputs first
        if self.enable_diff_views and hasattr(self.memory_module, 'qkv_receives_diff_views') and self.memory_module.qkv_receives_diff_views:
            # Add class token for initial pass
            init_cls_token = self.base_model.cls_token.expand(batch_size, -1, -1)
            init_x = torch.cat([init_cls_token, x], dim=1)
            
            # Add position embeddings
            init_pos_embed = self.interpolate_pos_encoding(self.base_model.pos_embed, x.size(1))
            init_x = init_x + init_pos_embed
            init_x = self.base_model.pos_drop(init_x)
            
            # Process through blocks to collect outputs
            block_outputs = []
            current = init_x
            
            for block in self.base_model.blocks:
                current = block(current)
                block_outputs.append(current)
            
            # Get memory output with differentiated views
            if hasattr(self.memory_module, 'get_memory_tokens'):
                if hasattr(self.memory_module.get_memory_tokens, '__code__') and 'block_outputs' in self.memory_module.get_memory_tokens.__code__.co_varnames:
                    memory_output = self.memory_module.get_memory_tokens(
                        num_tokens=self.num_longterm_mem_tokens,
                        block_outputs=block_outputs
                    ).expand(batch_size, -1, -1)
                else:
                    # Fallback if memory module doesn't support block_outputs
                    memory_output = self.memory_module.get_memory_tokens(
                        num_tokens=self.num_longterm_mem_tokens
                    ).expand(batch_size, -1, -1)
            else:
                # For simple memory module, just process the query and expand
                memory_output = self.memory_module(query).expand(
                    batch_size, self.num_longterm_mem_tokens, -1
                )
        else:
            # Standard memory retrieval without differentiated views
            if hasattr(self.memory_module, 'get_memory_tokens'):
                memory_output = self.memory_module.get_memory_tokens(
                    num_tokens=self.num_longterm_mem_tokens
                ).expand(batch_size, -1, -1)
            else:
                # For simple memory module, just process the query and expand
                memory_output = self.memory_module(query).expand(
                    batch_size, self.num_longterm_mem_tokens, -1
                )
        
        # Expand persistent memory tokens
        persist_mem = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add class token
        class_token = self.base_model.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate [class token, persistent memory, long-term memory, patch tokens]
        x = torch.cat([class_token, persist_mem, memory_output, x], dim=1)
        
        # Add position embeddings - handling the expanded sequence
        extra_tokens = 1 + self.num_persist_mem_tokens + self.num_longterm_mem_tokens
        
        # Interpolate position embeddings for the patches
        num_patches = x.shape[1] - extra_tokens
        pos_embed_patch = self.interpolate_pos_encoding(self.base_model.pos_embed, num_patches)
        
        # Create position embeddings for memory tokens (zeros or learned)
        pos_embed_cls = pos_embed_patch[:, 0:1, :]
        pos_embed_mem = torch.zeros(1, extra_tokens - 1, self.embed_dim, device=x.device)
        
        # Combine position embeddings
        pos_embed = torch.cat([pos_embed_cls, pos_embed_mem, pos_embed_patch[:, 1:, :]], dim=1)
        
        # Add position embeddings
        x = x + pos_embed
        
        # Apply dropout
        x = self.base_model.pos_drop(x)
        
        # Process through transformer blocks
        for block in self.base_model.blocks:
            x = block(x)
        
        # Apply norm and get class token for classification
        x = self.base_model.norm(x[:, 0])
        x = self.base_model.head(x)
        
        # Update memory if it has forward method
        if hasattr(self.memory_module, 'forward'):
            if hasattr(self.memory_module.forward, '__code__') and 'state' in self.memory_module.forward.__code__.co_varnames:
                # For stateful memory modules (like AdvancedNeuralMemory)
                # Use a detached version of the output to update memory without affecting gradients
                with torch.no_grad():
                    if self.enable_diff_views and hasattr(self.memory_module, 'qkv_receives_diff_views') and self.memory_module.qkv_receives_diff_views and hasattr(self.memory_module.forward, '__code__') and 'block_outputs' in self.memory_module.forward.__code__.co_varnames:
                        _, _ = self.memory_module(
                            x.detach().mean(dim=0, keepdim=True).unsqueeze(1),
                            None, block_outputs
                        )
                    else:
                        _, _ = self.memory_module(
                            x.detach().mean(dim=0, keepdim=True).unsqueeze(1)
                        )
            else:
                # For simpler memory modules
                _ = self.memory_module(
                    x.detach().mean(dim=0, keepdim=True).unsqueeze(1)
                )
        
        return x