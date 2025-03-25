"""
Factory module for creating Vision Transformers with memory integration.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

# Import base models
from .base_models import ViTTiny, ViTSmall, ViTBase

# Import memory modules
from .memory_modules.memory_mlp import MemoryMLP
from .memory_modules.simple_memory import SimpleNeuralMemory
from .memory_modules.advanced_memory import AdvancedNeuralMemory

# Import integration approaches
from .integration.mal import ViTWithMAL
from .integration.mac import ViTWithMAC
from .integration.mag import ViTWithMAG

def create_memory_module(
    memory_type: str,
    embed_dim: int,
    memory_dim: Optional[int] = None,
    memory_depth: int = 2,
    integration_type: str = "mal",
    qkv_receives_diff_views: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create a memory module based on type.
    
    Args:
        memory_type: "mlp", "simple", or "advanced"
        embed_dim: Dimension of the embedding
        memory_dim: Inner dimension for neural memory (default: embed_dim)
        memory_depth: Depth of the memory module
        integration_type: Integration approach ("mal", "mac", "mag")
        qkv_receives_diff_views: Whether QKV should use different views (advanced memory only)
        **kwargs: Additional arguments for specific memory types
        
    Returns:
        Memory module instance
    """
    if memory_dim is None:
        memory_dim = embed_dim
        
    if memory_type == "mlp":
        return MemoryMLP(
            dim=embed_dim,
            depth=memory_depth,
            expansion_factor=kwargs.get("expansion_factor", 4.0)
        )
    elif memory_type == "simple":
        return SimpleNeuralMemory(
            dim=embed_dim,
            memory_depth=memory_depth,
            memory_dim=memory_dim,
            momentum=kwargs.get("momentum", True),
            momentum_factor=kwargs.get("momentum_factor", 0.9),
            forget_factor=kwargs.get("forget_factor", 0.1),
            integration_type=integration_type
        )
    elif memory_type == "advanced":
        # For advanced memory, we need to create the inner model first
        inner_memory = MemoryMLP(
            dim=memory_dim,
            depth=memory_depth
        )
        
        return AdvancedNeuralMemory(
            dim=embed_dim,
            chunk_size=kwargs.get("chunk_size", 8),
            neural_memory_model=inner_memory,
            memory_depth=memory_depth,
            memory_dim=memory_dim,
            momentum=kwargs.get("momentum", True),
            momentum_order=kwargs.get("momentum_order", 1),
            qk_rmsnorm=kwargs.get("qk_rmsnorm", True),
            qkv_receives_diff_views=qkv_receives_diff_views,
            integration_type=integration_type,
            use_accelerated_scan=kwargs.get("use_accelerated_scan", False)
        )
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")

def create_vit_with_memory(
    model_size: str,
    approach: str,
    memory_type: str,
    num_classes: int = 1000,
    image_size: int = 224,
    pretrained: bool = True,
    qkv_receives_diff_views: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create a ViT model with memory integration.
    
    Args:
        model_size: "tiny", "small", or "base"
        approach: "mal", "mac", or "mag"
        memory_type: "mlp", "simple", or "advanced"
        num_classes: Number of output classes
        image_size: Input image size
        pretrained: Whether to use pretrained weights
        qkv_receives_diff_views: Allow different views for QKV (advanced memory only)
        memory_depth: Depth of memory module
        memory_interval: Add memory after every N blocks (MAL only)
        num_persist_mem_tokens: Number of persistent memory tokens (MAC/MAG)
        num_longterm_mem_tokens: Number of long-term memory tokens (MAC only)
        **kwargs: Additional arguments for specific models
        
    Returns:
        Vision Transformer with memory integration
    """
    # Create base model (filter out memory-specific parameters)
    base_kwargs = {k: v for k, v in kwargs.items() 
                  if k not in ['memory_depth', 'expansion_factor', 'momentum', 'momentum_factor',
                             'forget_factor', 'chunk_size', 'momentum_order', 'qk_rmsnorm',
                             'use_accelerated_scan', 'memory_interval', 'num_persist_mem_tokens', 
                             'num_longterm_mem_tokens', 'qkv_receives_diff_views']}
    
    # Create base model
    if model_size == "tiny":
        base_model = ViTTiny(
            num_classes=num_classes,
            img_size=image_size,
            pretrained=pretrained,
            **base_kwargs
        )
    elif model_size == "small":
        base_model = ViTSmall(
            num_classes=num_classes,
            img_size=image_size,
            pretrained=pretrained,
            **base_kwargs
        )
    elif model_size == "base":
        base_model = ViTBase(
            num_classes=num_classes,
            img_size=image_size,
            pretrained=pretrained,
            **base_kwargs
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # If memory_type is "none", return the base model directly
    if memory_type == "none":
        return base_model
    
    # Get embedding dimension from the base model
    embed_dim = base_model.embed_dim
    
    # Create memory module (pass only memory-relevant parameters)
    memory_kwargs = {
        k: v for k, v in kwargs.items() 
        if k in ['memory_depth', 'expansion_factor', 'momentum', 'momentum_factor',
                'forget_factor', 'chunk_size', 'momentum_order', 'qk_rmsnorm',
                'use_accelerated_scan']
    }
    
    # Add differentiated views flag for advanced memory
    if memory_type == "advanced":
        pass
    
    memory_module = create_memory_module(
        memory_type=memory_type,
        embed_dim=embed_dim,  # Now using the embed_dim from the base model
        integration_type=approach,
        **memory_kwargs
    )

    # Create integrated model (pass only integration-relevant parameters)
    integration_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ['memory_interval', 'num_persist_mem_tokens', 'num_longterm_mem_tokens']
    }
    
    # Add differentiated views flag for all integration methods when using advanced memory
    enable_diff_views = qkv_receives_diff_views and memory_type == "advanced"
    
    # Create integrated model based on approach
    if approach == "mal":
        return ViTWithMAL(
            base_model=base_model,
            memory_module=memory_module,
            enable_diff_views=enable_diff_views,
            memory_interval=integration_kwargs.get('memory_interval', 1)
        )
    elif approach == "mac":
        return ViTWithMAC(
            base_model=base_model,
            memory_module=memory_module,
            enable_diff_views=enable_diff_views,
            num_persist_mem_tokens=integration_kwargs.get('num_persist_mem_tokens', 4),
            num_longterm_mem_tokens=integration_kwargs.get('num_longterm_mem_tokens', 4)
        )
    elif approach == "mag":
        return ViTWithMAG(
            base_model=base_model,
            memory_module=memory_module,
            enable_diff_views=enable_diff_views,
            num_persist_mem_tokens=integration_kwargs.get('num_persist_mem_tokens', 4)
        )
    else:
        raise ValueError(f"Unknown approach: {approach}")