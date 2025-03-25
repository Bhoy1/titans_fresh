"""
Base Vision Transformer models loaded from Timm's pretrained models.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Dict, Any, Tuple

class PretrainedViT(nn.Module):
    """
    Wrapper for Timm's pretrained Vision Transformer models.
    This wrapper makes it easier to integrate with our memory modules.
    """
    def __init__(
        self,
        model_name: str,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Load pretrained model
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained,
            img_size=img_size,
            num_classes=num_classes,
            **kwargs
        )
        
        # Extract key components for easier access
        self.patch_embed = self.model.patch_embed
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.pos_drop = self.model.pos_drop
        self.blocks = self.model.blocks
        self.norm = self.model.norm
        self.head = self.model.head
        
        # Get embedding dimension from the model
        self.embed_dim = self.model.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output logits tensor of shape [batch_size, num_classes]
        """
        return self.model(x)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor (everything except the final classifier head).
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Features tensor
        """
        return self.model.forward_features(x)
    
    def get_block_outputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Forward pass that returns outputs from each transformer block.
        Useful for differentiated views in memory modules.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Tuple of (final output, list of block outputs)
        """
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Collect outputs from each block
        block_outputs = []
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)
        
        x = self.norm(x)
        return x, block_outputs

    @property
    def num_features(self) -> int:
        """Get the feature dimension of the model."""
        return self.embed_dim

def ViTTiny(num_classes: int = 1000, img_size: int = 224, pretrained: bool = True, **kwargs) -> PretrainedViT:
    """
    Create a pretrained Tiny ViT model.
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Pretrained ViT Tiny model
    """
    return PretrainedViT(
        model_name='vit_tiny_patch16_224',
        img_size=img_size,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )

def ViTSmall(num_classes: int = 1000, img_size: int = 224, pretrained: bool = True, **kwargs) -> PretrainedViT:
    """
    Create a pretrained Small ViT model.
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Pretrained ViT Small model
    """
    return PretrainedViT(
        model_name='vit_small_patch16_224',
        img_size=img_size,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )

def ViTBase(num_classes: int = 1000, img_size: int = 224, pretrained: bool = True, **kwargs) -> PretrainedViT:
    """
    Create a pretrained Base ViT model.
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Pretrained ViT Base model
    """
    return PretrainedViT(
        model_name='vit_base_patch16_224',
        img_size=img_size,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )