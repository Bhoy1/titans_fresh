"""
Default configurations for model training and evaluation.
"""

# Training configurations
TRAINING_CONFIG = {
    # Basic training parameters
    'batch_size': 256,
    'num_workers': 8,
    'epochs': 50,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    
    # Two-phase training
    'unfreeze_epoch': 100,  # Epoch to unfreeze backbone
    
    # Model saving
    'save_dir': './checkpoints',
    'save_interval': 5,  # Save every N epochs
    
    # Scheduler parameters
    'scheduler': 'cosine',  # 'cosine' or 'step'
    'min_lr': 1e-6,  # Minimum learning rate for cosine scheduler
    'step_size': 10,  # Step size for step scheduler
    'gamma': 0.1,  # Gamma for step scheduler
    
    # Mixed precision training
    'use_amp': True,  # Use automatic mixed precision
    
    # Other parameters
    'seed': 42,  # Random seed
    'debug': False,  # Debug mode
}

# Model configurations for each size
MODEL_CONFIGS = {
    'tiny': {
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'mlp_ratio': 4.0,
    },
    'small': {
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
    },
    'base': {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
    }
}

# Memory module configurations
MEMORY_CONFIGS = {
    'none': {
        # Empty config for baseline model without memory
    },
    'mlp': {
        'memory_depth': 2,
        'expansion_factor': 4.0,
    },
    'simple': {
        'memory_depth': 2,
        'momentum': True,
        'momentum_factor': 0.9,
        'forget_factor': 0.1,
    },
    'advanced': {
        'memory_depth': 2,
        'chunk_size': 8,
        'momentum': True,
        'momentum_order': 1,
        'qk_rmsnorm': True,
        'qkv_receives_diff_views': True,
        'use_accelerated_scan': False,
    }
}

# Integration approach configurations
INTEGRATION_CONFIGS = {
    'mal': {
        'memory_interval': 1,  # Insert memory after every N blocks
    },
    'mac': {
        'num_persist_mem_tokens': 4,
        'num_longterm_mem_tokens': 4,
    },
    'mag': {
        'num_persist_mem_tokens': 4,
    }
}

# Dataset configurations
DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': 10,
        'image_size': 224,  # Resize for pretrained models
        'patch_size': 16,
        'in_channels': 3
    },
    'cifar100': {
        'num_classes': 100,
        'image_size': 224,  # Resize for pretrained models
        'patch_size': 16,
        'in_channels': 3
    },
    'tiny-imagenet': {
        'num_classes': 200,
        'image_size': 224,
        'patch_size': 16,
        'in_channels': 3
    },
    'ants-bees': {
        'num_classes': 2,
        'image_size': 224,
        'patch_size': 16,
        'in_channels': 3
    },
    'imagenet': {
        'num_classes': 1000,
        'image_size': 224,
        'patch_size': 16,
        'in_channels': 3
    }
}