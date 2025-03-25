#!/usr/bin/env python
"""
Main training script for ViT with Titans memory integration.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from datetime import datetime

# Import model components
from models.model_factory import create_vit_with_memory
from models.base_models import ViTTiny, ViTSmall, ViTBase

# Import data utilities
from data.data_loading import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_tiny_imagenet_loaders,
    get_ants_bees_loaders,
    get_imagenet_loaders
)

# Import training utilities
from utils.training import (
    train_epoch,
    validate,
    get_trainable_params,
    create_phase2_optimizer,
    save_checkpoint
)

# Import configuration
from configs.default_configs import (
    TRAINING_CONFIG,
    MODEL_CONFIGS,
    MEMORY_CONFIGS,
    INTEGRATION_CONFIGS,
    DATASET_CONFIGS
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ViT with Titans Memory')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'tiny-imagenet', 'ants-bees', 'imagenet'],
                        help='Dataset to use')
    
    # Model parameters
    parser.add_argument('--model-size', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'],
                        help='Size of the ViT model')
    parser.add_argument('--approach', type=str, default='mag',
                        choices=['mal', 'mac', 'mag'],
                        help='Memory integration approach')
    parser.add_argument('--memory-type', type=str, default='advanced',
                        choices=['none', 'mlp', 'simple', 'advanced'],
                        help='Type of memory module (none uses base model only)')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Disable use of pretrained weights')
    
    # Memory parameters
    parser.add_argument('--memory-depth', type=int, default=None,
                        help='Depth of memory module (overrides config)')
    
    # Advanced memory parameters
    parser.add_argument('--qkv-diff-views', action='store_true',
                        help='Use different views for QKV in advanced memory')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=TRAINING_CONFIG['batch_size'],
                        help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'],
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=TRAINING_CONFIG['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--unfreeze-epoch', type=int, 
                        default=TRAINING_CONFIG['unfreeze_epoch'],
                        help='Epoch to unfreeze backbone')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    
    # Integration approach parameters
    parser.add_argument('--memory-interval', type=int, default=None,
                        help='Memory interval for MAL (overrides config)')
    parser.add_argument('--persist-mem-tokens', type=int, default=None,
                        help='Number of persistent memory tokens (MAC/MAG) (overrides config)')
    parser.add_argument('--longterm-mem-tokens', type=int, default=None,
                        help='Number of long-term memory tokens (MAC only) (overrides config)')

    
    # Other parameters
    parser.add_argument('--seed', type=int, default=TRAINING_CONFIG['seed'],
                        help='Random seed for reproducibility')
    parser.add_argument('--save-dir', type=str, default=TRAINING_CONFIG['save_dir'],
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with verbose logging')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU usage')
    
    return parser.parse_args()

def main():
    """Main function to run training."""
    args = parse_args()
    
    # Set debug mode from args or environment variable
    debug = args.debug or os.environ.get('TITANS_DEBUG') == '1'
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create save directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.model_size}_{args.approach}_{args.memory_type}_{args.dataset}"
    save_dir = os.path.join(args.save_dir, f"{exp_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset_config = DATASET_CONFIGS[args.dataset]
    
    # Override image size if needed
    img_size = dataset_config['image_size']
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size, img_size=img_size)
    elif args.dataset == 'cifar100':
        train_loader, test_loader = get_cifar100_loaders(batch_size=args.batch_size, img_size=img_size)
    elif args.dataset == 'tiny-imagenet':
        train_loader, test_loader = get_tiny_imagenet_loaders(batch_size=args.batch_size, img_size=img_size)
    elif args.dataset == 'ants-bees':
        train_loader, test_loader = get_ants_bees_loaders(batch_size=args.batch_size, img_size=img_size)
    elif args.dataset == 'imagenet':
        train_loader, test_loader = get_imagenet_loaders(batch_size=args.batch_size, img_size=img_size)
    
    # Get configs
    memory_config = MEMORY_CONFIGS[args.memory_type]
    integration_config = INTEGRATION_CONFIGS[args.approach]
    
    # Override configs with command line arguments if provided
    if args.memory_depth is not None:
        memory_config['memory_depth'] = args.memory_depth
    if args.memory_interval is not None and args.approach == 'mal':
        integration_config['memory_interval'] = args.memory_interval
    if args.persist_mem_tokens is not None and args.approach in ['mac', 'mag']:
        integration_config['num_persist_mem_tokens'] = args.persist_mem_tokens
    if args.longterm_mem_tokens is not None and args.approach == 'mac':
        integration_config['num_longterm_mem_tokens'] = args.longterm_mem_tokens
    
    # Create model
    if args.memory_type == 'none':
        if args.model_size == 'tiny':
            model = ViTTiny(
                num_classes=dataset_config['num_classes'],
                img_size=dataset_config['image_size'],
                pretrained=not args.no_pretrained
            )
        elif args.model_size == 'small':
            model = ViTSmall(
                num_classes=dataset_config['num_classes'],
                img_size=dataset_config['image_size'],
                pretrained=not args.no_pretrained
            )
        elif args.model_size == 'base':
            model = ViTBase(
                num_classes=dataset_config['num_classes'],
                img_size=dataset_config['image_size'],
                pretrained=not args.no_pretrained
            )
    else:
        # Create memory model as before
        model = create_vit_with_memory(
            model_size=args.model_size,
            approach=args.approach,
            memory_type=args.memory_type,
            num_classes=dataset_config['num_classes'],
            image_size=dataset_config['image_size'],
            pretrained=not args.no_pretrained,
            qkv_receives_diff_views=args.qkv_diff_views,
            **{**memory_config, **integration_config}
        )
    
    model = model.to(device)
    
    print(f"Created {args.approach.upper()} model with {args.memory_type} memory "
          f"(size: {args.model_size}, pretrained: {not args.no_pretrained})")
    
    # Print model architecture in debug mode
    if debug:
        print(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set starting epoch
    start_epoch = 1
    best_acc = 0
    
    # Resume from checkpoint if provided
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set starting epoch and best accuracy
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('test_acc', 0)
            
            print(f"Resumed from epoch {start_epoch - 1} with best accuracy {best_acc:.2f}%")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Phase 1: Train only memory components
    trainable_params = get_trainable_params(model, args.approach, args.memory_type)
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=TRAINING_CONFIG['weight_decay'])
    
    # Choose scheduler
    if TRAINING_CONFIG['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=TRAINING_CONFIG['min_lr']
        )
    else:  # step scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=TRAINING_CONFIG['step_size'], gamma=TRAINING_CONFIG['gamma']
        )
    
    # Load optimizer state if resuming
    if args.resume and os.path.isfile(args.resume):
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        # Phase 2: Unfreeze backbone with smaller learning rate
        if epoch == args.unfreeze_epoch:
            print("Phase 2: Unfreezing backbone with smaller learning rate...")
            for param in model.parameters():
                param.requires_grad = True
                
            # Create new optimizer with different learning rates for different components
            optimizer = create_phase2_optimizer(
                model, args.approach, args.memory_type, args.lr,
                weight_decay=TRAINING_CONFIG['weight_decay']
            )
            
            # Create new scheduler
            if TRAINING_CONFIG['scheduler'] == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - args.unfreeze_epoch + 1,
                    eta_min=TRAINING_CONFIG['min_lr']
                )
            else:  # step scheduler
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=TRAINING_CONFIG['step_size'], gamma=TRAINING_CONFIG['gamma']
                )
        
        # Train for one epoch
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scheduler=None, use_amp=not args.no_amp, debug=debug
        )
        train_time = time.time() - start_time
        
        # Validate
        val_start_time = time.time()
        test_loss, test_acc = validate(
            model, test_loader, criterion, device, debug=debug
        )
        val_time = time.time() - val_start_time
        
        # Update learning rate
        scheduler.step()
        
        # Log results
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
              f"LR: {lr:.6f}, "
              f"Time: {train_time:.2f}s (train) + {val_time:.2f}s (val)")
        
        # Check if this is the best model
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
        
        # Save checkpoint
        should_save_epoch = epoch % TRAINING_CONFIG['save_interval'] == 0
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            save_path=save_dir,
            is_best=is_best,
            save_as_epoch=should_save_epoch,
            args=vars(args)
        )
    
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")
    print(f"Model checkpoints saved in {save_dir}")

if __name__ == "__main__":
    main()