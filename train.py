#!/usr/bin/env python
"""
Fast debug training script for ViT with Titans memory integration.
Based on the faster quick_test.py approach.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
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

# Import configurations
from configs.default_configs import DATASET_CONFIGS, MEMORY_CONFIGS, INTEGRATION_CONFIGS

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fast Debug Train Script for ViT with Titans Memory')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='ants-bees',
                        choices=['cifar10', 'cifar100', 'tiny-imagenet', 'ants-bees', 'imagenet'],
                        help='Dataset to use')
    
    # Model parameters
    parser.add_argument('--model-size', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'],
                        help='Size of the ViT model')
    parser.add_argument('--approach', type=str, default='mal',
                        choices=['mal', 'mac', 'mag'],
                        help='Memory integration approach')
    parser.add_argument('--memory-type', type=str, default='none',
                        choices=['none', 'mlp', 'simple', 'advanced'],
                        help='Type of memory module (none uses base model only)')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Disable use of pretrained weights')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--debug', action='store_true',
                        help='Enable more verbose debug logging')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU usage')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    
    return parser.parse_args()

def debug_batch(model, loader, device, criterion, phase="debug"):
    """Debug a batch from the data loader."""
    batch_data = next(iter(loader))
    inputs, targets = batch_data
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
    
    # Calculate metrics
    accuracy = 100 * predicted.eq(targets).sum().item() / targets.size(0)
    
    # Count class distribution in targets and predictions
    target_counts = {}
    pred_counts = {}
    for c in loader.dataset.classes:
        target_counts[c] = 0
        pred_counts[c] = 0
    
    for t in targets:
        target_counts[loader.dataset.classes[t.item()]] += 1
    
    for p in predicted:
        pred_counts[loader.dataset.classes[p.item()]] += 1
    
    # Print debug info
    print(f"\n===== {phase.upper()} DEBUG INFO =====")
    print(f"Batch size: {targets.size(0)}")
    print(f"Target distribution: {target_counts}")
    print(f"Prediction distribution: {pred_counts}")
    print(f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
    
    # Show individual predictions (up to 8 samples)
    #print("Sample predictions (True -> Predicted, Correct?):")
    #for i in range(min(8, targets.size(0))):
    #    t = loader.dataset.classes[targets[i].item()]
    #    p = loader.dataset.classes[predicted[i].item()]
    #    correct = "✓" if predicted[i] == targets[i] else "✗"
    #    print(f"  {t:8s} -> {p:8s} {correct}")
    
    #print("=" * 50)
    
    # Return to train mode if needed
    model.train()
    return accuracy

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Log only at specific intervals
    log_interval = max(1, len(loader) // 4)
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        batch_correct = predicted.eq(targets).sum().item()
        correct += batch_correct
        total += targets.size(0)
        
        # Print batch statistics (limited to reduce overhead)
        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: Loss: {loss.item():.4f}, "
                  f"Batch Acc: {100 * batch_correct / targets.size(0):.2f}%, "
                  f"Running Acc: {100 * correct / total:.2f}%")
    
    return running_loss/len(loader), 100.*correct/total

def validate(model, loader, criterion, device):
    """Validate model on test data."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    # Track class-wise metrics
    num_classes = len(loader.dataset.classes)
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Track class-wise accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += (predicted[i] == targets[i]).item()
                class_total[label] += 1
    
    # Print class-wise accuracy
    print("\nClass-wise accuracy:")
    for i in range(num_classes):
        print(f"  {loader.dataset.classes[i]}: {100 * class_correct[i] / max(1, class_total[i]):.2f}% "
              f"({class_correct[i]}/{class_total[i]})")
    
    return val_loss/len(loader), 100.*correct/total

def main():
    """Main function to run training."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Get dataset configs
    dataset_config = DATASET_CONFIGS[args.dataset]
    img_size = dataset_config['image_size']
    
    # Get memory and integration configs
    memory_config = MEMORY_CONFIGS[args.memory_type]
    integration_config = INTEGRATION_CONFIGS[args.approach]
    
    # Load dataset with adjusted num_workers
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size, num_workers=args.num_workers, img_size=img_size)
    elif args.dataset == 'cifar100':
        train_loader, test_loader = get_cifar100_loaders(batch_size=args.batch_size, num_workers=args.num_workers, img_size=img_size)
    elif args.dataset == 'tiny-imagenet':
        train_loader, test_loader = get_tiny_imagenet_loaders(batch_size=args.batch_size, num_workers=args.num_workers, img_size=img_size)
    elif args.dataset == 'ants-bees':
        train_loader, test_loader = get_ants_bees_loaders(batch_size=args.batch_size, num_workers=args.num_workers, img_size=img_size)
    elif args.dataset == 'imagenet':
        train_loader, test_loader = get_imagenet_loaders(batch_size=args.batch_size, num_workers=args.num_workers, img_size=img_size)
    
    # Print dataset statistics
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Number of classes: {len(train_loader.dataset.classes)}")
    print(f"Class names: {train_loader.dataset.classes}")
    
    # Check class distribution
    train_targets = [sample[1] for sample in train_loader.dataset]
    class_counts = {}
    for i, cls in enumerate(train_loader.dataset.classes):
        class_counts[cls] = train_targets.count(i)
    print(f"Class distribution in training set: {class_counts}")
    
    # Create model using factory function - EXACTLY as in test_mal_none.py
    print(f"Creating model: {args.model_size}-{args.approach}-{args.memory_type}")
    model = create_vit_with_memory(
        model_size=args.model_size,
        approach=args.approach,
        memory_type=args.memory_type,
        num_classes=dataset_config['num_classes'],
        image_size=img_size,
        pretrained=not args.no_pretrained,
        **{**memory_config, **integration_config}
    )
    
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_acc = 0
    start_time = time.time()
    
    # Check initial model predictions
    print("\nChecking initial model predictions...")
    debug_batch(model, train_loader, device, criterion, "initial_train")
    
    with torch.no_grad():
        initial_val_loss, initial_val_acc = validate(model, test_loader, criterion, device)
        print(f"Initial validation - Loss: {initial_val_loss:.4f}, Accuracy: {initial_val_acc:.2f}%")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        epoch_train_time = time.time() - epoch_start
        
        # Debug a batch mid-training
        debug_batch(model, train_loader, device, criterion, f"epoch_{epoch+1}")
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        epoch_total_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Time: {epoch_train_time:.2f}s (train) + {epoch_total_time-epoch_train_time:.2f}s (val) = {epoch_total_time:.2f}s total")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(args.save_dir, f"{args.dataset}_{args.model_size}_{args.approach}_{args.memory_type}_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved new best model with accuracy: {val_acc:.2f}%")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed! Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()