#!/usr/bin/env python
"""
Simplified debug training script for ViT with Titans memory integration.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm
import argparse

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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Debug Train Script for ViT with Titans Memory')
    
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
                        help='Enable debug mode with verbose logging')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU usage')
    
    return parser.parse_args()

def debug_batch(model, loader, device, criterion, batch_idx=0, phase="train"):
    """Get detailed information about a single batch."""
    batch_iter = iter(loader)
    for _ in range(min(batch_idx, len(loader)-1)):
        next(batch_iter)
    
    inputs, targets = next(batch_iter)
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Count per class
    unique_targets, counts = torch.unique(targets, return_counts=True)
    target_counts = {loader.dataset.classes[t.item()]: c.item() for t, c in zip(unique_targets, counts)}
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Get predictions
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    accuracy = 100 * correct / targets.size(0)
    
    # Count predictions per class
    unique_preds, pred_counts = torch.unique(predicted, return_counts=True)
    pred_class_counts = {loader.dataset.classes[p.item()]: c.item() for p, c in zip(unique_preds, pred_counts)}
    
    # Get individual predictions
    individual = [(loader.dataset.classes[t.item()], 
                   loader.dataset.classes[p.item()], 
                   "✓" if t == p else "✗") 
                 for t, p in zip(targets[:min(8, len(targets))], 
                                predicted[:min(8, len(predicted))])]
    
    print(f"\n===== {phase.upper()} DEBUG INFO (Batch {batch_idx}) =====")
    print(f"Batch size: {targets.size(0)}")
    print(f"Target distribution: {target_counts}")
    print(f"Prediction distribution: {pred_class_counts}")
    print(f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
    print("\nSample predictions (True, Predicted, Correct):")
    for true_class, pred_class, correct_mark in individual:
        print(f"  {true_class:10s} → {pred_class:10s} {correct_mark}")
    print("="*50)
    
    return accuracy, loss.item()

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train model for one epoch with detailed logging."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Analyze initial batch
    print("\nAnalyzing first batch before training:")
    debug_batch(model, loader, device, criterion, 0, "initial")
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
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
        
        # Print batch statistics
        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}: Loss: {loss.item():.4f}, "
                  f"Batch Acc: {100 * batch_correct / targets.size(0):.2f}%, "
                  f"Running Acc: {100 * correct / total:.2f}%")
            
            # Debug a batch midway
            if batch_idx > 0 and batch_idx % 10 == 0:
                debug_batch(model, loader, device, criterion, batch_idx, "mid_train")
    
    return running_loss/len(loader), 100.*correct/total

def validate(model, loader, criterion, device):
    """Validate model on test data."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc="Validating")):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Debug the first validation batch
            if batch_idx == 0:
                debug_batch(model, loader, device, criterion, batch_idx, "validation")
    
    # Calculate class-wise accuracy
    class_correct = list(0. for _ in range(len(loader.dataset.classes)))
    class_total = list(0. for _ in range(len(loader.dataset.classes)))
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == targets).squeeze()
            
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Print class-wise accuracy
    print("\nClass-wise accuracy:")
    for i in range(len(loader.dataset.classes)):
        print(f'Accuracy of {loader.dataset.classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
    
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
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_loader, test_loader = get_cifar100_loaders(batch_size=args.batch_size)
        num_classes = 100
    elif args.dataset == 'tiny-imagenet':
        train_loader, test_loader = get_tiny_imagenet_loaders(batch_size=args.batch_size)
        num_classes = 200
    elif args.dataset == 'ants-bees':
        train_loader, test_loader = get_ants_bees_loaders(batch_size=args.batch_size, num_workers=4)
        num_classes = 2
    elif args.dataset == 'imagenet':
        train_loader, test_loader = get_imagenet_loaders(batch_size=args.batch_size)
        num_classes = 1000
    
    # Print dataset statistics
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {train_loader.dataset.classes}")
    
    # Create model
    print(f"Creating model: {args.model_size}-{args.approach}-{args.memory_type}")
    if args.memory_type == 'none':
        # Create base model directly for memory_type=none
        if args.model_size == 'tiny':
            model = ViTTiny(num_classes=num_classes, pretrained=not args.no_pretrained)
        elif args.model_size == 'small':
            model = ViTSmall(num_classes=num_classes, pretrained=not args.no_pretrained)
        elif args.model_size == 'base':
            model = ViTBase(num_classes=num_classes, pretrained=not args.no_pretrained)
    else:
        # Use factory function for models with memory
        model = create_vit_with_memory(
            model_size=args.model_size,
            approach=args.approach,
            memory_type=args.memory_type,
            num_classes=num_classes,
            pretrained=not args.no_pretrained
        )
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training loop
    best_acc = 0
    start_time = time.time()
    
    # Check initial model predictions
    print("\nChecking initial model predictions...")
    with torch.no_grad():
        initial_val_loss, initial_val_acc = validate(model, test_loader, criterion, device)
        print(f"Initial validation - Loss: {initial_val_loss:.4f}, Accuracy: {initial_val_acc:.2f}%")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        epoch_train_time = time.time() - epoch_start
        
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