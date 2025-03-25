"""
Training utilities for ViT with Titans Memory models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List, Union, Callable, Any

def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    use_amp: bool = False,
    debug: bool = False
) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Data loader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training
        epoch: Current epoch number
        scheduler: Optional learning rate scheduler
        use_amp: Whether to use automatic mixed precision
        debug: Whether to print debugging information
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    
    # Get scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward and backward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Update scheduler if batch-wise
        if scheduler is not None and hasattr(scheduler, 'step_batch'):
            scheduler.step_batch()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss/(batch_idx+1), 
            'acc': 100.*correct/total,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        if debug and batch_idx % 10 == 0:
            print(f"Debug - Batch {batch_idx}, Loss: {loss.item():.4f}, "
                  f"Acc: {100.*correct/total:.2f}%")
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    debug: bool = False
) -> Tuple[float, float]:
    """
    Validate model on test data.
    
    Args:
        model: Model to validate
        test_loader: Data loader for test data
        criterion: Loss function
        device: Device to use for validation
        debug: Whether to print debugging information
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if debug and total % 1000 == 0:
                print(f"Debug - Processed {total} samples, "
                      f"Current Acc: {100.*correct/total:.2f}%")
    
    return test_loss/len(test_loader), 100.*correct/total

def get_trainable_params(
    model: nn.Module,
    approach: str,
    memory_type: str
) -> List[nn.Parameter]:
    """
    Get trainable parameters based on approach.
    This is used for phase 1 training (only train memory components).
    
    Args:
        model: Model to get parameters from
        approach: Integration approach ("mal", "mac", "mag")
        memory_type: Memory module type ("mlp", "simple", "advanced")
        
    Returns:
        List of trainable parameters
    """
    trainable_params = []
    
    if approach == "mal":
        # MAL approach: Freeze backbone, train memory modules
        for name, param in model.named_parameters():
            if ('memory_module' in name or 'MemoryMLP' in name or 
                'NeuralMemory' in name or 'SimpleNeuralMemory' in name or 
                'AdvancedNeuralMemory' in name or 'head' in name):
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
    
    elif approach == "mac":
        # MAC approach: Freeze backbone, train memory and persistent memory
        for name, param in model.named_parameters():
            if ('memory_module' in name or 'persistent_memory' in name or 
                'to_memory_query' in name or 'head' in name):
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
    
    elif approach == "mag":
        # MAG approach: Freeze backbone, train memory, gate, and persistent memory
        for name, param in model.named_parameters():
            if ('memory_module' in name or 'persistent_memory' in name or 
                'gate_norm' in name or 'to_gate' in name or 'head' in name):
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
    
    return trainable_params

def create_phase2_optimizer(
    model: nn.Module,
    approach: str,
    memory_type: str,
    base_lr: float,
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """
    Create optimizer for phase 2 training (fine-tuning).
    This uses different learning rates for backbone and memory components.
    
    Args:
        model: Model to optimize
        approach: Integration approach ("mal", "mac", "mag")
        memory_type: Memory module type ("mlp", "simple", "advanced")
        base_lr: Base learning rate
        weight_decay: Weight decay
        
    Returns:
        Optimizer with parameter groups
    """
    param_groups = []
    
    if approach == "mal":
        param_groups = [
            {'params': [p for n, p in model.named_parameters() 
                      if not any(module in n for module in 
                               ['memory_module', 'MemoryMLP', 'NeuralMemory', 
                                'SimpleNeuralMemory', 'AdvancedNeuralMemory', 'head'])], 
             'lr': base_lr / 10},  # Slower learning rate for backbone
            {'params': [p for n, p in model.named_parameters() 
                      if any(module in n for module in 
                           ['memory_module', 'MemoryMLP', 'NeuralMemory', 
                            'SimpleNeuralMemory', 'AdvancedNeuralMemory', 'head'])], 
             'lr': base_lr}  # Regular learning rate for memory components
        ]
    
    elif approach == "mac":
        param_groups = [
            {'params': [p for n, p in model.named_parameters() 
                      if not any(module in n for module in 
                               ['memory_module', 'persistent_memory', 
                                'to_memory_query', 'head'])], 
             'lr': base_lr / 10},
            {'params': [p for n, p in model.named_parameters() 
                      if any(module in n for module in 
                           ['memory_module', 'persistent_memory', 
                            'to_memory_query', 'head'])], 
             'lr': base_lr}
        ]
    
    elif approach == "mag":
        param_groups = [
            {'params': [p for n, p in model.named_parameters() 
                      if not any(module in n for module in 
                               ['memory_module', 'persistent_memory', 
                                'gate_norm', 'to_gate', 'head'])], 
             'lr': base_lr / 10},
            {'params': [p for n, p in model.named_parameters() 
                      if any(module in n for module in 
                           ['memory_module', 'persistent_memory', 
                            'gate_norm', 'to_gate', 'head'])], 
             'lr': base_lr}
        ]
    
    return optim.AdamW(param_groups, weight_decay=weight_decay)

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    train_acc: float,
    test_loss: float,
    test_acc: float,
    save_path: str,
    is_best: bool = False,
    save_as_epoch: bool = False,
    args: Optional[Dict] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        train_loss: Training loss
        train_acc: Training accuracy
        test_loss: Test loss
        test_acc: Test accuracy
        save_path: Path to save directory
        is_best: Whether this is the best model so far
        save_as_epoch: Whether to save as epoch-specific checkpoint
        args: Optional dictionary of additional arguments to save
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
    }
    
    # Add args if provided
    if args is not None:
        checkpoint['args'] = args
    
    # Save latest checkpoint
    latest_filename = os.path.join(save_path, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_filename)
    
    # Save based on different conditions
    if is_best:
        filename = os.path.join(save_path, 'best_model.pth')
        torch.save(checkpoint, filename)
        print(f"Saved best model with accuracy: {test_acc:.2f}%")
    
    if save_as_epoch:
        filename = os.path.join(save_path, f'checkpoint_epoch{epoch}.pth')
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint for epoch {epoch}")