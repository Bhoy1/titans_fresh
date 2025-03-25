#!/usr/bin/env python
"""
Evaluation script for ViT with Titans memory integration.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import model components
from models.model_factory import create_vit_with_memory

# Import data utilities
from data.data_loading import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_tiny_imagenet_loaders,
    get_ants_bees_loaders,
    get_imagenet_loaders
)

# Import configurations
from configs.default_configs import (
    DATASET_CONFIGS,
    MODEL_CONFIGS,
    MEMORY_CONFIGS,
    INTEGRATION_CONFIGS
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate ViT with Titans Memory')
    
    # Model and checkpoint parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation')
    
    # Evaluation options
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU usage')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize confusion matrix and save plots')
    parser.add_argument('--plot-dir', type=str, default='./plots',
                        help='Directory to save visualization plots')
    
    return parser.parse_args()

def evaluate(model, test_loader, device, classes=None):
    """
    Evaluate model on test data.
    
    Args:
        model: Model to evaluate
        test_loader: Data loader for test data
        device: Device to use for evaluation
        classes: Optional list of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Calculate per-class accuracy
    per_class_acc = 100. * cm.diagonal() / cm.sum(axis=1)
    
    # Create classification report
    if classes is not None:
        report = classification_report(all_targets, all_preds, target_names=classes)
    else:
        report = classification_report(all_targets, all_preds)
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'targets': all_targets
    }

def visualize_results(results, classes=None, save_dir=None):
    """
    Visualize evaluation results.
    
    Args:
        results: Dictionary with evaluation results
        classes: Optional list of class names
        save_dir: Directory to save plots
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    cm = results['confusion_matrix']
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    ax = sns.heatmap(
        cm_norm, annot=False, cmap='Blues', 
        xticklabels=classes if classes else "auto",
        yticklabels=classes if classes else "auto"
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), bbox_inches='tight')
    
    # Plot per-class accuracy
    plt.figure(figsize=(12, 6))
    per_class_acc = results['per_class_accuracy']
    x = np.arange(len(per_class_acc))
    plt.bar(x, per_class_acc)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    
    if classes:
        if len(classes) <= 20:  # Only show class names if there aren't too many
            plt.xticks(x, classes, rotation=90)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), bbox_inches='tight')
    
    plt.show()

def main():
    """Main function to run evaluation."""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract configurations from checkpoint
    ckpt_args = checkpoint.get('args', {})
    model_size = ckpt_args.get('model_size', 'tiny')
    approach = ckpt_args.get('approach', 'mag')
    memory_type = ckpt_args.get('memory_type', 'mlp')
    dataset_name = ckpt_args.get('dataset', 'cifar100')
    qkv_diff_views = ckpt_args.get('qkv_diff_views', False)
    
    print(f"Loaded checkpoint trained on {dataset_name} dataset")
    print(f"Model: {model_size} ViT with {approach.upper()} integration and {memory_type} memory")
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIGS[dataset_name]
    img_size = dataset_config['image_size']
    
    # Load dataset and class names
    if dataset_name == 'cifar10':
        _, test_loader = get_cifar10_loaders(batch_size=args.batch_size, img_size=img_size)
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name == 'cifar100':
        _, test_loader = get_cifar100_loaders(batch_size=args.batch_size, img_size=img_size)
        classes = None  # Too many classes to list
    elif dataset_name == 'tiny-imagenet':
        _, test_loader = get_tiny_imagenet_loaders(batch_size=args.batch_size, img_size=img_size)
        classes = None  # Too many classes to list
    elif dataset_name == 'ants-bees':
        _, test_loader = get_ants_bees_loaders(batch_size=args.batch_size, img_size=img_size)
        classes = ['ants', 'bees']
    elif dataset_name == 'imagenet':
        _, test_loader = get_imagenet_loaders(batch_size=args.batch_size, img_size=img_size)
        classes = None  # Too many classes to list
    
    # Get configs
    memory_config = MEMORY_CONFIGS[memory_type]
    integration_config = INTEGRATION_CONFIGS[approach]
    
    # Create model
    model = create_vit_with_memory(
        model_size=model_size,
        approach=approach,
        memory_type=memory_type,
        num_classes=dataset_config['num_classes'],
        image_size=dataset_config['image_size'],
        pretrained=True,  # Not important for evaluation since we'll load weights
        qkv_receives_diff_views=qkv_diff_views,
        **{**memory_config, **integration_config}
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate model
    results = evaluate(model, test_loader, device, classes)
    
    # Print results
    print(f"Test accuracy: {results['accuracy']:.2f}%")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Visualize results if requested
    if args.visualize:
        visualize_results(results, classes, args.plot_dir)

if __name__ == "__main__":
    main()