#!/usr/bin/env python
"""
Script to run multiple ViT with Titans Memory experiments.
"""

import os
import argparse
import itertools
import subprocess
from datetime import datetime
import json

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ViT with Titans Memory experiments')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'tiny-imagenet', 'ants-bees', 'imagenet'],
                        help='Dataset to use')
    
    # Model parameters
    parser.add_argument('--model-sizes', type=str, default='tiny',
                        help='Comma-separated list of model sizes to run (tiny,small,base)')
    
    parser.add_argument('--approaches', type=str, default='mag',
                        help='Comma-separated list of approaches to run (mal,mac,mag)')
    
    parser.add_argument('--memory-types', type=str, default='advanced',
                        help='Comma-separated list of memory types to run (mlp,simple,advanced)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Disable use of pretrained weights')
    
    # Other parameters
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU usage')
    parser.add_argument('--qkv-diff-views', action='store_true',
                        help='Use different views for QKV in advanced memory')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing them')
    
    return parser.parse_args()

def main():
    """Main function to run experiments."""
    args = parse_args()
    
    # Parse comma-separated lists
    model_sizes = [size.strip() for size in args.model_sizes.split(',')]
    approaches = [approach.strip() for approach in args.approaches.split(',')]
    memory_types = [mem_type.strip() for mem_type in args.memory_types.split(',')]
    
    # Validate model sizes
    valid_sizes = ['tiny', 'small', 'base']
    for size in model_sizes:
        if size not in valid_sizes:
            raise ValueError(f"Invalid model size: {size}. Choose from {valid_sizes}")
    
    # Validate approaches
    valid_approaches = ['mal', 'mac', 'mag']
    for approach in approaches:
        if approach not in valid_approaches:
            raise ValueError(f"Invalid approach: {approach}. Choose from {valid_approaches}")
    
    # Validate memory types
    valid_memory_types = ['none', 'mlp', 'simple', 'advanced']
    for mem_type in memory_types:
        if mem_type not in valid_memory_types:
            raise ValueError(f"Invalid memory type: {mem_type}. Choose from {valid_memory_types}")
    
    # Create combinations
    combinations = list(itertools.product(model_sizes, approaches, memory_types))
    print(f"Will run {len(combinations)} experiments with the following configurations:")
    
    for i, (model_size, approach, memory_type) in enumerate(combinations):
        print(f"{i+1}. Model: {model_size}, Approach: {approach}, Memory: {memory_type}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save run configuration
    run_config = {
        'timestamp': timestamp,
        'args': vars(args),
        'combinations': [
            {'model_size': m, 'approach': a, 'memory_type': t} 
            for m, a, t in combinations
        ]
    }
    
    with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    
    # Create results summary
    results_summary = {
        'experiments': []
    }
    
    # Run each experiment
    for i, (model_size, approach, memory_type) in enumerate(combinations):
        print(f"\n{'='*80}")
        print(f"Running experiment {i+1}/{len(combinations)}")
        print(f"Configuration: {model_size}-{approach}-{memory_type}")
        print(f"{'='*80}\n")
        
        # Create experiment name
        exp_name = f"{model_size}_{approach}_{memory_type}_{args.dataset}"
        exp_dir = os.path.join(run_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Build command
        cmd = [
            "python", "train.py",
            "--dataset", args.dataset,
            "--model-size", model_size,
            "--approach", approach,
            "--memory-type", memory_type,
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--save-dir", exp_dir
        ]
        
        # Add optional arguments
        if args.no_pretrained:
            cmd.append("--no-pretrained")
        if args.no_gpu:
            cmd.append("--no-gpu")
        if args.qkv_diff_views:
            cmd.append("--qkv-diff-views")
        
        # Print command
        cmd_str = " ".join(cmd)
        print(f"Running: {cmd_str}")
        
        # Execute command
        if not args.dry_run:
            start_time = datetime.now()
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                status = 'success'
                
                # Extract best accuracy from output
                output = result.stdout
                accuracy = None
                for line in output.splitlines():
                    if "Training completed! Best accuracy:" in line:
                        accuracy = float(line.split(":")[-1].strip().rstrip('%'))
                
                print(f"Experiment completed successfully.")
                print(f"Best accuracy: {accuracy}%")
            except subprocess.CalledProcessError as e:
                status = 'failed'
                accuracy = None
                print(f"Experiment failed with error: {e}")
                print("Error output:")
                print(e.stdout)
                print(e.stderr)
                print("Continuing with next experiment...")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Save experiment result
            exp_result = {
                'model_size': model_size,
                'approach': approach,
                'memory_type': memory_type,
                'dataset': args.dataset,
                'status': status,
                'accuracy': accuracy,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration
            }
            
            results_summary['experiments'].append(exp_result)
            
            # Update results file after each experiment
            with open(os.path.join(run_dir, 'results_summary.json'), 'w') as f:
                json.dump(results_summary, f, indent=2)
        else:
            print("Dry run: command not executed")
    
    # Print final summary
    if not args.dry_run:
        print("\nAll experiments completed. Summary:")
        successful = [exp for exp in results_summary['experiments'] if exp['status'] == 'success']
        if successful:
            best_exp = max(successful, key=lambda x: x['accuracy'] or 0)
            print(f"Best configuration: {best_exp['model_size']}-{best_exp['approach']}-{best_exp['memory_type']}")
            print(f"Best accuracy: {best_exp['accuracy']}%")
        
        print(f"Results saved to: {run_dir}")
    else:
        print("\nDry run completed. No experiments were executed.")

if __name__ == "__main__":
    main()