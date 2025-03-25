# Create a file named test_dataset.py in your project root
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from data.data_loading import get_ants_bees_loaders

def main():
    # Load a small batch
    train_loader, _ = get_ants_bees_loaders(batch_size=4, img_size=224, num_workers=0)  # Set num_workers to 0 to avoid multiprocessing issues
    
    # Get a batch
    images, labels = next(iter(train_loader))
    
    # Display class balance
    print(f"Class distribution in this batch: {labels.tolist()}")
    print(f"Overall dataset size: {len(train_loader.dataset)} images")
    print(f"Class names: {train_loader.dataset.classes}")
    
    # Display a few images
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axs.flat):
        if i < len(images):
            # Convert tensor to image
            img = images[i].permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.set_title(f"Class: {train_loader.dataset.classes[labels[i]]}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_check.png')
    print("Saved sample images to dataset_check.png")

if __name__ == "__main__":
    freeze_support()  # Required for Windows multiprocessing
    main()