"""
Data loading utilities for ViT with Titans Memory.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Dict, Any

def get_cifar10_loaders(
    batch_size: int = 64, 
    num_workers: int = 4,
    img_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 data loaders.
    
    Args:
        batch_size: Batch size for data loading
        num_workers: Number of worker threads for data loading
        img_size: Image size for resizing (default: 32)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(img_size) if img_size != 32 else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(img_size) if img_size != 32 else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Load datasets
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

def get_cifar100_loaders(
    batch_size: int = 64, 
    num_workers: int = 4,
    img_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-100 data loaders.
    
    Args:
        batch_size: Batch size for data loading
        num_workers: Number of worker threads for data loading
        img_size: Image size for resizing (default: 32)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(img_size) if img_size != 32 else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(img_size) if img_size != 32 else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load datasets
    train_set = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

def get_tiny_imagenet_loaders(
    batch_size: int = 64, 
    num_workers: int = 4,
    img_size: int = 64,
    data_path: str = './data/tiny-imagenet-200'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create Tiny-ImageNet data loaders.
    Note: Download Tiny-ImageNet dataset first if you use this
    https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4
    
    Args:
        batch_size: Batch size for data loading
        num_workers: Number of worker threads for data loading
        img_size: Image size for resizing (default: 64)
        data_path: Path to the Tiny-ImageNet dataset
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Define transformations for 64x64 images
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(img_size) if img_size != 64 else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(img_size) if img_size != 64 else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Paths to your downloaded Tiny-ImageNet
    train_path = f'{data_path}/train'
    val_path = f'{data_path}/val'
    
    # Load datasets
    train_set = torchvision.datasets.ImageFolder(
        root=train_path, transform=train_transform
    )
    test_set = torchvision.datasets.ImageFolder(
        root=val_path, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

def get_ants_bees_loaders(
    batch_size: int = 32, 
    num_workers: int = 4,
    img_size: int = 224,
    data_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create Ants vs Bees data loaders.
    
    Args:
        batch_size: Batch size for data loading
        num_workers: Number of worker threads for data loading
        img_size: Image size for resizing (default: 224)
        data_path: Path to the ants/bees dataset (if None, will use default path)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # If no data path provided, use the default from kagglehub
    if data_path is None:
        try:
            import kagglehub
            data_path = kagglehub.dataset_download("gauravduttakiit/ants-bees")
            print(f"Using downloaded dataset at: {data_path}")
        except (ImportError, Exception) as e:
            raise ValueError(f"Could not download dataset automatically: {e}. Please provide data_path.")
        
    # Define transformations for 224x224 images
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_path = f"{data_path}/train"
    val_path = f"{data_path}/val"
    
    train_set = torchvision.datasets.ImageFolder(
        root=train_path, transform=train_transform
    )
    test_set = torchvision.datasets.ImageFolder(
        root=val_path, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    print(f"Loaded ants/bees dataset with {len(train_set)} training and {len(test_set)} validation images")
    print(f"Classes: {train_set.classes}")
    
    return train_loader, test_loader

def get_imagenet_loaders(
    batch_size: int = 64, 
    num_workers: int = 8,
    img_size: int = 224,
    data_path: str = './data/imagenet'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create ImageNet data loaders.
    
    Args:
        batch_size: Batch size for data loading
        num_workers: Number of worker threads for data loading
        img_size: Image size for resizing (default: 224)
        data_path: Path to the ImageNet dataset
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_path = f'{data_path}/train'
    val_path = f'{data_path}/val'
    
    train_set = torchvision.datasets.ImageFolder(
        root=train_path, transform=train_transform
    )
    test_set = torchvision.datasets.ImageFolder(
        root=val_path, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader