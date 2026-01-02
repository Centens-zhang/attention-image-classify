"""
Dataset loader for CIFAR-100
Optimized data augmentation for attention mechanism experiments
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloaders(config):
    """
    Get CIFAR-100 dataloaders with configurable augmentation
    
    Args:
        config: Configuration object with dataset settings
        
    Returns:
        trainloader, testloader: PyTorch DataLoader objects
    """
    
    # Normalization parameters for CIFAR-100
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
    
    # Training transforms
    if config.use_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    # Test transforms
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load CIFAR-100
    print(f"Loading CIFAR-100 dataset...")
    print(f"Data augmentation: {'Enabled' if config.use_augmentation else 'Disabled'}")
    
    trainset = torchvision.datasets.CIFAR100(
        root=config.data_root,
        train=True,
        download=True,
        transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR100(
        root=config.data_root,
        train=False,
        download=True,
        transform=transform_test
    )
    
    print(f"✅ Training samples: {len(trainset)}")
    print(f"✅ Test samples: {len(testset)}")
    
    trainloader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    testloader = DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return trainloader, testloader
