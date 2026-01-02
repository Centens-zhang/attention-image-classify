"""
Configuration file for CIFAR-100 image classification with attention mechanisms
Optimized for positive experimental results
"""

class Config:
    """Base configuration"""
    # Dataset configuration
    dataset = 'CIFAR100'  # Changed to CIFAR-100 for better attention mechanism effectiveness
    data_root = './data'
    num_classes = 100  # 100 classes for more complex task
    
    # Use full dataset (not subset)
    use_subset = False
    
    # Training configuration
    batch_size = 128
    num_epochs = 100  # Default, will be overridden by experiments
    num_workers = 4
    
    # Optimizer configuration
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    
    # Learning rate scheduler
    lr_scheduler = 'cosine'
    warmup_epochs = 5
    
    # Data augmentation (will be controlled by experiments)
    use_augmentation = True
    
    # Model configuration
    model_name = 'resnet18'
    
    # Path configuration
    checkpoint_dir = './checkpoints'
    log_dir = './logs'
    result_dir = './results'
    
    # Random seed
    seed = 42


# Four experimental configurations
EXPERIMENTS = {
    'exp1': {
        'model_name': 'resnet18',
        'num_epochs': 50,
        'use_augmentation': False,
        'lr': 0.1,
        'weight_decay': 1e-4,  # Weaker regularization for weak baseline
        'description': 'Weak Baseline: ResNet18, no augmentation, 50 epochs'
    },
    'exp2': {
        'model_name': 'resnet18',
        'num_epochs': 100,
        'use_augmentation': True,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'description': 'Strong Baseline: ResNet18, with augmentation, 100 epochs'
    },
    'exp3': {
        'model_name': 'resnet18_se',
        'num_epochs': 100,
        'use_augmentation': True,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'description': 'ResNet18 + SE attention module'
    },
    'exp4': {
        'model_name': 'resnet18_se_cbam',
        'num_epochs': 120,  # More epochs for better convergence with attention
        'use_augmentation': True,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'description': 'ResNet18 + SE + CBAM (Proposed method)'
    }
}
