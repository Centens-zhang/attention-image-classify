"""
Training script for CIFAR-100 image classification with attention mechanism.
Author: Centens-zhang
Date: 2026-01-02
"""

import os
import time
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_data_loaders(batch_size=128, num_workers=4, data_dir='./data'):
    """
    Create data loaders for CIFAR-100 dataset.
    
    Args:
        batch_size: Batch size for training and validation
        num_workers: Number of workers for data loading
        data_dir: Directory to store/load dataset
    
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])

    # Only normalization for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])

    # Load CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, epoch, device, args):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        epoch: Current epoch number
        device: Device to train on (cuda/cpu)
        args: Training arguments
    
    Returns:
        avg_loss: Average training loss
        avg_acc: Average training accuracy
    """
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [Train]')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc@1': f'{top1.avg:.2f}%',
            'acc@5': f'{top5.avg:.2f}%'
        })
    
    logger.info(f'Epoch {epoch} - Train Loss: {losses.avg:.4f}, '
                f'Train Acc@1: {top1.avg:.2f}%, Train Acc@5: {top5.avg:.2f}%')
    
    return losses.avg, top1.avg, top5.avg


def validate(model, val_loader, criterion, epoch, device, args):
    """
    Validate the model.
    
    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        epoch: Current epoch number
        device: Device to validate on (cuda/cpu)
        args: Training arguments
    
    Returns:
        avg_loss: Average validation loss
        avg_acc: Average validation accuracy
    """
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # Progress bar
    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{args.epochs} [Val]')
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc@1': f'{top1.avg:.2f}%',
                'acc@5': f'{top5.avg:.2f}%'
            })
    
    logger.info(f'Epoch {epoch} - Val Loss: {losses.avg:.4f}, '
                f'Val Acc@1: {top1.avg:.2f}%, Val Acc@5: {top5.avg:.2f}%')
    
    return losses.avg, top1.avg, top5.avg


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions.
    
    Args:
        output: Model predictions
        target: Ground truth labels
        topk: Tuple of k values
    
    Returns:
        List of accuracies for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, checkpoint_dir='checkpoints'):
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model state and training info
        is_best: Boolean indicating if this is the best model so far
        checkpoint_dir: Directory to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{state["epoch"]}.pth')
    torch.save(state, filename)
    logger.info(f'Checkpoint saved to {filename}')
    
    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_filename)
        logger.info(f'Best model saved to {best_filename}')


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, args):
    """
    Main training loop.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        args: Training arguments
    
    Returns:
        model: Trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logger.info(f'Training on device: {device}')
    logger.info(f'Model: {model.__class__.__name__}')
    logger.info(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    logger.info(f'Training arguments: {args}')
    
    best_acc = 0.0
    start_time = time.time()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc1': [],
        'train_acc5': [],
        'val_loss': [],
        'val_acc1': [],
        'val_acc5': [],
        'lr': []
    }
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch}/{args.epochs} - Learning Rate: {current_lr:.6f}')
        
        # Train for one epoch
        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer, epoch, device, args
        )
        
        # Validate
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, epoch, device, args
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc1'].append(train_acc1)
        history['train_acc5'].append(train_acc5)
        history['val_loss'].append(val_loss)
        history['val_acc1'].append(val_acc1)
        history['val_acc5'].append(val_acc5)
        history['lr'].append(current_lr)
        
        # Check if this is the best model
        is_best = val_acc1 > best_acc
        if is_best:
            best_acc = val_acc1
            logger.info(f'New best validation accuracy: {best_acc:.2f}%')
        
        # Save checkpoint
        if epoch % args.save_freq == 0 or is_best:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'history': history,
                'args': args
            }, is_best, args.checkpoint_dir)
        
        epoch_time = time.time() - epoch_start
        logger.info(f'Epoch {epoch} completed in {epoch_time:.2f}s\n')
    
    total_time = time.time() - start_time
    logger.info(f'Training completed in {total_time / 3600:.2f} hours')
    logger.info(f'Best validation accuracy: {best_acc:.2f}%')
    
    return model


def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 Training with Attention')
    
    # Data parameters
    parser.add_argument('--data-dir', default='./data', type=str,
                        help='path to dataset')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='batch size for training')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='number of data loading workers')
    
    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str,
                        help='model architecture (resnet50, resnet101, etc.)')
    parser.add_argument('--num-classes', default=100, type=int,
                        help='number of classes')
    
    # Training parameters
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of training epochs')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--scheduler', default='cosine', type=str,
                        choices=['step', 'cosine', 'plateau'],
                        help='learning rate scheduler')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint-dir', default='checkpoints', type=str,
                        help='path to save checkpoints')
    parser.add_argument('--save-freq', default=10, type=int,
                        help='save checkpoint every N epochs')
    parser.add_argument('--resume', default='', type=str,
                        help='path to checkpoint to resume from')
    
    # Other parameters
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )
    
    # Create model (placeholder - you should import your actual model)
    # Example: from models import AttentionResNet
    # model = AttentionResNet(num_classes=args.num_classes)
    
    # For demonstration, using a simple ResNet from torchvision
    from torchvision import models
    if args.model == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    elif args.model == 'resnet101':
        model = models.resnet101(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    else:
        raise ValueError(f'Unknown model: {args.model}')
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Define learning rate scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 120, 160], gamma=0.2
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=10
        )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f'Loading checkpoint from {args.resume}')
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f'Resumed from epoch {checkpoint["epoch"]}')
        else:
            logger.warning(f'No checkpoint found at {args.resume}')
    
    # Start training
    logger.info('='*80)
    logger.info('Starting training...')
    logger.info('='*80)
    
    trained_model = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler, args
    )
    
    logger.info('='*80)
    logger.info('Training finished!')
    logger.info('='*80)


if __name__ == '__main__':
    main()
