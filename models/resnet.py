"""
ResNet18 implementation with attention modules (SE, CBAM) for CIFAR-100 classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""
    
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class BasicBlock(nn.Module):
    """Basic ResNet block with optional attention module."""
    
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, attention=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

        # Attention module
        self.attention = None
        if attention == 'se':
            self.attention = SEBlock(planes)
        elif attention == 'cbam':
            self.attention = CBAMBlock(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.attention is not None:
            out = self.attention(out)
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture with configurable attention modules."""
    
    def __init__(self, block, num_blocks, num_classes=100, attention=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.attention = attention

        # Initial convolution for CIFAR-100 (32x32 input)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Classification head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.attention))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes=100, attention=None):
    """
    Create ResNet18 model.
    
    Args:
        num_classes (int): Number of output classes. Default: 100 for CIFAR-100.
        attention (str): Type of attention module. Options: None, 'se', 'cbam'.
    
    Returns:
        ResNet model instance.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, attention=attention)


def get_model(model_name='resnet18', num_classes=100, attention=None):
    """
    Factory function to get model by name.
    
    Args:
        model_name (str): Name of the model. Default: 'resnet18'.
        num_classes (int): Number of output classes. Default: 100.
        attention (str): Type of attention module. Options: None, 'se', 'cbam'.
    
    Returns:
        Model instance.
    
    Example:
        >>> model = get_model('resnet18', num_classes=100, attention='se')
        >>> model = get_model('resnet18', num_classes=100, attention='cbam')
        >>> model = get_model('resnet18', num_classes=100, attention=None)
    """
    if model_name.lower() == 'resnet18':
        return ResNet18(num_classes=num_classes, attention=attention)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == '__main__':
    # Test the models
    print("Testing ResNet18 models...")
    
    # Test vanilla ResNet18
    model = get_model('resnet18', num_classes=100, attention=None)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Vanilla ResNet18 output shape: {y.shape}")
    
    # Test ResNet18 with SE attention
    model_se = get_model('resnet18', num_classes=100, attention='se')
    y_se = model_se(x)
    print(f"ResNet18-SE output shape: {y_se.shape}")
    
    # Test ResNet18 with CBAM attention
    model_cbam = get_model('resnet18', num_classes=100, attention='cbam')
    y_cbam = model_cbam(x)
    print(f"ResNet18-CBAM output shape: {y_cbam.shape}")
    
    # Print number of parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nNumber of parameters:")
    print(f"Vanilla ResNet18: {count_parameters(model):,}")
    print(f"ResNet18-SE: {count_parameters(model_se):,}")
    print(f"ResNet18-CBAM: {count_parameters(model_cbam):,}")
