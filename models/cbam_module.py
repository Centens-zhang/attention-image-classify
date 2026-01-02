"""
CBAM: Convolutional Block Attention Module
Paper: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
Authors: Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Applies attention mechanism along the channel dimension using both
    max pooling and average pooling features.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Args:
            in_channels (int): Number of input channels
            reduction_ratio (int): Reduction ratio for the intermediate layer (default: 16)
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map with shape (B, C, H, W)
        
        Returns:
            Tensor: Channel attention weighted feature map with shape (B, C, H, W)
        """
        # Average pooling and max pooling
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        attention = self.sigmoid(out)
        
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Applies attention mechanism along the spatial dimension using both
    max pooling and average pooling across channels.
    """
    def __init__(self, kernel_size=7):
        """
        Args:
            kernel_size (int): Kernel size for the convolutional layer (default: 7)
        """
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map with shape (B, C, H, W)
        
        Returns:
            Tensor: Spatial attention weighted feature map with shape (B, C, H, W)
        """
        # Channel-wise max pooling and average pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        out = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid
        attention = self.sigmoid(self.conv(out))
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines channel attention and spatial attention sequentially.
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        """
        Args:
            in_channels (int): Number of input channels
            reduction_ratio (int): Reduction ratio for channel attention (default: 16)
            kernel_size (int): Kernel size for spatial attention (default: 7)
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map with shape (B, C, H, W)
        
        Returns:
            Tensor: CBAM attention weighted feature map with shape (B, C, H, W)
        """
        # Apply channel attention first
        out = self.channel_attention(x)
        
        # Then apply spatial attention
        out = self.spatial_attention(out)
        
        return out


# Example usage and testing
if __name__ == "__main__":
    # Test CBAM module
    batch_size = 4
    channels = 64
    height, width = 32, 32
    
    # Create sample input
    x = torch.randn(batch_size, channels, height, width)
    
    # Test Channel Attention
    print("Testing Channel Attention...")
    ca = ChannelAttention(channels)
    ca_out = ca(x)
    print(f"Input shape: {x.shape}")
    print(f"Channel Attention output shape: {ca_out.shape}")
    assert ca_out.shape == x.shape, "Channel Attention output shape mismatch"
    
    # Test Spatial Attention
    print("\nTesting Spatial Attention...")
    sa = SpatialAttention()
    sa_out = sa(x)
    print(f"Input shape: {x.shape}")
    print(f"Spatial Attention output shape: {sa_out.shape}")
    assert sa_out.shape == x.shape, "Spatial Attention output shape mismatch"
    
    # Test CBAM
    print("\nTesting CBAM...")
    cbam = CBAM(channels)
    cbam_out = cbam(x)
    print(f"Input shape: {x.shape}")
    print(f"CBAM output shape: {cbam_out.shape}")
    assert cbam_out.shape == x.shape, "CBAM output shape mismatch"
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nChannel Attention parameters: {count_parameters(ca)}")
    print(f"Spatial Attention parameters: {count_parameters(sa)}")
    print(f"CBAM parameters: {count_parameters(cbam)}")
    
    print("\nAll tests passed successfully!")
