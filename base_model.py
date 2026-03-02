"""
Base U-Net model class with common functionality
All U-Net variants inherit from this base class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseUNet(nn.Module, ABC):
    """
    Abstract base class for U-Net models
    Provides common functionality for all U-Net variants
    """
    
    def __init__(self, n_channels=4, n_classes=5, bilinear=False):
        super(BaseUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Color map for visualization (shared across all models)
        self.color_map = {
            0: [0, 0, 0],        # Background (transparent)
            1: [255, 0, 0],      # Necrotic tumor (red)
            2: [255, 255, 0],    # Edema (yellow)
            3: [0, 0, 255],      # Enhancing tumor (blue)
            4: [0, 255, 0],      # Healthy tissue (green)
        }
    
    @abstractmethod
    def forward(self, x):
        """Forward pass - must be implemented by subclasses"""
        pass
    
    def predict(self, x):
        """
        Get probability predictions
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Tensor with softmax probabilities (B, n_classes, H, W)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            # Handle deep supervision outputs
            if isinstance(logits, list):
                logits = logits[-1]
            return F.softmax(logits, dim=1)
    
    def get_segmentation_mask(self, x):
        """
        Get predicted segmentation mask
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Tensor with class indices (B, H, W)
        """
        probs = self.predict(x)
        return torch.argmax(probs, dim=1)
    
    def create_labeled_overlay(self, input_image, segmentation_mask, alpha=0.5):
        """
        Create a labeled overlay image with color-coded segmentation
        
        Args:
            input_image: Original brain scan (B, 1, H, W) or (B, H, W)
            segmentation_mask: Predicted segmentation (B, H, W)
            alpha: Transparency factor for overlay
        
        Returns:
            RGB image with colored overlay (B, 3, H, W)
        """
        batch_size, height, width = segmentation_mask.shape
        
        # Normalize input image to [0, 1]
        if len(input_image.shape) == 4:
            input_image = input_image[:, 0, :, :]  # Take first channel if multi-channel
        
        input_normalized = (input_image - input_image.min()) / (input_image.max() - input_image.min() + 1e-8)
        
        # Create RGB overlay
        overlay = torch.zeros(batch_size, 3, height, width, device=input_image.device)
        
        for class_idx, color in self.color_map.items():
            if class_idx >= self.n_classes:
                continue  # Skip if class not in model output
            
            mask = (segmentation_mask == class_idx).float()
            for channel in range(3):
                if class_idx == 0:  # Background - show original image
                    overlay[:, channel, :, :] += mask * input_normalized * 255
                else:  # Colored overlay
                    overlay[:, channel, :, :] += mask * (
                        alpha * color[channel] + (1 - alpha) * input_normalized * 255
                    )
        
        return overlay.clamp(0, 255).byte()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def count_parameters(self):
        """Count model parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def get_model_info(self):
        """Get model information as a dictionary"""
        total, trainable = self.count_parameters()
        return {
            'model_name': self.__class__.__name__,
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'bilinear': self.bilinear,
            'total_params': total,
            'trainable_params': trainable,
        }


class DoubleConv(nn.Module):
    """Shared Double Convolution block: (Conv2d -> BatchNorm -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Shared Downsampling block: MaxPool2d -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Shared Upsampling block: Upsample -> DoubleConv with skip connection"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        x1: feature map from decoder path
        x2: feature map from encoder path (skip connection)
        """
        x1 = self.up(x1)
        
        # Handle input sizes that are not perfectly divisible
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Shared final 1x1 convolution for output"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
