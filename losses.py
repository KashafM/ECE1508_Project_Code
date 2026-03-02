import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    
    Args:
        smooth: Smoothing factor to avoid division by zero
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    
    def __init__(self, smooth=1e-5, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, input, target):
        """
        Args:
            input: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        
        Returns:
            Dice loss value
        """
        num_classes = input.size(1)
        input = F.softmax(input, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate Dice loss for each class
        dice_losses = []
        for i in range(num_classes):
            input_flat = input[:, i, :, :].contiguous().view(-1)
            target_flat = target_one_hot[:, i, :, :].contiguous().view(-1)
            
            intersection = (input_flat * target_flat).sum()
            dice = (2.0 * intersection + self.smooth) / (
                input_flat.sum() + target_flat.sum() + self.smooth
            )
            dice_losses.append(1 - dice)
        
        dice_loss = torch.stack(dice_losses)
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Args:
        alpha: Weighting factor for class balance (scalar or list)
        gamma: Focusing parameter for hard examples
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input, target):
        """
        Args:
            input: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(input, target, reduction='none')
        
        # Get probabilities
        p = F.softmax(input, dim=1)
        
        # Get class probabilities at target indices
        batch_size, _, h, w = input.shape
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Apply focal term
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Assume alpha is a tensor of class weights
                alpha_t = torch.tensor(self.alpha, device=input.device)
                alpha_t = alpha_t.gather(0, target.view(-1)).view_as(target)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss with adjustable FP/FN weights
    
    Args:
        alpha: Weight for false positives
        beta: Weight for false negatives
        smooth: Smoothing factor
        reduction: Reduction method
    """
    
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5, reduction='mean'):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, input, target):
        """
        Args:
            input: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        
        Returns:
            Tversky loss value
        """
        num_classes = input.size(1)
        input = F.softmax(input, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate Tversky loss for each class
        tversky_losses = []
        for i in range(num_classes):
            input_flat = input[:, i, :, :].contiguous().view(-1)
            target_flat = target_one_hot[:, i, :, :].contiguous().view(-1)
            
            # True positives, false positives, and false negatives
            tp = (input_flat * target_flat).sum()
            fp = (input_flat * (1 - target_flat)).sum()
            fn = ((1 - input_flat) * target_flat).sum()
            
            tversky_index = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
            )
            tversky_losses.append(1 - tversky_index)
        
        tversky_loss = torch.stack(tversky_losses)
        
        if self.reduction == 'mean':
            return tversky_loss.mean()
        elif self.reduction == 'sum':
            return tversky_loss.sum()
        else:
            return tversky_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function using multiple loss components
    
    Args:
        ce_weight: Weight for Cross Entropy loss
        dice_weight: Weight for Dice loss
        focal_weight: Weight for Focal loss
        use_deep_supervision: Whether to apply loss at multiple scales
    """
    
    def __init__(self, ce_weight=1.0, dice_weight=1.0, focal_weight=0.0, 
                 use_deep_supervision=False):
        super(CombinedLoss, self).__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.use_deep_supervision = use_deep_supervision
        
        # Initialize loss components
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=2.0) if focal_weight > 0 else None
    
    def forward(self, input, target):
        """
        Args:
            input: Predicted logits (B, C, H, W) or list of outputs for deep supervision
            target: Ground truth labels (B, H, W)
        
        Returns:
            Combined loss value
        """
        if self.use_deep_supervision and isinstance(input, list):
            # Apply loss at multiple scales
            total_loss = 0
            weights = [0.5, 0.3, 0.2]  # Weights for different scales
            
            for i, (output, weight) in enumerate(zip(input, weights[:len(input)])):
                # Ensure output and target have the same spatial dimensions
                if output.shape[2:] != target.shape[1:]:
                    # Interpolate output to match target size
                    output = F.interpolate(output, size=target.shape[1:], mode='bilinear', align_corners=True)
                loss = self._compute_loss(output, target)
                total_loss += weight * loss
            
            return total_loss
        else:
            # Single scale loss
            if isinstance(input, list):
                input = input[-1]  # Use final output
            return self._compute_loss(input, target)
    
    def _compute_loss(self, input, target):
        """Compute combined loss for a single scale"""
        total_loss = 0
        
        if self.ce_weight > 0:
            total_loss += self.ce_weight * self.ce_loss(input, target)
        
        if self.dice_weight > 0:
            total_loss += self.dice_weight * self.dice_loss(input, target)
        
        if self.focal_weight > 0 and self.focal_loss is not None:
            total_loss += self.focal_weight * self.focal_loss(input, target)
        
        return total_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance
    
    Args:
        weight: Manual rescaling weight for each class
        ignore_index: Index to ignore in loss computation
        reduction: Reduction method
    """
    
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        
        if weight is not None:
            self.weight = torch.tensor(weight, dtype=torch.float32)
        else:
            # Default weights for brain tumor segmentation
            # [background, necrotic, edema, enhancing, healthy]
            self.weight = torch.tensor([0.1, 1.5, 1.2, 1.5, 0.5], dtype=torch.float32)
        
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, input, target):
        """
        Args:
            input: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        
        Returns:
            Weighted cross entropy loss
        """
        # Move weight to same device as input
        weight = self.weight.to(input.device)
        
        return F.cross_entropy(
            input, target, 
            weight=weight, 
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for improving segmentation boundaries
    
    This loss focuses on the boundaries of segmented regions
    """
    
    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, input, target):
        """
        Args:
            input: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        
        Returns:
            Boundary loss value
        """
        n, c, h, w = input.shape
        
        # Softmax on predictions
        input_soft = F.softmax(input, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=c)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Compute boundaries using gradients
        target_boundary = self._compute_boundary(target_one_hot)
        pred_boundary = self._compute_boundary(input_soft)
        
        # Compute boundary loss
        boundary_loss = F.mse_loss(pred_boundary, target_boundary)
        
        return boundary_loss
    
    def _compute_boundary(self, mask):
        """Compute boundaries using Sobel filters"""
        # Simplified boundary computation using differences
        laplacian = mask[:, :, 1:, :] - mask[:, :, :-1, :]
        laplacian = torch.abs(laplacian)
        
        return laplacian


def get_loss_function(loss_name='combined', **kwargs):
    """
    Factory function to get loss function by name
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional parameters for the loss function
    
    Returns:
        Loss function instance
    """
    loss_functions = {
        'ce': nn.CrossEntropyLoss,
        'dice': DiceLoss,
        'focal': FocalLoss,
        'tversky': TverskyLoss,
        'combined': CombinedLoss,
        'weighted_ce': WeightedCrossEntropyLoss,
        'boundary': BoundaryLoss,
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_functions[loss_name](**kwargs)


if __name__ == "__main__":
    print("Testing loss functions...")
    print("-" * 60)
    
    # Create dummy data
    batch_size = 2
    num_classes = 5
    height, width = 256, 256
    
    # Random predictions and targets
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test different loss functions
    losses = {
        'CrossEntropy': nn.CrossEntropyLoss(),
        'Dice': DiceLoss(),
        'Focal': FocalLoss(),
        'Tversky': TverskyLoss(),
        'Combined': CombinedLoss(),
        'WeightedCE': WeightedCrossEntropyLoss(),
    }
    
    print("Loss values on random data:")
    for name, loss_fn in losses.items():
        loss_value = loss_fn(predictions, targets)
        print(f"  {name}: {loss_value.item():.4f}")
    
    print("\nLoss functions loaded successfully!")
