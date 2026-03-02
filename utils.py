import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from datetime import datetime

from training.metrics import (
    dice_coefficient,
    multi_class_dice,
    iou_score,
    pixel_accuracy,
    MetricsTracker,
)


def visualize_segmentation(image, mask, prediction=None, save_path=None, title="Brain Tumor Segmentation"):
    """
    Visualize brain MRI image with segmentation overlay
    
    Args:
        image: Input image (C, H, W) or (H, W)
        mask: Ground truth mask (H, W)
        prediction: Predicted mask (H, W) - optional
        save_path: Path to save the figure
        title: Title for the plot
    """
    # Convert tensors to numpy
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if prediction is not None and torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    
    # Handle multi-channel images
    if len(image.shape) == 3:
        image = image[0]  # Take first channel for visualization
    
    # Normalize image for display
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Create color map for segmentation
    colors = [
        [0, 0, 0, 0],        # Background (transparent)
        [1, 0, 0, 0.7],      # Necrotic tumor (red)
        [1, 1, 0, 0.7],      # Edema (yellow)
        [0, 0, 1, 0.7],      # Enhancing tumor (blue)
        [0, 1, 0, 0.7],      # Healthy tissue (green)
    ]
    cmap = ListedColormap(colors)
    
    # Create figure
    num_plots = 3 if prediction is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original MRI')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(mask, cmap=cmap, alpha=0.5, vmin=0, vmax=4)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    if prediction is not None:
        axes[2].imshow(image, cmap='gray')
        axes[2].imshow(prediction, cmap=cmap, alpha=0.5, vmin=0, vmax=4)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    plt.close()


def save_checkpoint(model, optimizer, epoch, loss, save_path, model_name="model"):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        save_path: Directory to save checkpoint
        model_name: Name of the model
    """
    os.makedirs(save_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_epoch{epoch}_{timestamp}.pth"
    filepath = os.path.join(save_path, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    
    print(f"Checkpoint saved: {filepath}")
    return filepath


def load_checkpoint(model, checkpoint_path, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to load state (optional)
    
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    
    return checkpoint


def print_metrics(metrics_dict, prefix=""):
    """
    Pretty print metrics
    
    Args:
        metrics_dict: Dictionary of metrics
        prefix: Prefix for printing (e.g., "Train", "Val")
    """
    if prefix:
        print(f"\n{prefix} Metrics:")
    else:
        print("\nMetrics:")
    
    print("-" * 40)
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("-" * 40)


def create_visualization_grid(images, masks, predictions, num_samples=4):
    """
    Create a grid visualization of multiple samples
    
    Args:
        images: Batch of images (B, C, H, W)
        masks: Batch of ground truth masks (B, H, W)
        predictions: Batch of predictions (B, H, W)
        num_samples: Number of samples to display
    
    Returns:
        Figure object
    """
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Color map for segmentation
    colors = [
        [0, 0, 0, 0],        # Background
        [1, 0, 0, 0.7],      # Necrotic tumor
        [1, 1, 0, 0.7],      # Edema
        [0, 0, 1, 0.7],      # Enhancing tumor
        [0, 1, 0, 0.7],      # Healthy tissue
    ]
    cmap = ListedColormap(colors)
    
    for i in range(num_samples):
        # Get image (take first channel if multi-channel)
        img = images[i].cpu().numpy()
        if len(img.shape) == 3:
            img = img[0]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Ground truth mask
        gt_mask = masks[i].cpu().numpy()
        
        # Prediction
        pred_mask = predictions[i].cpu().numpy()
        
        # Original image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1}: Original')
        axes[i, 0].axis('off')
        
        # Ground truth overlay
        axes[i, 1].imshow(img, cmap='gray')
        axes[i, 1].imshow(gt_mask, cmap=cmap, alpha=0.5, vmin=0, vmax=4)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction overlay
        axes[i, 2].imshow(img, cmap='gray')
        axes[i, 2].imshow(pred_mask, cmap=cmap, alpha=0.5, vmin=0, vmax=4)
        
        # Calculate and show Dice score
        dice = multi_class_dice(
            torch.tensor(pred_mask).unsqueeze(0),
            torch.tensor(gt_mask).unsqueeze(0),
            num_classes=5
        )
        axes[i, 2].set_title(f'Prediction (Dice: {dice["mean_dice"]:.3f})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience: How long to wait after last improvement
            verbose: If True, prints a message for each improvement
            delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
    
    def __call__(self, val_loss, model=None):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        self.val_loss_min = val_loss


class CheckpointSaver:
    """Save model checkpoints"""
    
    def __init__(self, checkpoint_dir='checkpoints', max_checkpoints=5, verbose=False):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            verbose: Whether to print messages
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        self.verbose = verbose
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, model, optimizer, epoch, loss, metrics=None, model_name="model"):
        """Save checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_epoch{epoch}_{timestamp}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, filepath)
        self.checkpoints.append(filepath)
        
        # Remove old checkpoints if exceeding max
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                if self.verbose:
                    print(f"Removed old checkpoint: {old_checkpoint}")
        
        return filepath
    
    def load_latest(self, model, optimizer=None):
        """Load the latest checkpoint"""
        if not self.checkpoints:
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
            if checkpoints:
                checkpoints.sort()
                self.checkpoints = [os.path.join(self.checkpoint_dir, f) for f in checkpoints]
        
        if self.checkpoints:
            return load_checkpoint(model, self.checkpoints[-1], optimizer)
        return None


if __name__ == "__main__":
    print("Testing utility functions...")
    print("-" * 60)
    
    # Test metrics calculation
    batch_size = 2
    height, width = 256, 256
    num_classes = 5
    
    # Create dummy predictions and targets
    pred = torch.randint(0, num_classes, (batch_size, height, width))
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Calculate metrics
    dice_scores = multi_class_dice(pred, target, num_classes)
    accuracy = pixel_accuracy(pred, target)
    
    print("Sample Metrics:")
    print(f"  Pixel Accuracy: {accuracy:.4f}")
    print(f"  Mean Dice Score: {dice_scores['mean_dice']:.4f}")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update('loss', 0.5)
    tracker.update('loss', 0.3)
    tracker.update('dice', 0.8)
    
    print("\nMetrics Tracker Test:")
    print(f"  Average loss: {tracker.get_average('loss'):.4f}")
    print(f"  Average dice: {tracker.get_average('dice'):.4f}")
    
    print("\nUtils module loaded successfully!")
