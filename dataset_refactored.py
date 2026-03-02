"""
Refactored dataset module for brain tumor segmentation
Uses data processing classes for consistent processing
Ensures correlation between generated structures and ground truth
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from data_processing import BrainMRIProcessor, BrainStructureGenerator


class BrainTumorDataset(Dataset):
    """
    Dataset for brain tumor segmentation
    Generates synthetic data with proper correlation between images and masks
    """
    
    def __init__(self, 
                 num_samples: int = 1000,
                 image_size: Tuple[int, int] = (256, 256),
                 n_channels: int = 4,
                 n_classes: int = 5,
                 mode: str = 'train',
                 augment: bool = True,
                 normalize: bool = True,
                 seed: Optional[int] = None,
                 cache_dir: Optional[str] = "dataset_cache",
                 regenerate_cache: bool = False):
        """
        Args:
            num_samples: Number of samples to generate
            image_size: Size of images (H, W)
            n_channels: Number of input channels (MRI modalities)
            n_classes: Number of segmentation classes
            mode: 'train', 'val', or 'test'
            augment: Whether to apply augmentation
            normalize: Whether to normalize images
            seed: Random seed for reproducibility
            cache_dir: Directory to store or load cached dataset tensors
            regenerate_cache: Force regeneration even if a cache file exists
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.regenerate_cache = regenerate_cache
        
        # Set random seed for reproducibility
        self.seed = seed
        if seed is not None:
            mode_offset = {'train': 0, 'val': 1000, 'test': 2000}[mode]
            np.random.seed(seed + mode_offset)
            torch.manual_seed(seed + mode_offset)
        
        # Initialize data processor
        self.processor = BrainMRIProcessor(
            image_size=image_size,
            normalize=normalize,
            augment=(augment and mode == 'train'),
            n_channels=n_channels
        )
        
        # Initialize structure generator
        self.structure_generator = BrainStructureGenerator(
            image_size=image_size,
            n_channels=n_channels
        )
        
        # Pre-generate data or load from cache for consistency
        self.data_cache = []
        self.cache_path = self._resolve_cache_path()
        if not self._load_cached_data():
            self._generate_data()
            self._persist_cache()
    
    def _resolve_cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        seed_label = self.seed if self.seed is not None else "none"
        filename = (
            f"{self.mode}_samples{self.num_samples}_"
            f"{self.image_size[0]}x{self.image_size[1]}_"
            f"ch{self.n_channels}_seed{seed_label}.pt"
        )
        return self.cache_dir / filename
    
    def _load_cached_data(self) -> bool:
        if self.cache_path is None or self.regenerate_cache:
            return False
        try:
            if self.cache_path.exists() and self.cache_path.stat().st_size > 0:
                cached = torch.load(self.cache_path)
                if isinstance(cached, list) and len(cached) >= self.num_samples:
                    self.data_cache = cached[:self.num_samples]
                    return True
        except Exception as exc:
            print(f"Warning: failed to load dataset cache from {self.cache_path}: {exc}")
        self.data_cache = []
        return False
    
    def _persist_cache(self) -> None:
        if self.cache_path is None:
            return
        snapshots = []
        for sample in self.data_cache:
            structures = sample.get('structures', {})
            structures_cpu = {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in structures.items()
            }
            snapshots.append({
                'image': sample['image'].cpu(),
                'mask': sample['mask'].cpu(),
                'structures': structures_cpu
            })
        torch.save(snapshots, self.cache_path)
    
    def _generate_data(self):
        """Pre-generate all data samples"""
        self.data_cache = []
        for _ in range(self.num_samples):
            # Generate brain structures
            structures = self.structure_generator.generate_brain_structure()
            
            # Generate corresponding MRI images and ground truth mask
            images, mask = self.structure_generator.generate_from_structures(structures)
            
            structures_cpu = {
                key: value.clone().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in structures.items()
            }
            
            self.data_cache.append({
                'image': images.clone().cpu(),
                'mask': mask.clone().cpu(),
                'structures': structures_cpu  # Store for potential analysis
            })
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a data sample
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (image, mask)
        """
        # Get cached data
        sample = self.data_cache[idx]
        image = sample['image'].clone()
        mask = sample['mask'].clone()
        
        # Add batch dimension for processing
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        # Apply preprocessing (normalization, augmentation)
        image, mask = self.processor.preprocess(image, mask)
        
        # Remove batch dimension
        image = image.squeeze(0)
        mask = mask.squeeze(0)
        
        return image, mask
    
    def get_sample_with_structures(self, idx: int):
        """
        Get a sample with its underlying structures
        Useful for visualization and debugging
        """
        sample = self.data_cache[idx]
        image = sample['image'].clone()
        mask = sample['mask'].clone()
        structures = {
            key: value.clone()
            if isinstance(value, torch.Tensor) else value
            for key, value in sample['structures'].items()
        }
        
        # Apply preprocessing
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        image, mask = self.processor.preprocess(image, mask)
        image = image.squeeze(0)
        mask = mask.squeeze(0)
        
        return image, mask, structures


class RealBrainTumorDataset(Dataset):
    """
    Dataset for real brain tumor data
    Can be extended to load actual MRI data from files
    """
    
    def __init__(self,
                 data_path: str,
                 image_size: Tuple[int, int] = (256, 256),
                 n_channels: int = 4,
                 n_classes: int = 5,
                 mode: str = 'train',
                 augment: bool = True,
                 normalize: bool = True):
        """
        Args:
            data_path: Path to real data
            image_size: Size of images (H, W)
            n_channels: Number of input channels
            n_classes: Number of segmentation classes
            mode: 'train', 'val', or 'test'
            augment: Whether to apply augmentation
            normalize: Whether to normalize images
        """
        self.data_path = data_path
        self.image_size = image_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        
        # Initialize data processor
        self.processor = BrainMRIProcessor(
            image_size=image_size,
            normalize=normalize,
            augment=(augment and mode == 'train'),
            n_channels=n_channels
        )
        
        # In a real implementation, load file paths here
        self.samples = []
        
        # For now, generate synthetic data as placeholder
        self.structure_generator = BrainStructureGenerator(
            image_size=image_size,
            n_channels=n_channels
        )
        
        # Generate some synthetic samples as placeholder
        self.num_samples = 100
        for _ in range(self.num_samples):
            structures = self.structure_generator.generate_brain_structure()
            images, mask = self.structure_generator.generate_from_structures(structures)
            self.samples.append({'image': images, 'mask': mask})
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a data sample"""
        sample = self.samples[idx]
        image = sample['image'].clone()
        mask = sample['mask'].clone()
        
        # Add batch dimension for processing
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        # Apply preprocessing
        image, mask = self.processor.preprocess(image, mask)
        
        # Remove batch dimension
        image = image.squeeze(0)
        mask = mask.squeeze(0)
        
        return image, mask


def create_dataloaders(
        batch_size: int = 4,
        train_samples: int = 1000,
        val_samples: int = 200,
        test_samples: int = 200,
        image_size: Tuple[int, int] = (256, 256),
        n_channels: int = 4,
        n_classes: int = 5,
        num_workers: int = 0,
        seed: Optional[int] = 42,
        cache_dir: Optional[str] = "dataset_cache",
        regenerate_cache: bool = False):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        batch_size: Batch size for data loaders
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        image_size: Size of images
        n_channels: Number of input channels
        n_classes: Number of output classes
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        cache_dir: Directory used to cache generated datasets
        regenerate_cache: Force regeneration even if cache exists
    
    Returns:
        Dictionary with train, val, and test data loaders
    """
    # Create datasets
    train_dataset = BrainTumorDataset(
        num_samples=train_samples,
        image_size=image_size,
        n_channels=n_channels,
        n_classes=n_classes,
        mode='train',
        augment=True,
        normalize=True,
        seed=seed,
        cache_dir=cache_dir,
        regenerate_cache=regenerate_cache
    )
    
    val_dataset = BrainTumorDataset(
        num_samples=val_samples,
        image_size=image_size,
        n_channels=n_channels,
        n_classes=n_classes,
        mode='val',
        augment=False,
        normalize=True,
        seed=seed,
        cache_dir=cache_dir,
        regenerate_cache=regenerate_cache
    )
    
    test_dataset = BrainTumorDataset(
        num_samples=test_samples,
        image_size=image_size,
        n_channels=n_channels,
        n_classes=n_classes,
        mode='test',
        augment=False,
        normalize=True,
        seed=seed,
        cache_dir=cache_dir,
        regenerate_cache=regenerate_cache
    )
    
    # Create data loaders
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def visualize_sample(dataset: BrainTumorDataset, idx: int = 0):
    """
    Visualize a sample from the dataset
    
    Args:
        dataset: Dataset to visualize from
        idx: Sample index
    """
    import matplotlib.pyplot as plt
    
    # Get sample with structures
    if hasattr(dataset, 'get_sample_with_structures'):
        image, mask, structures = dataset.get_sample_with_structures(idx)
    else:
        image, mask = dataset[idx]
        structures = None
    
    # Denormalize image for visualization
    if dataset.processor.normalize:
        image = dataset.processor.denormalize(image.unsqueeze(0)).squeeze(0)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Show different MRI modalities
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
    for i in range(min(4, image.shape[0])):
        ax = axes[i // 2, i % 2]
        ax.imshow(image[i].cpu().numpy(), cmap='gray')
        ax.set_title(f'{modality_names[i]} Modality')
        ax.axis('off')
    
    # Show ground truth mask
    axes[0, 2].imshow(mask.cpu().numpy(), cmap='tab20')
    axes[0, 2].set_title('Ground Truth Mask')
    axes[0, 2].axis('off')
    
    # Show structures if available
    if structures is not None:
        combined_structure = torch.zeros_like(mask).float()
        if 'necrotic' in structures:
            combined_structure[structures['necrotic'].bool()] = 1
        if 'edema' in structures:
            combined_structure[structures['edema'].bool()] = 2
        if 'enhancing' in structures:
            combined_structure[structures['enhancing'].bool()] = 3
        if 'healthy' in structures:
            combined_structure[structures['healthy'].bool()] = 4
        
        axes[1, 2].imshow(combined_structure.cpu().numpy(), cmap='tab20')
        axes[1, 2].set_title('Generated Structures')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Testing refactored dataset module...")
    print("-" * 60)
    
    # Create a small dataset for testing
    dataset = BrainTumorDataset(
        num_samples=10,
        image_size=(256, 256),
        n_channels=4,
        n_classes=5,
        mode='train',
        seed=42
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Unique mask values: {mask.unique().tolist()}")
    
    # Test data loaders
    print("\nTesting data loaders...")
    loaders = create_dataloaders(
        batch_size=2,
        train_samples=10,
        val_samples=5,
        test_samples=5,
        seed=42
    )
    
    # Test train loader
    train_batch = next(iter(loaders['train']))
    print(f"Train batch - Images: {train_batch[0].shape}, Masks: {train_batch[1].shape}")
    
    print("\nDataset module test completed successfully!")
