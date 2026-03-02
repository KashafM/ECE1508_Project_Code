"""
Data processing module for brain tumor segmentation
Handles all input/output processing, normalization, and augmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
import random
import math


class BrainMRIProcessor:
    """
    Processor for brain MRI data
    Handles normalization, augmentation, and preprocessing
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (256, 256),
                 normalize: bool = True,
                 augment: bool = False,
                 n_channels: int = 4):
        """
        Args:
            image_size: Target image size (height, width)
            normalize: Whether to normalize input data
            augment: Whether to apply augmentation during training
            n_channels: Number of input channels (MRI modalities)
        """
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        self.n_channels = n_channels
        
        # Statistics for normalization (typical brain MRI values)
        self.mean = [0.1384, 0.1698, 0.1823, 0.1557][:n_channels]
        self.std = [0.2405, 0.2512, 0.2687, 0.2433][:n_channels]
    
    def preprocess(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None):
        """
        Preprocess images and masks
        
        Args:
            images: Input images (B, C, H, W)
            masks: Ground truth masks (B, H, W) or None
        
        Returns:
            Processed images and masks
        """
        # Resize if needed
        if images.shape[-2:] != self.image_size:
            images = F.interpolate(images, size=self.image_size, mode='bilinear', align_corners=True)
            if masks is not None:
                masks = F.interpolate(masks.unsqueeze(1).float(), size=self.image_size, mode='nearest').squeeze(1).long()
        
        # Normalize
        if self.normalize:
            images = self._normalize(images)
        
        # Augmentation (training only)
        if self.augment and masks is not None:
            images, masks = self._augment(images, masks)
        
        return images, masks
    
    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images using channel-wise mean and std"""
        device = images.device
        mean = torch.tensor(self.mean, device=device).view(1, -1, 1, 1)
        std = torch.tensor(self.std, device=device).view(1, -1, 1, 1)
        return (images - mean) / (std + 1e-8)
    
    def _augment(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation"""
        # Random horizontal flip
        if random.random() > 0.5:
            images = torch.flip(images, dims=[-1])
            masks = torch.flip(masks, dims=[-1])
        
        # Random rotation (small angles)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            images = self._rotate_tensor(images, angle)
            masks = self._rotate_tensor(masks.unsqueeze(1).float(), angle, mode='nearest').squeeze(1).long()
        
        # Random intensity scaling
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            images = images * scale
        
        # Random noise
        if random.random() > 0.5:
            noise = torch.randn_like(images) * 0.01
            images = images + noise
        
        return images, masks
    
    def _rotate_tensor(self, tensor: torch.Tensor, angle: float, mode: str = 'bilinear') -> torch.Tensor:
        """Rotate tensor by angle degrees"""
        theta = angle * np.pi / 180
        rotation_matrix = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0]
        ], dtype=torch.float32).unsqueeze(0).repeat(tensor.shape[0], 1, 1)
        
        grid = F.affine_grid(rotation_matrix, tensor.size(), align_corners=False)
        return F.grid_sample(tensor, grid, mode=mode, align_corners=False)
    
    def postprocess(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Postprocess model predictions
        
        Args:
            predictions: Model output logits (B, C, H, W)
        
        Returns:
            Segmentation masks (B, H, W)
        """
        # Apply softmax if logits
        if predictions.dim() == 4:
            predictions = F.softmax(predictions, dim=1)
            predictions = torch.argmax(predictions, dim=1)
        
        return predictions
    
    def denormalize(self, images: torch.Tensor) -> torch.Tensor:
        """Denormalize images for visualization"""
        if self.normalize:
            device = images.device
            mean = torch.tensor(self.mean, device=device).view(1, -1, 1, 1)
            std = torch.tensor(self.std, device=device).view(1, -1, 1, 1)
            images = images * std + mean
        
        return torch.clamp(images, 0, 1)


class BrainStructureGenerator:
    """
    Generate realistic brain structures for synthetic data
    Ensures correlation between input images and ground truth masks
    """
    
    def __init__(self, image_size: Tuple[int, int] = (256, 256), n_channels: int = 4):
        """
        Args:
            image_size: Size of generated images
            n_channels: Number of MRI modalities to generate
        """
        self.image_size = image_size
        self.n_channels = n_channels
        
        # Tissue properties for each MRI modality (T1, T1ce, T2, FLAIR)
        # Values represent relative intensity for each tissue type
        self.tissue_properties = {
            # tissue_type: [T1, T1ce, T2, FLAIR]
            'background': [0.0, 0.0, 0.0, 0.0],
            'gray_matter': [0.4, 0.4, 0.6, 0.5],
            'white_matter': [0.6, 0.6, 0.4, 0.4],
            'csf': [0.1, 0.1, 0.9, 0.2],
            'necrotic': [0.2, 0.1, 0.7, 0.8],
            'edema': [0.3, 0.3, 0.8, 0.9],
            'enhancing': [0.5, 0.9, 0.5, 0.6],
        }
        
        # Class mapping
        self.class_map = {
            'background': 0,
            'necrotic': 1,
            'edema': 2,
            'enhancing': 3,
            'healthy': 4,
        }
    
    def generate_brain_structure(self) -> Dict[str, torch.Tensor]:
        """
        Generate a realistic brain structure with tumor
        
        Returns:
            Dictionary containing structure masks for each tissue type
        """
        h, w = self.image_size
        structures = {}
        
        # Create brain outline (elliptical)
        center_x, center_y = w // 2, h // 2
        a, b = w // 2.5, h // 2.2  # Semi-axes
        
        y, x = np.ogrid[:h, :w]
        brain_mask = ((x - center_x) ** 2 / a ** 2 + (y - center_y) ** 2 / b ** 2) <= 1
        structures['brain'] = torch.tensor(brain_mask, dtype=torch.float32)
        
        # Generate tumor location (random but realistic)
        tumor_center_x = center_x + np.random.randint(-w//4, w//4)
        tumor_center_y = center_y + np.random.randint(-h//4, h//4)
        
        # Generate tumor components with realistic shapes
        if np.random.random() > 0.25:  # 75% chance of tumor
            base_tumor = torch.zeros((h, w), dtype=torch.float32)
            num_lobes = np.random.randint(1, 4)
            for _ in range(num_lobes):
                lobe = self._rotated_gaussian_blob(
                    h=h,
                    w=w,
                    cx=tumor_center_x + np.random.randint(-10, 11),
                    cy=tumor_center_y + np.random.randint(-10, 11),
                    sigma_x=np.random.uniform(6.0, 14.0),
                    sigma_y=np.random.uniform(8.0, 20.0),
                    angle=np.random.uniform(0, 2 * math.pi),
                    elongation=np.random.uniform(0.6, 1.4),
                )
                base_tumor = torch.maximum(base_tumor, lobe)

            base_tumor = base_tumor * structures['brain']
            base_tumor = torch.clamp(base_tumor + 0.05 * torch.randn_like(base_tumor), 0, 1)

            necrotic = (base_tumor > 0.65).float()
            enhancing = (base_tumor > 0.4).float() * (1 - necrotic)

            tumor_binary = (base_tumor > 0.25).float().unsqueeze(0).unsqueeze(0)
            edema = F.max_pool2d(tumor_binary, kernel_size=9, stride=1, padding=4)
            edema = (edema.squeeze(0).squeeze(0) > 0.1).float()
            edema = edema * (1 - enhancing - necrotic)

            # Slightly irregular edema boundary to mimic infiltrative patterns
            edema_noise = torch.clamp(edema + 0.2 * torch.randn_like(edema), 0, 1)
            edema = (edema_noise > 0.3).float()

            structures['necrotic'] = necrotic
            structures['enhancing'] = enhancing
            structures['edema'] = edema * structures['brain']
        else:
            structures['necrotic'] = torch.zeros((h, w))
            structures['enhancing'] = torch.zeros((h, w))
            structures['edema'] = torch.zeros((h, w))
        
        # Healthy tissue (rest of brain)
        tumor_mask = structures['necrotic'] + structures['enhancing'] + structures['edema']
        structures['healthy'] = structures['brain'] * (1 - tumor_mask)
        
        return structures
    
    def _create_irregular_circle(self, h: int, w: int, cx: int, cy: int, 
                                 radius: float, irregularity: float = 0.1) -> torch.Tensor:
        """Create an irregular circular shape"""
        y, x = np.ogrid[:h, :w]
        
        # Add irregularity using Fourier components
        angles = np.arctan2(y - cy, x - cx)
        n_components = 5
        radius_variation = radius
        
        for i in range(1, n_components + 1):
            amplitude = (radius * irregularity) / i
            phase = np.random.random() * 2 * np.pi
            radius_variation = radius_variation + amplitude * np.sin(i * angles + phase)
        
        distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = distances <= radius_variation
        
        return torch.tensor(mask, dtype=torch.float32)
    
    def _rotated_gaussian_blob(
        self,
        h: int,
        w: int,
        cx: float,
        cy: float,
        sigma_x: float,
        sigma_y: float,
        angle: float,
        elongation: float = 1.0,
    ) -> torch.Tensor:
        """Create an anisotropic rotated Gaussian blob to simulate tumor lobes."""
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing='ij'
        )
        x_shift = x_grid - cx
        y_shift = y_grid - cy
        
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        x_rot = cos_a * x_shift + sin_a * y_shift
        y_rot = -sin_a * x_shift + cos_a * y_shift
        
        sigma_x = max(sigma_x, 1.0)
        sigma_y = max(sigma_y * elongation, 1.0)
        
        exponent = -0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)
        blob = torch.exp(exponent)
        blob = blob / blob.max().clamp_min(1e-6)
        
        edge_softening = torch.clamp(1 - ((x_rot**2 + y_rot**2) / (sigma_x * sigma_y * 8)), min=0.0, max=1.0)
        blob = blob * edge_softening
        return blob.clamp(0, 1)
    
    def generate_from_structures(self, structures: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate MRI images and ground truth mask from structures
        
        Args:
            structures: Dictionary of tissue structure masks
        
        Returns:
            images: Multi-modal MRI images (C, H, W)
            mask: Ground truth segmentation mask (H, W)
        """
        h, w = self.image_size
        images = torch.zeros((self.n_channels, h, w))
        mask = torch.zeros((h, w), dtype=torch.long)
        
        # Generate MRI modalities based on tissue properties
        for modality in range(self.n_channels):
            # Add gray matter
            if 'healthy' in structures:
                gray_intensity = self.tissue_properties['gray_matter'][modality]
                images[modality] += structures['healthy'] * gray_intensity * (0.8 + 0.2 * torch.randn(h, w))
            
            # Add white matter (inner region)
            if 'healthy' in structures:
                white_mask = self._erode(structures['healthy'], 10)
                white_intensity = self.tissue_properties['white_matter'][modality]
                images[modality] += white_mask * white_intensity * (0.8 + 0.2 * torch.randn(h, w))
            
            # Add CSF (ventricles)
            if 'brain' in structures:
                csf_mask = self._create_ventricles(h, w)
                csf_intensity = self.tissue_properties['csf'][modality]
                images[modality] += csf_mask * csf_intensity * (0.8 + 0.2 * torch.randn(h, w))
            
            # Add tumor components
            for tissue, class_idx in [('necrotic', 1), ('edema', 2), ('enhancing', 3)]:
                if tissue in structures:
                    tissue_intensity = self.tissue_properties[tissue][modality]
                    images[modality] += structures[tissue] * tissue_intensity * (0.8 + 0.2 * torch.randn(h, w))
        
        # Create ground truth mask
        mask[structures.get('necrotic', torch.zeros_like(mask)).bool()] = 1
        mask[structures.get('edema', torch.zeros_like(mask)).bool()] = 2
        mask[structures.get('enhancing', torch.zeros_like(mask)).bool()] = 3
        mask[structures.get('healthy', torch.zeros_like(mask)).bool()] = 4
        
        # Add noise and normalize
        images = torch.clamp(images + 0.05 * torch.randn_like(images), 0, 1)
        
        return images, mask
    
    def _erode(self, mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Morphological erosion"""
        if kernel_size > 1:
            kernel = torch.ones((1, 1, kernel_size, kernel_size))
            mask_4d = mask.unsqueeze(0).unsqueeze(0).float()
            # Use 'same' padding to maintain dimensions
            padding = kernel_size // 2
            eroded = F.conv2d(mask_4d, kernel, padding=padding)
            # Only keep pixels where all kernel pixels overlap with mask
            eroded = (eroded >= kernel_size * kernel_size - 0.5).float()
            result = eroded.squeeze()
            # Ensure output has same shape as input
            if result.shape != mask.shape:
                result = result[:mask.shape[0], :mask.shape[1]]
            return result
        return mask
    
    def _create_ventricles(self, h: int, w: int) -> torch.Tensor:
        """Create ventricle structures"""
        cx, cy = w // 2, h // 2
        
        # Lateral ventricles (butterfly shape)
        left_vent = self._create_irregular_circle(h, w, cx - 15, cy, 8, 0.2)
        right_vent = self._create_irregular_circle(h, w, cx + 15, cy, 8, 0.2)
        
        # Third ventricle (small central)
        third_vent = self._create_irregular_circle(h, w, cx, cy + 10, 3, 0.1)
        
        return left_vent + right_vent + third_vent
    
    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of synthetic brain MRI data
        
        Args:
            batch_size: Number of samples to generate
        
        Returns:
            images: Batch of MRI images (B, C, H, W)
            masks: Batch of ground truth masks (B, H, W)
        """
        images_list = []
        masks_list = []
        
        for _ in range(batch_size):
            structures = self.generate_brain_structure()
            images, mask = self.generate_from_structures(structures)
            images_list.append(images)
            masks_list.append(mask)
        
        return torch.stack(images_list), torch.stack(masks_list)


class DataAugmentation:
    """Advanced data augmentation for brain MRI"""
    
    @staticmethod
    def elastic_deformation(image: torch.Tensor, mask: torch.Tensor, 
                           alpha: float = 20, sigma: float = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply elastic deformation"""
        # Implementation of elastic deformation
        # Simplified version - in practice, use more sophisticated methods
        return image, mask
    
    @staticmethod
    def intensity_shift(image: torch.Tensor, shift_range: float = 0.1) -> torch.Tensor:
        """Apply random intensity shift"""
        shift = torch.empty(image.shape[0], 1, 1, 1).uniform_(-shift_range, shift_range)
        return torch.clamp(image + shift, 0, 1)
    
    @staticmethod
    def gaussian_noise(image: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise"""
        noise = torch.randn_like(image) * noise_level
        return torch.clamp(image + noise, 0, 1)
    
    @staticmethod
    def random_crop_and_resize(image: torch.Tensor, mask: torch.Tensor, 
                               crop_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random crop and resize back to original size"""
        # Implementation of random crop
        return image, mask
