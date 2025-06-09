#!/usr/bin/env python3
# padim_integration.py

"""
PaDiM Integration Module
======================================
Integrates PaDiM (Patch Distribution Modeling) for anomaly detection
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import Optional, Tuple, List
import cv2
import logging
from pathlib import Path

class PaDiMDetector:
    """
    PaDiM-based anomaly detector for fiber optic defects
    """
    
    def __init__(self, backbone: str = 'resnet18', device: str = 'cpu'):
        self.device = torch.device(device)
        self.backbone = self._load_backbone(backbone)
        self.backbone.eval()
        self.backbone.to(self.device)
        
        # PaDiM parameters
        self.image_size = 224
        self.patch_size = 16
        self.n_neighbors = 9
        
        # Memory bank for normal patches
        self.memory_bank = None
        self.fitted = False
        
    def _load_backbone(self, backbone_name: str):
        """Load pre-trained backbone network"""
        import torchvision.models as models
        
        if backbone_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            # Remove final layers
            self.feature_layers = ['layer1', 'layer2', 'layer3']
            return torch.nn.Sequential(*list(model.children())[:-2])
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    def extract_features(self, image: np.ndarray) -> List[torch.Tensor]:
        """Extract multi-scale features from image"""
        # Preprocess image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Extract features at multiple scales
        features = []
        with torch.no_grad():
            x = img_tensor
            for i, layer in enumerate(self.backbone):
                x = layer(x)
                if i in [4, 5, 6]:  # Specific layers for ResNet18
                    features.append(x)
        
        return features
    
    def fit(self, normal_images: List[np.ndarray]):
        """Fit the model on normal (defect-free) images"""
        logging.info(f"Fitting PaDiM on {len(normal_images)} normal images")
        
        all_features = []
        for img in normal_images:
            features = self.extract_features(img)
            # Concatenate multi-scale features
            combined = []
            for feat in features:
                # Resize to common size
                feat_resized = F.interpolate(feat, size=(28, 28), mode='bilinear')
                combined.append(feat_resized)
            
            combined_features = torch.cat(combined, dim=1)
            all_features.append(combined_features)
        
        # Create memory bank
        self.memory_bank = torch.cat(all_features, dim=0)
        self.fitted = True
        logging.info("PaDiM fitting complete")
    
    def predict_anomaly(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict anomaly map for input image
        
        Returns:
            anomaly_map: Pixel-wise anomaly scores
            anomaly_score: Image-level anomaly score
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Extract features
        features = self.extract_features(image)
        combined = []
        for feat in features:
            feat_resized = F.interpolate(feat, size=(28, 28), mode='bilinear')
            combined.append(feat_resized)
        
        test_features = torch.cat(combined, dim=1)
        
        # Compute distances to memory bank
        B, C, H, W = test_features.shape
        test_features_flat = test_features.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Euclidean distance to nearest neighbors
        distances = torch.cdist(test_features_flat, self.memory_bank.reshape(-1, C))
        min_distances, _ = torch.topk(distances, k=self.n_neighbors, dim=1, largest=False)
        anomaly_scores = torch.mean(min_distances, dim=1)
        
        # Reshape to spatial dimensions
        anomaly_map = anomaly_scores.reshape(H, W).cpu().numpy()
        
        # Resize to original image size
        anomaly_map_resized = cv2.resize(anomaly_map, (image.shape[1], image.shape[0]))
        
        # Normalize scores
        anomaly_map_normalized = (anomaly_map_resized - np.min(anomaly_map_resized)) / \
                                (np.max(anomaly_map_resized) - np.min(anomaly_map_resized) + 1e-8)
        
        anomaly_score = np.mean(anomaly_map_normalized)
        
        return anomaly_map_normalized, anomaly_score

def integrate_padim_detection(image: np.ndarray, zone_mask: np.ndarray, 
                            padim_model: Optional[PaDiMDetector] = None,
                            threshold: float = 0.5) -> np.ndarray:
    """
    Integrate PaDiM detection into the main pipeline
    
    Args:
        image: Input image
        zone_mask: Binary mask for the zone
        padim_model: Trained PaDiM model
        threshold: Anomaly threshold
        
    Returns:
        Binary defect mask
    """
    if padim_model is None or not padim_model.fitted:
        return np.zeros_like(zone_mask)
    
    # Apply zone mask
    masked_image = cv2.bitwise_and(image, image, mask=zone_mask)
    
    # Get anomaly map
    anomaly_map, score = padim_model.predict_anomaly(masked_image)
    
    # Threshold to get binary mask
    defect_mask = (anomaly_map > threshold).astype(np.uint8) * 255
    
    # Apply zone mask again
    defect_mask = cv2.bitwise_and(defect_mask, defect_mask, mask=zone_mask)
    
    logging.debug(f"PaDiM detection: anomaly score = {score:.3f}")
    
    return defect_mask