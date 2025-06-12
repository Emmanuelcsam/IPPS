#!/usr/bin/env python3
# padim_specific.py

"""
Specific PaDiM Implementation
===========================================
Based on xiahaifeng1995's implementation with fiber-specific adaptations
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import cv2
from typing import List, Tuple, Optional
import logging

class FiberPaDiM:
    """
    PaDiM specifically adapted for fiber optic defect detection
    """
    
    def __init__(self, backbone='resnet18', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load backbone
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.t_d = 448  # Total dimension from layers 1,2,3
            self.d = 100    # Reduced dimension
        elif backbone == 'wide_resnet50_2':
            self.model = models.wide_resnet50_2(pretrained=True)
            self.t_d = 1792
            self.d = 550
        
        self.model.to(self.device)
        self.model.eval()
        
        # Remove last layers
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-2]))
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Training data storage
        self.training_features = []
        self.patch_lib = []
        self.N = 0  # Number of training samples
        
    def extract_features(self, x):
        """Extract multi-scale features from ResNet"""
        features = []
        
        # Hook functions to extract intermediate features
        outputs = []
        def hook(module, input, output):
            outputs.append(output)
        
        # Register hooks for layers 1, 2, 3
        if 'resnet' in str(type(self.model[0])):
            self.model[4].register_forward_hook(hook)  # layer1
            self.model[5].register_forward_hook(hook)  # layer2
            self.model[6].register_forward_hook(hook)  # layer3
        
        with torch.no_grad():
            _ = self.model(x)
        
        # Collect features
        for k, output in enumerate(outputs):
            m = torch.nn.AvgPool2d(3, 1, 1)
            output = m(output)
            features.append(output)
        
        return features
    
    def fit(self, train_images: List[np.ndarray]):
        """
        Fit PaDiM on normal (defect-free) fiber images
        
        Args:
            train_images: List of normal fiber images
        """
        train_outputs = []
        
        for img in train_images:
            # Preprocess image
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Convert to PIL and apply transforms
            from PIL import Image
            img_pil = Image.fromarray(img)
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            
            # Extract features
            features = self.extract_features(img_tensor)
            
            # Concatenate features
            embeddings = []
            for feature in features:
                m = torch.nn.AvgPool2d(3, 1, 1)
                embeddings.append(m(feature))
            
            # Reshape and concatenate
            embedding_vectors = []
            for embedding in embeddings:
                B, C, H, W = embedding.size()
                embedding = embedding.permute(0, 2, 3, 1).reshape(B, H*W, C)
                embedding_vectors.append(embedding)
            
            embedding_concat = torch.cat(embedding_vectors, dim=2)
            train_outputs.append(embedding_concat.cpu().numpy())
        
        # Concatenate all training outputs
        train_outputs = np.concatenate(train_outputs, axis=0)
        
        # Random projection to reduce dimensionality
        self.N = train_outputs.shape[0]
        
        # Create random projection matrix
        np.random.seed(42)
        self.R = np.random.randn(self.t_d, self.d)
        
        # Project features
        train_outputs_reduced = np.dot(train_outputs.reshape(-1, self.t_d), self.R)
        train_outputs_reduced = train_outputs_reduced.reshape(self.N, -1, self.d)
        
        # Calculate multivariate Gaussian parameters
        self.patch_means = np.mean(train_outputs_reduced, axis=0)
        
        # Calculate covariance
        C = np.zeros((train_outputs_reduced.shape[1], self.d, self.d))
        for i in range(train_outputs_reduced.shape[1]):
            patch_features = train_outputs_reduced[:, i, :]
            mean = self.patch_means[i]
            C[i] = np.cov(patch_features.T) + 0.01 * np.eye(self.d)
        
        self.C = C
        self.C_inv = np.linalg.inv(C)
        
        logging.info(f"PaDiM fitted on {len(train_images)} normal images")
    
    def predict(self, test_image: np.ndarray, zone_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Predict anomaly map for test image
        
        Args:
            test_image: Test image
            zone_mask: Optional zone mask
            
        Returns:
            Anomaly map and anomaly score
        """
        # Preprocess
        if len(test_image.shape) == 2:
            test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)
        
        # Apply zone mask if provided
        if zone_mask is not None:
            mask_3ch = cv2.cvtColor(zone_mask, cv2.COLOR_GRAY2RGB)
            test_image = cv2.bitwise_and(test_image, mask_3ch)
        
        # Transform
        from PIL import Image
        img_pil = Image.fromarray(test_image)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        # Extract features
        features = self.extract_features(img_tensor)
        
        # Process features
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        
        # Concatenate
        embedding_vectors = []
        for embedding in embeddings:
            B, C, H, W = embedding.size()
            embedding_vectors.append(embedding.permute(0, 2, 3, 1).reshape(B, H*W, C))
        
        embedding_concat = torch.cat(embedding_vectors, dim=2).cpu().numpy()
        
        # Project
        embedding_reduced = np.dot(embedding_concat.reshape(-1, self.t_d), self.R)
        embedding_reduced = embedding_reduced.reshape(1, -1, self.d)
        
        # Calculate Mahalanobis distance
        H = int(np.sqrt(embedding_reduced.shape[1]))
        W = H
        
        dist_list = []
        for i in range(H * W):
            mean = self.patch_means[i]
            conv_inv = self.C_inv[i]
            dist = mahalanobis(embedding_reduced[0, i], mean, conv_inv)
            dist_list.append(dist)
        
        # Reshape to spatial dimensions
        dist_map = np.array(dist_list).reshape(H, W)
        
        # Upsample to original size
        dist_map_resized = cv2.resize(dist_map, (test_image.shape[1], test_image.shape[0]))
        
        # Apply Gaussian smoothing
        anomaly_map = gaussian_filter(dist_map_resized, sigma=4)
        
        # Normalize
        anomaly_map = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map))
        
        # Apply zone mask to output
        if zone_mask is not None:
            anomaly_map = cv2.bitwise_and(anomaly_map, anomaly_map, mask=zone_mask)
        
        # Calculate score
        anomaly_score = np.mean(anomaly_map)
        
        return anomaly_map, anomaly_score