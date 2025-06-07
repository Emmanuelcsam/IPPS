#!/usr/bin/env python3
# unet_segmentation.py

"""
D-Scope Blink: U-Net Segmentation Module
=======================================
Implements U-Net architecture for fiber optic defect segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
import logging

class UNetBlock(nn.Module):
    """Basic U-Net convolutional block"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    """U-Net architecture for defect segmentation"""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, 
                 init_features: int = 32):
        super().__init__()
        
        features = init_features
        
        # Encoder
        self.encoder1 = UNetBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = UNetBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = UNetBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = UNetBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = UNetBlock(features * 8, features * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, stride=2)
        self.decoder4 = UNetBlock(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, stride=2)
        self.decoder3 = UNetBlock(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, stride=2)
        self.decoder2 = UNetBlock(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, stride=2)
        self.decoder1 = UNetBlock(features * 2, features)
        
        # Final convolution
        self.final_conv = nn.Conv2d(features, out_channels, 1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.final_conv(dec1))

class UNetDefectDetector:
    """Wrapper for U-Net based defect detection"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = UNet(in_channels=1, out_channels=1)
        self.model.to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
        
    def load_model(self, model_path: str):
        """Load trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded U-Net model from {model_path}")
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for U-Net input"""
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size (divisible by 16 for U-Net)
        h, w = image.shape
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16
        
        if new_h != h or new_w != w:
            image = cv2.resize(image, (new_w, new_h))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        
        return tensor.to(self.device), (h, w)
        
    def detect_defects(self, image: np.ndarray, zone_mask: Optional[np.ndarray] = None,
                      threshold: float = 0.5) -> np.ndarray:
        """
        Detect defects using U-Net
        
        Args:
            image: Input grayscale image
            zone_mask: Optional zone mask
            threshold: Confidence threshold
            
        Returns:
            Binary defect mask
        """
        # Preprocess
        input_tensor, original_size = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Convert to numpy
        prob_map = output.squeeze().cpu().numpy()
        
        # Resize back to original size
        prob_map = cv2.resize(prob_map, (original_size[1], original_size[0]))
        
        # Threshold
        defect_mask = (prob_map > threshold).astype(np.uint8) * 255
        
        # Apply zone mask if provided
        if zone_mask is not None:
            defect_mask = cv2.bitwise_and(defect_mask, defect_mask, mask=zone_mask)
        
        return defect_mask

def integrate_unet_detection(image: np.ndarray, zone_mask: np.ndarray,
                           unet_model: Optional[UNetDefectDetector] = None) -> np.ndarray:
    """
    Integrate U-Net detection into main pipeline
    
    Args:
        image: Input image
        zone_mask: Zone mask
        unet_model: Trained U-Net model
        
    Returns:
        Binary defect mask
    """
    if unet_model is None:
        return np.zeros_like(zone_mask)
    
    try:
        defect_mask = unet_model.detect_defects(image, zone_mask)
        logging.debug("U-Net defect detection complete")
        return defect_mask
    except Exception as e:
        logging.error(f"U-Net detection failed: {e}")
        return np.zeros_like(zone_mask)