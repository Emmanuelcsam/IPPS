#!/usr/bin/env python3
# segdecnet_integration.py

"""
SegDecNet Integration
====================================
Segmentation-Decision Network for fiber defect detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List
import logging
from torch.utils.data import DataLoader
from torchvision import transforms

class SegmentationNetwork(nn.Module):
    """Segmentation sub-network of SegDecNet"""

    def __init__(self, in_channels=3, init_features=64):
        super().__init__()

        # Encoder
        self.enc1 = self._block(in_channels, init_features)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self._block(init_features, init_features * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self._block(init_features * 2, init_features * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = self._block(init_features * 4, init_features * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._block(init_features * 8, init_features * 16)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(init_features * 16, init_features * 8, 2, stride=2)
        self.dec4 = self._block(init_features * 16, init_features * 8)

        self.upconv3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, 2, stride=2)
        self.dec3 = self._block(init_features * 8, init_features * 4)

        self.upconv2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, 2, stride=2)
        self.dec2 = self._block(init_features * 4, init_features * 2)

        self.upconv1 = nn.ConvTranspose2d(init_features * 2, init_features, 2, stride=2)
        self.dec1 = self._block(init_features * 2, init_features)

        # Output
        self.out = nn.Conv2d(init_features, 1, 1)

    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Get feature maps for decision network
        features = {
            'enc1': enc1,
            'enc2': enc2,
            'enc3': enc3,
            'enc4': enc4,
            'dec1': dec1,
            'dec2': dec2,
            'dec3': dec3,
            'dec4': dec4
        }

        return torch.sigmoid(self.out(dec1)), features

class DecisionNetwork(nn.Module):
    """Decision sub-network of SegDecNet"""

    def __init__(self, num_features=8, hidden_dim=256):
        super().__init__()

        # Global average pooling will be applied to each feature map
        self.feature_dims = {
            'enc1': 64,
            'enc2': 128,
            'enc3': 256,
            'enc4': 512,
            'dec1': 64,
            'dec2': 128,
            'dec3': 256,
            'dec4': 512
        }

        total_features = sum(self.feature_dims.values())

        self.fc1 = nn.Linear(total_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, features):
        # Global average pooling on each feature map
        pooled_features = []
        for key in ['enc1', 'enc2', 'enc3', 'enc4', 'dec1', 'dec2', 'dec3', 'dec4']:
            feat = features[key]
            pooled = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
            pooled_features.append(pooled)

        # Concatenate all features
        x = torch.cat(pooled_features, dim=1)

        # Fully connected layers
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = torch.sigmoid(self.fc3(x))

        return x

class SegDecNet(nn.Module):
    """Complete SegDec-Net for defect detection"""

    def __init__(self):
        super().__init__()
        self.segmentation_net = SegmentationNetwork()
        self.decision_net = DecisionNetwork()

    def forward(self, x):
        seg_output, features = self.segmentation_net(x)
        decision = self.decision_net(features)
        return seg_output, decision

class FiberSegDecNet:
    """
    SegDecNet wrapper for fiber optic defect detection
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = SegDecNet().to(self.device)

        if model_path:
            self.load_model(model_path)

        self.model.eval()

        # Image normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded SegDecNet model from {model_path}")

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for SegDecNet"""
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to multiple of 16
        h, w = image.shape[:2]
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16

        if new_h != h or new_w != w:
            image_resized = cv2.resize(image, (new_w, new_h))
        else:
            image_resized = image

        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
        image_tensor = self.normalize(image_tensor).unsqueeze(0)

        return image_tensor.to(self.device), (h, w)

    def detect_defects(self, image: np.ndarray, zone_mask: Optional[np.ndarray] = None,
                      seg_threshold: float = 0.5, decision_threshold: float = 0.5) -> Tuple[np.ndarray, float, float]:
        """
        Detect defects using SegDecNet

        Args:
            image: Input image
            zone_mask: Optional zone mask
            seg_threshold: Threshold for segmentation output
            decision_threshold: Threshold for decision network

        Returns:
            Defect mask, segmentation confidence, decision score
        """
        # Preprocess
        input_tensor, original_size = self.preprocess_image(image)

        # Forward pass
        with torch.no_grad():
            seg_output, decision = self.model(input_tensor)

        # Process segmentation output
        seg_prob = seg_output.squeeze().cpu().numpy()

        # Resize to original size
        seg_prob_resized = cv2.resize(seg_prob, (original_size[1], original_size[0]))

        # Apply zone mask if provided
        if zone_mask is not None:
            seg_prob_resized = cv2.bitwise_and(seg_prob_resized, seg_prob_resized,
                                              mask=zone_mask.astype(np.uint8))

        # Get decision score
        decision_score = decision.item()

        # Threshold segmentation based on decision
        if decision_score > decision_threshold:
            # High confidence in defect presence
            defect_mask = (seg_prob_resized > seg_threshold).astype(np.uint8) * 255
        else:
            # Low confidence - raise threshold
            defect_mask = (seg_prob_resized > seg_threshold * 1.5).astype(np.uint8) * 255

        # Calculate segmentation confidence
        seg_confidence = np.mean(seg_prob_resized[defect_mask > 0]) if np.any(defect_mask) else 0

        return defect_mask, seg_confidence, decision_score

    def train_on_fiber_dataset(self, train_loader: DataLoader, val_loader: DataLoader,
                              epochs: int = 50, lr: float = 1e-4):
        """
        Train SegDecNet on fiber dataset

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        seg_criterion = nn.BCELoss()
        dec_criterion = nn.BCELoss()

        self.model.train()

        for epoch in range(epochs):
            train_seg_loss = 0
            train_dec_loss = 0

            for batch_idx, (images, seg_masks, has_defect) in enumerate(train_loader):
                images = images.to(self.device)
                seg_masks = seg_masks.to(self.device)
                has_defect = has_defect.float().to(self.device)

                optimizer.zero_grad()

                # Forward pass
                seg_output, decision = self.model(images)

                # Calculate losses
                seg_loss = seg_criterion(seg_output, seg_masks)
                dec_loss = dec_criterion(decision.squeeze(), has_defect)

                # Combined loss
                total_loss = seg_loss + 0.5 * dec_loss

                # Backward pass
                total_loss.backward()
                optimizer.step()

                train_seg_loss += seg_loss.item()
                train_dec_loss += dec_loss.item()

            # Validation
            self.model.eval()
            val_seg_loss = 0
            val_dec_loss = 0

            with torch.no_grad():
                for images, seg_masks, has_defect in val_loader:
                    images = images.to(self.device)
                    seg_masks = seg_masks.to(self.device)
                    has_defect = has_defect.float().to(self.device)

                    seg_output, decision = self.model(images)

                    seg_loss = seg_criterion(seg_output, seg_masks)
                    dec_loss = dec_criterion(decision.squeeze(), has_defect)

                    val_seg_loss += seg_loss.item()
                    val_dec_loss += dec_loss.item()

            self.model.train()

            logging.info(f"Epoch {epoch}: Train Seg Loss: {train_seg_loss/len(train_loader):.4f}, "
                        f"Train Dec Loss: {train_dec_loss/len(train_loader):.4f}, "
                        f"Val Seg Loss: {val_seg_loss/len(val_loader):.4f}, "
                        f"Val Dec Loss: {val_dec_loss/len(val_loader):.4f}")