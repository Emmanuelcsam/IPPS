#!/usr/bin/env python3
# scratch_dataset_handler.py

"""
Scratch Dataset Handler
=====================================
Handles integration of external scratch dataset for validation and augmentation.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any, Tuple, Optional
import random

class ScratchDatasetHandler:
    """
    Manages scratch dataset for validation and augmentation.
    """

    def __init__(self, dataset_path: str):
        """
        Initialize the dataset handler.

        Args:
            dataset_path: Path to the scratch dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.scratch_samples = []
        self.scratch_features = []

        if self.dataset_path.exists():
            self._load_dataset()
        else:
            logging.warning(f"Scratch dataset path not found: {dataset_path}")

    def _load_dataset(self):
        """Load scratch samples from the dataset."""
        scratch_files = sorted(self.dataset_path.glob("*_s.bmp"))

        for file_path in scratch_files:
            try:
                img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.scratch_samples.append({
                        'image': img,
                        'path': file_path,
                        'id': file_path.stem
                    })

                    # Extract features for matching
                    features = self._extract_scratch_features(img)
                    self.scratch_features.append(features)

            except Exception as e:
                logging.warning(f"Failed to load scratch sample {file_path}: {e}")

        logging.info(f"Loaded {len(self.scratch_samples)} scratch samples from dataset")

    def _extract_scratch_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract features from a scratch image for matching."""
        features = {}

        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size

        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        features['line_count'] = len(lines) if lines is not None else 0

        # Texture features using Gabor filters
        gabor_features = []
        for theta in np.arange(0, np.pi, np.pi / 4):
            kernel = cv2.getGaborKernel((21, 21), 4, theta, 10, 0.5, 0)
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            gabor_features.append(np.mean(np.abs(filtered)))
        features['gabor'] = gabor_features

        # Morphological features
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        features['blackhat_mean'] = np.mean(blackhat)

        return features

    def _compute_similarity(self, f1: Dict[str, Any], f2: Dict[str, Any]) -> float:
        """
        Compute a simple similarity score between two feature dicts.
        Uses normalized differences of feature values (edge_density, line_count, gabor, blackhat_mean).
        """
        # Example: Euclidean distance in feature space, then convert to similarity (1 / (1 + distance))
        vals1 = [
            f1.get("edge_density", 0),
            f1.get("line_count", 0),
            f1.get("blackhat_mean", 0),
        ] + f1.get("gabor", [])
        vals2 = [
            f2.get("edge_density", 0),
            f2.get("line_count", 0),
            f2.get("blackhat_mean", 0),
        ] + f2.get("gabor", [])
        # Make sure same length
        min_len = min(len(vals1), len(vals2))
        a = np.array(vals1[:min_len], dtype=np.float32)
        b = np.array(vals2[:min_len], dtype=np.float32)
        dist = np.linalg.norm(a - b)
        return 1.0 / (1.0 + dist)

    def augment_scratch_detection(self, image: np.ndarray, threshold: float = 0.7) -> np.ndarray:
        """
        Augment scratch detection using the dataset.

        Args:
            image: Input image to analyze
            threshold: Similarity threshold for matching

        Returns:
            Probability map of scratch locations
        """
        if not self.scratch_samples:
            return np.zeros_like(image, dtype=np.float32)

        h, w = image.shape[:2]
        scratch_probability = np.zeros((h, w), dtype=np.float32)

        # Extract features from input image
        input_features = self._extract_scratch_features(image)

        # Find similar scratches in dataset
        similar_scratches = []
        for i, features in enumerate(self.scratch_features):
            similarity = self._compute_similarity(input_features, features)
            if similarity > threshold:
                similar_scratches.append((i, similarity))

        if similar_scratches:
            # Use template matching with similar scratches
            for idx, similarity in similar_scratches[:5]:  # Use top 5 matches
                scratch_img = self.scratch_samples[idx]['image']

                # Resize if needed
                if scratch_img.shape != image.shape:
                    scratch_img_resized = cv2.resize(scratch_img, (w, h), interpolation=cv2.INTER_NEAREST)
                    if image.ndim == 3 and scratch_img_resized.ndim == 2:
                        scratch_img_resized = cv2.cvtColor(scratch_img_resized, cv2.COLOR_GRAY2BGR)
                    scratch_img = scratch_img_resized

                # Apply template matching
                result = cv2.matchTemplate(image, scratch_img, cv2.TM_CCOEFF_NORMED)

                # Add weighted result to probability map
                scratch_probability += result * similarity * 0.5

        # Normalize
        if np.max(scratch_probability) > 0:
            scratch_probability = scratch_probability / np.max(scratch_probability)

        return scratch_probability

    def validate_detection(self, detected_scratches: List[Dict], image: np.ndarray) -> List[Dict]:
        """
        Validate detected scratches against the dataset.

        Args:
            detected_scratches: List of detected scratch dictionaries
            image: Original image

        Returns:
            Updated list with validation scores
        """
        if not self.scratch_samples:
            return detected_scratches

        for scratch in detected_scratches:
            # Extract region around scratch
            bbox = scratch.get('bbox_x_px', 0), scratch.get('bbox_y_px', 0), \
                   scratch.get('bbox_w_px', 10), scratch.get('bbox_h_px', 10)

            x, y, w, h = bbox
            scratch_region = image[max(0, y):min(y+h, image.shape[0]),
                                  max(0, x):min(x+w, image.shape[1])]

            if scratch_region.size > 0:
                # Compare with dataset
                scratch_features = self._extract_scratch_features(scratch_region)

                max_similarity = 0
                for features in self.scratch_features:
                    similarity = self._compute_similarity(scratch_features, features)
                    max_similarity = max(max_similarity, similarity)

                # Add validation score
                scratch['dataset_validation_score'] = max_similarity

                # Boost confidence if high similarity
                if max_similarity > 0.8:
                    scratch['confidence_score'] = min(1.0, scratch.get('confidence_score', 0.5) * 1.2)

        return detected_scratches