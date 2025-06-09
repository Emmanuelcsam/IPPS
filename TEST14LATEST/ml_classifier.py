#!/usr/bin/env python3
# ml_classifier.py

"""
 ML-Based Defect Classifier
========================================
Machine learning classifier for defect type determination
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, List, Optional, Tuple
import logging

class DefectClassifier:
    """
    ML-based classifier for defect types
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.feature_names = [
            'area_px', 'aspect_ratio', 'solidity', 'eccentricity',
            'mean_intensity', 'std_intensity', 'perimeter_ratio',
            'hu_moment_1', 'hu_moment_2', 'orientation_deg'
        ]
        self.fitted = False
        
        if model_path:
            self.load_model(model_path)
    
    def extract_features(self, defect_dict: Dict, 
                        original_image: np.ndarray) -> np.ndarray:
        """
        Extract features for classification
        
        Args:
            defect_dict: Dictionary with defect properties
            original_image: Original grayscale image
            
        Returns:
            Feature vector
        """
        features = []
        
        # Basic geometric features
        features.append(defect_dict.get('area_px', 0))
        features.append(defect_dict.get('aspect_ratio', 1))
        
        # Calculate additional features
        contour_points = defect_dict.get('contour_points_px', [])
        if len(contour_points) > 5:
            contour_np = np.array(contour_points, dtype=np.int32)
            
            # Solidity
            hull = cv2.convexHull(contour_np)
            hull_area = cv2.contourArea(hull)
            solidity = defect_dict['area_px'] / hull_area if hull_area > 0 else 0
            features.append(solidity)
            
            # Eccentricity
            if len(contour_np) >= 5:
                ellipse = cv2.fitEllipse(contour_np)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2) if major_axis > 0 else 0
                features.append(eccentricity)
            else:
                features.append(0)
            
            # Intensity features
            mask = np.zeros(original_image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour_np], -1, 255, -1)
            
            masked_pixels = original_image[mask > 0]
            if len(masked_pixels) > 0:
                features.append(np.mean(masked_pixels))
                features.append(np.std(masked_pixels))
            else:
                features.extend([0, 0])
            
            # Perimeter ratio
            perimeter = cv2.arcLength(contour_np, True)
            expected_perimeter = 2 * np.pi * np.sqrt(defect_dict['area_px'] / np.pi)
            perimeter_ratio = perimeter / expected_perimeter if expected_perimeter > 0 else 1
            features.append(perimeter_ratio)
            
            # Hu moments (shape descriptors)
            moments = cv2.moments(contour_np)
            hu_moments = cv2.HuMoments(moments).flatten()
            features.extend([hu_moments[0], hu_moments[1]])
            
            # Orientation
            angle = ellipse[2] if len(contour_np) >= 5 else 0
            features.append(angle)
            
        else:
            # Default values if contour is too small
            features.extend([0] * 8)
        
        return np.array(features)
    
    def train(self, training_data: List[Tuple[Dict, np.ndarray, str]]):
        """
        Train the classifier
        
        Args:
            training_data: List of (defect_dict, original_image, label) tuples
        """
        X = []
        y = []
        
        for defect_dict, original_image, label in training_data:
            features = self.extract_features(defect_dict, original_image)
            X.append(features)
            y.append(label)
        
        X = np.array(X)
        
        # Fit scaler and transform features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier.fit(X_scaled, y)
        self.fitted = True
        
        logging.info(f"Trained defect classifier on {len(y)} samples")
        
    def predict(self, defect_dict: Dict, original_image: np.ndarray) -> Tuple[str, float]:
        """
        Predict defect type
        
        Args:
            defect_dict: Defect properties
            original_image: Original image
            
        Returns:
            Predicted class and confidence
        """
        if not self.fitted:
            # Fallback to rule-based classification
            aspect_ratio = defect_dict.get('aspect_ratio', 1)
            return ("Scratch" if aspect_ratio >= 3.0 else "Pit/Dig", 0.5)
        
        features = self.extract_features(defect_dict, original_image)
        features_scaled = self.scaler.transform([features])
        
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence
    
    def save_model(self, path: str):
        """Save trained model"""
        joblib.dump({
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'fitted': self.fitted
        }, path)
        
    def load_model(self, path: str):
        """Load trained model"""
        data = joblib.load(path)
        self.classifier = data['classifier']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.fitted = data['fitted']