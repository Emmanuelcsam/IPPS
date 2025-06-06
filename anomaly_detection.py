#!/usr/bin/env python3
# anomaly_detection.py

"""
D-Scope Blink: Anomaly Detection Module
======================================
Integrates deep learning-based anomaly detection using Anomalib
for enhanced defect detection capabilities.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import cv2 # Added import for OpenCV

try:
    from anomalib.data.utils import read_image
    from anomalib.deploy import OpenVINOInferencer
    from anomalib.models import Padim
    ANOMALIB_AVAILABLE = True
except ImportError:
    ANOMALIB_AVAILABLE = False
    logging.warning("Anomalib not available. Deep learning features disabled.")

class AnomalyDetector:
    """
    Wrapper for Anomalib-based anomaly detection.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the anomaly detector."""
        self.model = None
        self.inferencer = None

        if not ANOMALIB_AVAILABLE:
            logging.warning("Anomalib not installed. Anomaly detection disabled.")
            return

        if model_path:
            model_file_path = Path(model_path) # Convert to Path object once
            if model_file_path.exists():
                try:
                    # OpenVINOInferencer might expect a string path, or Path object.
                    # Pass the Path object; if it strictly needs a string,
                    # use str(model_file_path).
                    self.inferencer = OpenVINOInferencer(
                        path=model_file_path, 
                        device="CPU"  # To use GPU, change to "GPU" and ensure OpenVINO is configured for it.
                    )
                    logging.info(f"Loaded anomaly detection model from {model_file_path}")
                except Exception as e:
                    logging.error(f"Failed to load anomaly model: {e}")
                    self.inferencer = None
            else:
                logging.error(f"Provided anomaly model path does not exist: {model_file_path}")
                self.inferencer = None
        else:
            logging.info("No anomaly model path provided; anomaly detection will be skipped.")

    
    def detect_anomalies(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect anomalies in the given image.

        Returns:
            Binary mask of detected anomalies, or None if detection fails.
        """
        if not self.inferencer:
            return None

        try:
            # Ensure input image is the correct shape (e.g., BGR or grayscale)
            # Anomalib's OpenVINOInferencer typically expects images in RGB format.
            # If your input 'image' is BGR (common with OpenCV), convert it.
            if image.ndim == 3 and image.shape[2] == 3:
                # Convert BGR to RGB
                inp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.ndim == 2: # Grayscale image
                # If the model expects a 3-channel image, convert grayscale to RGB
                # This depends on the model's training. Assuming it might need 3 channels:
                inp = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                inp = image.copy() # Or handle other cases as needed

            # Run inference
            predictions = self.inferencer.predict(image=inp)
            if not hasattr(predictions, "anomaly_map"):
                logging.error("Anomaly detector returned unexpected prediction format - missing anomaly_map.")
                return None
            
            # Extract anomaly map
            anomaly_map = predictions.anomaly_map
            
            # Handle different threshold scenarios
            if hasattr(predictions, "pred_score"):
                # Use model's prediction score as threshold
                threshold = predictions.pred_score
            else:
                # Use Otsu's method for automatic threshold
                if anomaly_map.dtype != np.uint8:
                    anomaly_map_uint8 = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                else:
                    anomaly_map_uint8 = anomaly_map
                threshold, _ = cv2.threshold(anomaly_map_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                threshold = threshold / 255.0  # Normalize back to [0,1] if anomaly_map is normalized
            
            # Apply threshold
            if isinstance(threshold, (float, np.floating)):
                anomaly_mask = (anomaly_map > threshold).astype(np.uint8) * 255
            else:
                # If threshold is an array, use element-wise comparison
                anomaly_mask = (anomaly_map > threshold).astype(np.uint8) * 255
                # if pred_score is not a threshold, this logic needs to be replaced.
                # logging.warning("pred_score is not a scalar, default thresholding strategy might be needed.")
                # As a fallback or if the above is not desired:
                # default_threshold = 0.5 # Example
                # norm_anomaly_map = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map) + 1e-6)
                # anomaly_mask = (norm_anomaly_map > default_threshold).astype(np.uint8) * 255


            return anomaly_mask

        except Exception as e:
            logging.error(f"Anomaly detection failed: {e}", exc_info=True) # Added exc_info for more details
            return None

    
    def train_on_good_samples(self, good_sample_dir: str, save_path: str):
        """
        Train a new anomaly detection model on good samples.
        """
        if not ANOMALIB_AVAILABLE:
            logging.error("Cannot train: Anomalib not available")
            return False
            
        # This is a placeholder for training logic
        # In practice, you'd use Anomalib's training pipeline
        # Example using Padim (requires a full training setup):
        # from anomalib.models import Padim
        # from anomalib.engine import Engine
        # from anomalib.data import MVTec
        # from pytorch_lightning import Trainer

        # model = Padim()
        # datamodule = MVTec(root=good_sample_dir, category="<your_category_name>")
        # engine = Engine(model=model, data=datamodule)
        # trainer = Trainer(...) # Configure trainer
        # engine.train(trainer)
        # # Logic to export the model to OpenVINO format and save to save_path

        logging.info("Training functionality would be implemented here using Anomalib's training pipeline.")
        logging.info(f"Hypothetical training with samples from {good_sample_dir}, saving to {save_path}")
        # Placeholder return
        return True