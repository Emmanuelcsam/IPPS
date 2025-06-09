#!/usr/bin/env python3
# anomalib_integration.py

"""
D-Scope Blink: Anomalib Full Integration
=======================================
Integrates the complete Anomalib library for advanced anomaly detection
"""

import numpy as np
import cv2
from pathlib import Path
import torch
from typing import Dict, Any, Optional, Tuple, List
import logging

# Anomalib imports
from anomalib.models import Padim, Patchcore, DFM, STFPM
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.data import TaskType
from anomalib.deploy import OpenVINOInferencer, TorchInferencer
from anomalib.data.utils import read_image
from anomalib.pre_processing import PreProcessor
from omegaconf import OmegaConf

class AnomalibDefectDetector:
    """
    Advanced anomaly detector using multiple Anomalib models
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Default configuration
        self.config = {
            "dataset": {
                "task": TaskType.SEGMENTATION,
                "image_size": [256, 256],
                "normalization_method": InputNormalizationMethod.IMAGENET,
            },
            "model": {
                "padim": {
                    "backbone": "resnet18",
                    "layers": ["layer1", "layer2", "layer3"],
                    "pre_trained": True,
                    "n_features": 100,
                },
                "patchcore": {
                    "backbone": "wide_resnet50_2",
                    "layers": ["layer2", "layer3"],
                    "pre_trained": True,
                    "coreset_sampling_ratio": 0.1,
                    "num_neighbors": 9,
                },
                "stfpm": {
                    "backbone": "resnet18",
                    "layers": ["layer1", "layer2", "layer3"],
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            loaded_config = OmegaConf.load(config_path)
            self.config = OmegaConf.merge(self.config, loaded_config)
    
    def initialize_models(self, model_types: List[str] = ["padim", "patchcore"]):
        """Initialize specified anomaly detection models"""
        
        for model_type in model_types:
            if model_type == "padim":
                self.models["padim"] = Padim(
                    backbone=self.config["model"]["padim"]["backbone"],
                    layers=self.config["model"]["padim"]["layers"],
                    pre_trained=self.config["model"]["padim"]["pre_trained"],
                    n_features=self.config["model"]["padim"]["n_features"],
                )
            elif model_type == "patchcore":
                self.models["patchcore"] = Patchcore(
                    backbone=self.config["model"]["patchcore"]["backbone"],
                    layers=self.config["model"]["patchcore"]["layers"],
                    pre_trained=self.config["model"]["patchcore"]["pre_trained"],
                    coreset_sampling_ratio=self.config["model"]["patchcore"]["coreset_sampling_ratio"],
                    num_neighbors=self.config["model"]["patchcore"]["num_neighbors"],
                )
            elif model_type == "stfpm":
                self.models["stfpm"] = STFPM(
                    backbone=self.config["model"]["stfpm"]["backbone"],
                    layers=self.config["model"]["stfpm"]["layers"],
                )
            
            logging.info(f"Initialized {model_type} model")
    
    def train_on_fiber_dataset(self, normal_images_dir: Path, model_type: str = "padim"):
        """
        Train a model on normal fiber optic images
        
        Args:
            normal_images_dir: Directory containing defect-free fiber images
            model_type: Type of model to train
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not initialized")
        
        model = self.models[model_type]
        
        # Create custom dataset for fiber optics
        from anomalib.data import Folder
        
        datamodule = Folder(
            root=normal_images_dir.parent,
            normal_dir=normal_images_dir.name,
            task=TaskType.SEGMENTATION,
            image_size=self.config["dataset"]["image_size"],
            normalization=self.config["dataset"]["normalization_method"],
            train_batch_size=8,
            eval_batch_size=8,
            num_workers=4,
        )
        
        # Train the model
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint
        
        callbacks = [
            ModelCheckpoint(
                dirpath=f"models/anomalib/{model_type}",
                filename=f"fiber_optics_{model_type}",
                monitor="image_AUROC",
                mode="max",
                save_top_k=1,
            )
        ]
        
        trainer = Trainer(
            callbacks=callbacks,
            accelerator="auto",
            devices=1,
            max_epochs=50,
            check_val_every_n_epoch=5,
        )
        
        trainer.fit(model=model, datamodule=datamodule)
        logging.info(f"Training completed for {model_type}")
        
        # Export to OpenVINO for faster inference
        export_path = Path(f"models/anomalib/{model_type}/openvino")
        model.export_model(
            export_type="openvino",
            export_path=export_path,
        )
        
        return export_path
    
    def create_ensemble_detector(self, model_paths: Dict[str, Path]):
        """Create an ensemble of different anomaly detectors"""
        
        self.ensemble_models = {}
        
        for model_type, model_path in model_paths.items():
            if model_path.exists():
                # Use OpenVINO for faster inference
                self.ensemble_models[model_type] = OpenVINOInferencer(
                    path=model_path,
                    device="CPU",  # or "GPU" if available
                )
                logging.info(f"Loaded {model_type} model from {model_path}")
    
    def detect_with_ensemble(self, image: np.ndarray, zone_mask: Optional[np.ndarray] = None,
                           weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
        """
        Detect anomalies using ensemble of models
        
        Args:
            image: Input image (BGR or grayscale)
            zone_mask: Optional zone mask
            weights: Model weights for ensemble
            
        Returns:
            Combined anomaly map, score, and individual model outputs
        """
        if not self.ensemble_models:
            raise RuntimeError("No models loaded in ensemble")
        
        # Default weights
        if weights is None:
            weights = {model: 1.0 for model in self.ensemble_models}
        
        # Ensure RGB format for Anomalib
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Apply zone mask if provided
        if zone_mask is not None:
            # Create 3-channel mask
            mask_3ch = cv2.cvtColor(zone_mask, cv2.COLOR_GRAY2RGB)
            image_rgb = cv2.bitwise_and(image_rgb, mask_3ch)
        
        # Get predictions from each model
        individual_maps = {}
        combined_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        for model_name, model in self.ensemble_models.items():
            try:
                # Run inference
                predictions = model.predict(image=image_rgb)
                
                # Extract anomaly map
                if hasattr(predictions, "anomaly_map"):
                    anomaly_map = predictions.anomaly_map
                elif hasattr(predictions, "heat_map"):
                    anomaly_map = predictions.heat_map
                else:
                    logging.warning(f"Model {model_name} returned unexpected output format")
                    continue
                
                # Ensure correct shape
                if anomaly_map.shape[:2] != image.shape[:2]:
                    anomaly_map = cv2.resize(anomaly_map, (image.shape[1], image.shape[0]))
                
                # Store individual result
                individual_maps[model_name] = anomaly_map
                
                # Add weighted contribution to ensemble
                weight = weights.get(model_name, 1.0)
                combined_map += anomaly_map * weight
                
            except Exception as e:
                logging.error(f"Error in {model_name} inference: {e}")
                continue
        
        # Normalize combined map
        if len(individual_maps) > 0:
            combined_map /= sum(weights.values())
            
            # Calculate overall anomaly score
            anomaly_score = np.mean(combined_map)
            
            # Apply zone mask to final result
            if zone_mask is not None:
                combined_map = cv2.bitwise_and(combined_map, combined_map, mask=zone_mask)
        else:
            anomaly_score = 0.0
        
        return combined_map, anomaly_score, individual_maps