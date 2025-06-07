#!/usr/bin/env python3
# train_advanced_models.py

"""
Training script for advanced models
"""

import argparse
from pathlib import Path
import logging
from anomalib_integration import AnomalibDefectDetector
from padim_specific import FiberPaDiM
from segdecnet_integration import FiberSegDecNet
import torch
import numpy as np
import cv2

def train_anomalib_models(normal_images_dir: Path, output_dir: Path):
    """Train Anomalib models on normal fiber images"""
    detector = AnomalibDefectDetector()
    detector.initialize_models(["padim", "patchcore"])
    
    for model_type in ["padim", "patchcore"]:
        logging.info(f"Training {model_type}...")
        export_path = detector.train_on_fiber_dataset(normal_images_dir, model_type)
        logging.info(f"Model exported to {export_path}")

def train_padim_specific(normal_images_dir: Path, output_path: Path):
    """Train specific PaDiM implementation"""
    model = FiberPaDiM()
    
    # Load normal images
    normal_images = []
    for img_path in normal_images_dir.glob("*.png"):
        img = cv2.imread(str(img_path))
        normal_images.append(img)
    
    # Fit model
    model.fit(normal_images)
    
    # Save model
    torch.save({
        'patch_means': model.patch_means,
        'C': model.C,
        'C_inv': model.C_inv,
        'R': model.R,
        'N': model.N,
        't_d': model.t_d,
        'd': model.d
    }, output_path)
    
    logging.info(f"PaDiM model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_images", type=str, required=True,
                       help="Directory with normal (defect-free) images")
    parser.add_argument("--output_dir", type=str, default="models",
                       help="Output directory for trained models")
    parser.add_argument("--models", nargs="+", default=["anomalib", "padim"],
                       help="Models to train")
    
    args = parser.parse_args()
    
    normal_dir = Path(args.normal_images)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if "anomalib" in args.models:
        train_anomalib_models(normal_dir, output_dir / "anomalib")
    
    if "padim" in args.models:
        train_padim_specific(normal_dir, output_dir / "padim_fiber.pth")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()