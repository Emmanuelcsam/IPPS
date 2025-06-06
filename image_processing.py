#!/usr/bin/env python3
# image_processing.py

"""
D-Scope Blink: Image Processing Engine
======================================
This module contains the core logic for processing fiber optic end face images.
It includes functions for preprocessing, fiber localization (cladding and core),
zone mask generation, and the multi-algorithm defect detection engine with fusion.

This version is enhanced with an optional C++ accelerator for the DO2MR algorithm.
"""
# Standard and third-party library imports
import cv2 
import numpy as np 
from typing import Dict, Any, Optional, List, Tuple 
import logging 
from pathlib import Path 
import pywt
from scipy import ndimage
from skimage.feature import local_binary_pattern

# --- C++ Accelerator Integration ---
# Attempt to import the compiled C++ accelerator module.
# If it's not found, the pure Python implementations will be used as a fallback.
try:
    import dscope_accelerator
    CPP_ACCELERATOR_AVAILABLE = True
    logging.info("Successfully imported 'dscope_accelerator' C++ module. DO2MR will be accelerated.")
except ImportError:
    CPP_ACCELERATOR_AVAILABLE = False
    logging.warning("C++ accelerator module ('dscope_accelerator') not found. "
                    "Falling back to pure Python implementations. "
                    "For a significant performance increase, compile the C++ module using setup.py.")

# D-Scope Blink module imports
try:
    from anomaly_detection import AnomalyDetector
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False
    
try:
    import circle_fit as cf
    CIRCLE_FIT_AVAILABLE = True
except ImportError:
    CIRCLE_FIT_AVAILABLE = False
    
try:
    from config_loader import get_config
except ImportError:
    logging.warning("Could not import get_config from config_loader. Using dummy config for standalone testing.")
    # Dummy get_config function for standalone testing remains the same...
    def get_config() -> Dict[str, Any]:
        return {
            "algorithm_parameters": {
                "flat_field_image_path": None, "morph_gradient_kernel_size": [5,5],
                "black_hat_kernel_size": [11,11], "lei_kernel_lengths": [11,17],
                "lei_angle_step_deg": 15, "sobel_scharr_ksize": 3,
                "skeletonization_dilation_kernel_size": [3,3]
            },
        }

# --- Image Loading and Preprocessing ---
def load_and_preprocess_image(image_path_str: str, profile_config: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Loads an image, converts it to grayscale, and applies configured preprocessing steps.
    (This function remains unchanged)
    """
    image_path = Path(image_path_str)
    if not image_path.exists() or not image_path.is_file():
        logging.error(f"Image file not found or is not a file: {image_path}")
        return None

    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        logging.error(f"Failed to load image: {image_path}")
        return None
    logging.info(f"Image '{image_path.name}' loaded successfully.")

    gray_image = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    logging.debug("Image converted to grayscale.")

    clahe_clip_limit = profile_config.get("preprocessing", {}).get("clahe_clip_limit", 2.0)
    clahe_tile_size_list = profile_config.get("preprocessing", {}).get("clahe_tile_grid_size", [8, 8])
    clahe_tile_grid_size = tuple(clahe_tile_size_list)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    illum_corrected_image = clahe.apply(gray_image)
    logging.debug(f"CLAHE applied with clipLimit={clahe_clip_limit}, tileGridSize={clahe_tile_grid_size}.")
    
    # ... rest of the preprocessing function is identical to the original ...
    blur_kernel_list = profile_config.get("preprocessing", {}).get("gaussian_blur_kernel_size", [5, 5])
    gaussian_blur_kernel_size = tuple(k if k % 2 == 1 else k + 1 for k in blur_kernel_list)
    processed_image = cv2.GaussianBlur(illum_corrected_image, gaussian_blur_kernel_size, 0)
    logging.debug(f"Gaussian blur applied with kernel size {gaussian_blur_kernel_size}.")

    return original_bgr, gray_image, processed_image

# --- All other functions like locate_fiber_structure, generate_zone_masks, etc. remain unchanged ---
# ...
# ... (locate_fiber_structure, generate_zone_masks, and other helper functions are identical)
# ...

def _do2mr_detection(masked_zone_image: np.ndarray, kernel_size: int = 5, gamma: float = 1.5) -> np.ndarray:
    """
    Enhanced DO2MR implementation that uses a C++ accelerator if available.
    If the compiled C++ module `dscope_accelerator` is not found, it gracefully
    falls back to the pure Python/NumPy implementation.

    Args:
        masked_zone_image: The input image for a specific zone, where non-zone pixels are 0.
        kernel_size: The size of the structuring element for morphological operations.
        gamma: The multiplier for the standard deviation in sigma-based thresholding.

    Returns:
        A binary numpy array representing the defect mask.
    """
    # --- C++ Accelerated Path ---
    if CPP_ACCELERATOR_AVAILABLE:
        try:
            # The C++ function expects a uint8 NumPy array.
            # It directly processes the image and returns the final mask.
            # The 'masked_zone_image' also serves as the mask for stats calculation inside C++.
            return dscope_accelerator.do2mr_detection(masked_zone_image, kernel_size, gamma)
        except Exception as e:
            logging.error(f"C++ accelerator call for DO2MR failed: {e}. Falling back to Python implementation.")
            # Fall through to the Python implementation upon failure.

    # --- Pure Python Fallback Implementation ---
    # This code block is executed only if the C++ module is not available or if it failed.
    # It is identical to your original implementation.
    if masked_zone_image.dtype != np.uint8:
        normalized = cv2.normalize(masked_zone_image, None, 0, 255, cv2.NORM_MINMAX)
        masked_zone_image = normalized.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    I_max = cv2.dilate(masked_zone_image, kernel, iterations=1)
    I_min = cv2.erode(masked_zone_image, kernel, iterations=1)
    I_residual = cv2.subtract(I_max, I_min)
    I_residual_filtered = cv2.medianBlur(I_residual, 3)
    
    active_pixels_mask = masked_zone_image > 0
    if np.sum(active_pixels_mask) == 0:
        return np.zeros_like(masked_zone_image, dtype=np.uint8)
    
    zone_residual_values = I_residual_filtered[active_pixels_mask].astype(np.float32)
    mu = np.mean(zone_residual_values)
    sigma = np.std(zone_residual_values)
    
    threshold_value = mu + gamma * sigma
    
    _, defect_binary = cv2.threshold(I_residual_filtered, threshold_value, 255, cv2.THRESH_BINARY)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    defect_binary_cleaned = cv2.morphologyEx(defect_binary, cv2.MORPH_OPEN, kernel_open)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_binary_cleaned, connectivity=8)
    final_mask = np.zeros_like(defect_binary_cleaned)
    
    min_defect_area_px = 5
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_defect_area_px:
            final_mask[labels == i] = 255
    
    return final_mask

# --- The rest of image_processing.py remains unchanged ---
# The main `detect_defects` function will now automatically call the new, faster
# `_do2mr_detection` function without any other changes needed.
# ...
# ... (all other functions like _lei_scratch_detection, detect_defects, etc. are identical to original)
# ...
