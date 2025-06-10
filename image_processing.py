#!/usr/bin/env python3
# image_processing.py

"""
Image Processing Engine
======================================
This module contains the core logic for processing fiber optic end face images.
It includes functions for preprocessing, fiber localization (cladding and core),
zone mask generation, and the multi-algorithm defect detection engine with fusion.

This version is enhanced with an optional C++ accelerator for the DO2MR algorithm
and includes improved core detection with multiple fallback methods.
"""
# Standard and third-party library imports
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import cv2
import logging
from pathlib import Path
import pywt
from scipy import ndimage
from skimage.feature import local_binary_pattern

try:
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logging.warning("scikit-image features not available")

# --- C++ Accelerator Integration ---
try:
    import accelerator  
    CPP_ACCELERATOR_AVAILABLE = True
    logging.info("Successfully imported 'accelerator' C++ module. DO2MR will be accelerated.")
except ImportError:
    CPP_ACCELERATOR_AVAILABLE = False
    logging.warning("C++ accelerator module ('accelerator') not found. "
                    "Falling back to pure Python implementations. "
                    "For a significant performance increase, compile the C++ module using setup.py.")

# Import all advanced detection modules
try:
    from anomalib_integration import AnomalibDefectDetector
    ANOMALIB_FULL_AVAILABLE = True
except ImportError:
    ANOMALIB_FULL_AVAILABLE = False

try:
    from padim_specific import FiberPaDiM
    PADIM_SPECIFIC_AVAILABLE = True
except ImportError:
    PADIM_SPECIFIC_AVAILABLE = False

try:
    from segdecnet_integration import FiberSegDecNet
    SEGDECNET_AVAILABLE = True
except ImportError:
    SEGDECNET_AVAILABLE = False

try:
    from advanced_scratch_detection import AdvancedScratchDetector
    ADVANCED_SCRATCH_AVAILABLE = True
except ImportError:
    ADVANCED_SCRATCH_AVAILABLE = False

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
    from padim_integration import PaDiMDetector, integrate_padim_detection
    PADIM_AVAILABLE = True
except ImportError:
    PADIM_AVAILABLE = False
    logging.warning("PaDiM integration not available")

try:
    from config_loader import get_config
except ImportError:
    logging.warning("Could not import get_config from config_loader. Using dummy config for standalone testing.")

def get_dummy_config():
    """Fallback configuration for standalone testing."""
    return {
        "processing_profiles": {
            "deep_inspection": {
                "defect_detection": {"min_defect_area_px": 5}
            }
        }
    }

# --- Image Loading and Preprocessing ---
def load_and_preprocess_image(image_path_str: str, profile_config: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Loads an image, converts it to grayscale, and applies configured preprocessing steps.
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
    
    blur_kernel_list = profile_config.get("preprocessing", {}).get("gaussian_blur_kernel_size", [5, 5])
    gaussian_blur_kernel_size = tuple(k if k % 2 == 1 else k + 1 for k in blur_kernel_list)
    processed_image = cv2.GaussianBlur(illum_corrected_image, gaussian_blur_kernel_size, 0)
    logging.debug(f"Gaussian blur applied with kernel size {gaussian_blur_kernel_size}.")

    return original_bgr, gray_image, processed_image

def _correct_illumination(gray_image: np.ndarray, original_dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    Performs advanced illumination correction using rolling ball algorithm.
    """
    # Estimate background using morphological closing with large kernel
    kernel_size = 50
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    
    # Subtract background
    corrected_int16 = cv2.subtract(gray_image.astype(np.int16), background.astype(np.int16))
    corrected_int16 = corrected_int16 + 128  # Shift to mid-gray
    # Clip and convert back to original dtype
    corrected = np.clip(corrected_int16, 0, 255).astype(original_dtype)
    
    return corrected

def detect_core_improved(image, cladding_center, cladding_radius, core_diameter_hint=None):
    """
    Enhanced core detection with intensity-based analysis to prevent false defect detection
    """
    import cv2
    import numpy as np
    from scipy import ndimage
    
    # Create tighter cladding mask focusing on actual core region
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Use smaller search radius (0.3 instead of 0.8) to focus on core area
    cv2.circle(mask, tuple(map(int, cladding_center)), int(cladding_radius * 0.3), 255, -1)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    gray_smooth = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # Intensity-based core detection
    masked_region = cv2.bitwise_and(gray_smooth, mask)
    
    # Calculate radial intensity profile
    cy, cx = int(cladding_center[1]), int(cladding_center[0])
    Y, X = np.ogrid[:gray.shape[0], :gray.shape[1]]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # Sample intensity values at different radii
    max_radius = int(cladding_radius * 0.3)
    intensity_profile = []
    
    for r in range(0, max_radius, 2):
        ring_mask = (dist_from_center >= r) & (dist_from_center < r + 2) & (mask > 0)
        if np.any(ring_mask):
            mean_intensity = np.mean(gray_smooth[ring_mask])
            intensity_profile.append((r, mean_intensity))
    
    if len(intensity_profile) > 5:
        # Find the radius where intensity changes significantly
        radii = np.array([p[0] for p in intensity_profile])
        intensities = np.array([p[1] for p in intensity_profile])
        
        # Calculate intensity gradient
        gradient = np.gradient(intensities)
        
        # Find the maximum gradient location (core-cladding boundary)
        max_gradient_idx = np.argmax(np.abs(gradient[1:-1])) + 1
        
        if max_gradient_idx < len(radii) - 1:
            core_radius = radii[max_gradient_idx]
            
            # Validate the detected radius
            if 3 < core_radius < cladding_radius * 0.15:  # Reasonable core size
                return tuple(cladding_center), core_radius * 2
    
    # Method 1: Adaptive thresholding (existing code enhanced)
    try:
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        adaptive_thresh = cv2.bitwise_and(adaptive_thresh, mask)
        
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter contours by area and circularity
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if 0.3 < circularity < 1.2 and area > 100:  # Reasonable circularity and size
                        valid_contours.append(contour)
            
            if valid_contours:
                # Find the most central contour
                best_contour = None
                min_distance = float('inf')
                
                for contour in valid_contours:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    distance = np.sqrt((x - cladding_center[0])**2 + (y - cladding_center[1])**2)
                    if distance < min_distance:
                        min_distance = distance
                        best_contour = contour
                
                if best_contour is not None:
                    (x, y), radius = cv2.minEnclosingCircle(best_contour)
                    return (x, y), radius * 2  # Return diameter
    except Exception as e:
        print(f"Adaptive threshold method failed: {e}")
    
    # Method 2: Edge-based detection (existing code)
    try:
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_and(edges, mask)
        
        # Dilate to connect edge fragments
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Similar filtering as above
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Minimum area threshold
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    distance = np.sqrt((x - cladding_center[0])**2 + (y - cladding_center[1])**2)
                    if distance < cladding_radius * 0.3:  # Core should be near center
                        return (x, y), radius * 2
    except Exception as e:
        print(f"Edge-based method failed: {e}")
    
    # Method 3: Improved fallback based on cladding
    print("Using improved fallback method for core detection")
    if core_diameter_hint:
        core_radius = core_diameter_hint / 2
    else:
        # Better estimation for single-mode fibers (typically 9µm core in 125µm cladding)
        core_radius = cladding_radius * 0.072
    
    return tuple(cladding_center), core_radius * 2

def locate_fiber_structure(
    processed_image: np.ndarray,
    profile_config: Dict[str, Any],
    original_gray_image: Optional[np.ndarray] = None
) -> Optional[Dict[str, Any]]:
    """
    Locates the fiber cladding and core using HoughCircles, contour fitting, or circle-fit library.
    Enhanced with improved core detection methods.
    """
    # Get localization parameters from the profile configuration
    loc_params = profile_config.get("localization", {})
    h, w = processed_image.shape[:2]
    min_img_dim = min(h, w)

    # Initialize Parameters for HoughCircles
    dp = loc_params.get("hough_dp", 1.2)
    min_dist_circles = int(min_img_dim * loc_params.get("hough_min_dist_factor", 0.15))
    param1 = loc_params.get("hough_param1", 70)
    param2 = loc_params.get("hough_param2", 35)
    min_radius_hough = int(min_img_dim * loc_params.get("hough_min_radius_factor", 0.08))
    max_radius_hough = int(min_img_dim * loc_params.get("hough_max_radius_factor", 0.45))

    # Initialize the result dictionary
    localization_result = {}

    # --- Cladding Detection using HoughCircles ---
    logging.info("Attempting cladding detection using HoughCircles...")
    circles = cv2.HoughCircles(
        processed_image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist_circles,
        param1=param1,
        param2=param2,
        minRadius=min_radius_hough,
        maxRadius=max_radius_hough
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if len(circles) > 0:
            # Use the first (strongest) circle for cladding
            x_cl, y_cl, r_cl = circles[0]
            localization_result['cladding_center_xy'] = (x_cl, y_cl)
            localization_result['cladding_radius_px'] = float(r_cl)
            localization_result['localization_method'] = 'HoughCircles'
            logging.info(f"Cladding (HoughCircles): Center=({x_cl},{y_cl}), Radius={r_cl}px")

    # --- Fallback: Contour-based detection ---
    if 'cladding_center_xy' not in localization_result:
        logging.info("HoughCircles failed, attempting contour-based detection...")
        try:
            # Apply Canny edge detection
            edges = cv2.Canny(processed_image, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (likely to be the cladding)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Fit circle to the largest contour
                if len(largest_contour) >= 5:  # Need at least 5 points for ellipse fitting
                    ellipse = cv2.fitEllipse(largest_contour)
                    (x_cl, y_cl), (minor_axis, major_axis), angle = ellipse
                    
                    # Check if it's reasonably circular
                    axis_ratio = minor_axis / major_axis if major_axis > 0 else 0
                    if axis_ratio > 0.7:  # Reasonably circular
                        avg_radius = (minor_axis + major_axis) / 4.0  # Average radius
                        localization_result['cladding_center_xy'] = (int(x_cl), int(y_cl))
                        localization_result['cladding_radius_px'] = float(avg_radius)
                        localization_result['cladding_ellipse_params'] = ellipse
                        localization_result['localization_method'] = 'ContourFitCircle'
                        logging.info(f"Cladding (Contour): Center=({int(x_cl)},{int(y_cl)}), Radius={avg_radius:.1f}px")
        except Exception as e:
            logging.error(f"Contour-based detection failed: {e}")

    # --- Circle-fit library fallback ---
    if 'cladding_center_xy' not in localization_result and CIRCLE_FIT_AVAILABLE:
        logging.info("Attempting circle-fit library method...")
        try:
            edges = cv2.Canny(processed_image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Combine all contour points
                all_points = np.vstack(contours).reshape(-1, 2)
                
                if len(all_points) > 10:  # Need sufficient points
                    # Use circle_fit library for robust fitting
                    xc_cf, yc_cf, r_cf, residual_cf = cf.hyper_fit(all_points)
                    
                    # Validate the result
                    if 0 < r_cf < min_img_dim * 0.5 and residual_cf < 50:
                        localization_result['cladding_center_xy'] = (int(xc_cf), int(yc_cf))
                        localization_result['cladding_radius_px'] = float(r_cf)
                        localization_result['localization_method'] = 'CircleFitLib'
                        localization_result['fit_residual'] = residual_cf
                        logging.info(f"Cladding (CircleFitLib): Center=({int(xc_cf)},{int(yc_cf)}), Radius={r_cf:.1f}px, Residual={residual_cf:.3f}")
        except Exception as e:
            logging.error(f"Circle-fit library method failed: {e}")

    # --- Check if cladding was found ---
    if 'cladding_center_xy' not in localization_result:
        logging.error("Failed to localize fiber cladding by any method.")
        return None

    # --- Core Detection (Enhanced) ---
    logging.info("Starting enhanced core detection...")
    
    # Ensure original_gray_image is used for better intensity distinction if available
    image_for_core_detect = original_gray_image if original_gray_image is not None else processed_image
    
    cladding_center = localization_result['cladding_center_xy']
    cladding_radius = localization_result['cladding_radius_px']
    
    # Get core diameter hint from config if available
    core_diameter_hint = loc_params.get("expected_core_diameter_px", None)
    
    try:
        # Use the improved core detection function
        core_center, core_diameter = detect_core_improved(
            image_for_core_detect, 
            cladding_center, 
            cladding_radius,
            core_diameter_hint
        )
        
        localization_result['core_center_xy'] = tuple(map(int, core_center))
        localization_result['core_radius_px'] = float(core_diameter / 2)
        logging.info(f"Core detected: Center=({int(core_center[0])},{int(core_center[1])}), Radius={core_diameter/2:.1f}px")
        
    except Exception as e:
        logging.error(f"Enhanced core detection failed: {e}")
        # Final fallback
        localization_result['core_center_xy'] = localization_result['cladding_center_xy']
        # Better estimation for single-mode fibers (typically 9µm core in 125µm cladding)
        localization_result['core_radius_px'] = localization_result['cladding_radius_px'] * 0.072
        logging.warning("Core detection failed, using fallback estimation")

    return localization_result

def generate_zone_masks(
    image_shape: Tuple[int, int],
    localization_data: Dict[str, Any],
    zone_definitions: List[Dict[str, Any]],
    um_per_px: Optional[float],
    user_core_diameter_um: Optional[float],
    user_cladding_diameter_um: Optional[float]
) -> Dict[str, np.ndarray]:
    """
    Generates binary masks for Core and Cladding zones only.
    """
    masks: Dict[str, np.ndarray] = {}
    h, w = image_shape[:2]
    Y, X = np.ogrid[:h, :w]

    # Get detected fiber parameters
    cladding_center = localization_data.get('cladding_center_xy')
    core_center = localization_data.get('core_center_xy', cladding_center)
    core_radius_px_detected = localization_data.get('core_radius_px')
    detected_cladding_radius_px = localization_data.get('cladding_radius_px')

    if cladding_center is None:
        logging.error("Cannot generate zone masks: Cladding center not localized.")
        return masks

    cx, cy = cladding_center
    core_cx, core_cy = core_center if core_center else (cx, cy)

    # Calculate distance maps
    dist_sq_from_cladding = (X - cx)**2 + (Y - cy)**2
    dist_sq_from_core = (X - core_cx)**2 + (Y - core_cy)**2

    # Determine core radius
    if user_core_diameter_um and um_per_px:
        core_radius_px = (user_core_diameter_um / 2) / um_per_px
    elif core_radius_px_detected and core_radius_px_detected > 0:
        core_radius_px = core_radius_px_detected
    else:
        # Better estimation for single-mode fibers (typically 9µm core in 125µm cladding)
        core_radius_px = detected_cladding_radius_px * 0.072 if detected_cladding_radius_px else 5

    # Determine cladding radius
    if user_cladding_diameter_um and um_per_px:
        cladding_radius_px = (user_cladding_diameter_um / 2) / um_per_px
    elif detected_cladding_radius_px:
        cladding_radius_px = detected_cladding_radius_px
    else:
        logging.error("Cannot create zone masks: No cladding radius available")
        return masks

    # Create Core mask
    masks["Core"] = (dist_sq_from_core <= core_radius_px**2).astype(np.uint8) * 255
    
    # Create Cladding mask (excluding core)
    cladding_mask = (dist_sq_from_cladding <= cladding_radius_px**2).astype(np.uint8)
    core_mask = (dist_sq_from_core <= core_radius_px**2).astype(np.uint8)
    masks["Cladding"] = (cladding_mask - core_mask) * 255

    logging.info(f"Generated zone masks - Core radius: {core_radius_px:.1f}px, Cladding radius: {cladding_radius_px:.1f}px")

    return masks

def do2mr_detection(image: np.ndarray, zone_mask: np.ndarray, 
                   zone_name: str, global_algo_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced DO2MR with adaptive parameters and multi-scale detection
    """
    # Try C++ accelerator first if available
    if CPP_ACCELERATOR_AVAILABLE:
        try:
            kernel_size = global_algo_params.get("do2mr_kernel_size", 5)
            gamma = global_algo_params.get(f"do2mr_gamma_{zone_name.lower()}", 
                                          global_algo_params.get("do2mr_gamma_default", 1.5))
            # Apply adaptive sensitivity for core zone
            if zone_name == "Core":
                gamma = gamma * global_algo_params.get("adaptive_sensitivity_core", 0.8)
            
            result = accelerator.do2mr_detection(image, kernel_size, gamma)
            confidence = result.astype(np.float32) / 255.0
            return result, confidence
        except Exception as e:
            logging.debug(f"C++ accelerator failed, falling back to Python: {e}")
    
    # Enhanced Python implementation with multi-scale
    # Multi-scale kernel sizes for different defect sizes
    kernel_sizes = [3, 5, 7, 9] if zone_name == "Core" else [5, 7, 11]
    combined_result = np.zeros_like(image, dtype=np.float32)
    
    # Apply zone mask once
    masked_image = cv2.bitwise_and(image, image, mask=zone_mask)
    
    # Pre-filter to reduce noise
    denoised = cv2.bilateralFilter(masked_image, 5, 50, 50)
    
    for kernel_size in kernel_sizes:
        # Get base gamma for this zone
        gamma_base = global_algo_params.get(f"do2mr_gamma_{zone_name.lower()}", 
                                           global_algo_params.get("do2mr_gamma_default", 1.5))
        
        # Apply adaptive sensitivity for core zone
        if zone_name == "Core":
            gamma_base = gamma_base * global_algo_params.get("adaptive_sensitivity_core", 0.8)
        
        # Create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Max and min filtering
        max_filtered = cv2.dilate(denoised, kernel)
        min_filtered = cv2.erode(denoised, kernel)
        
        # Calculate residual
        residual = cv2.subtract(max_filtered, min_filtered)
        
        # Apply guided filter for edge-preserving smoothing if available
        try:
            if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
                residual_filtered = cv2.ximgproc.guidedFilter(
                    guide=denoised, 
                    src=residual, 
                    radius=3, 
                    eps=10
                )
            else:
                residual_filtered = cv2.medianBlur(residual, 3)
        except:
            residual_filtered = cv2.medianBlur(residual, 3)
        
        # Get pixels within zone for statistics
        zone_pixels = residual_filtered[zone_mask > 0]
        if len(zone_pixels) < 100:
            continue
        
        # Use robust statistics (median and MAD instead of mean and std)
        median_val = np.median(zone_pixels)
        mad = np.median(np.abs(zone_pixels - median_val))
        std_robust = 1.4826 * mad  # Conversion factor for normal distribution
        
        # Calculate local contrast for adaptive gamma
        denoised_zone_pixels = denoised[zone_mask > 0]
        if len(denoised_zone_pixels) > 0:
            local_mean = np.mean(denoised_zone_pixels)
            local_std = np.std(denoised_zone_pixels)
            local_contrast = local_std / (local_mean + 1e-6)
            adaptive_gamma = gamma_base * (1 + 0.5 * np.clip(local_contrast, 0, 1))
        else:
            adaptive_gamma = gamma_base
        
        # Multi-level thresholding with hysteresis for better connectivity
        if global_algo_params.get("multi_threshold_levels", True):
            # High threshold for definite defects
            threshold_high = median_val + adaptive_gamma * std_robust
            # Low threshold for possible defects
            threshold_low = median_val + (adaptive_gamma * 0.6) * std_robust
            
            # Apply thresholds
            _, high_mask = cv2.threshold(residual_filtered, threshold_high, 255, cv2.THRESH_BINARY)
            _, low_mask = cv2.threshold(residual_filtered, threshold_low, 255, cv2.THRESH_BINARY)
            
            # Apply zone mask to threshold results
            high_mask = cv2.bitwise_and(high_mask, zone_mask)
            low_mask = cv2.bitwise_and(low_mask, zone_mask)
            
            # Morphological reconstruction to connect regions
            kernel_recon = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # Use high confidence regions as markers
            marker = cv2.erode(high_mask, kernel_recon)
            
            # Reconstruct from markers within low threshold regions
            reconstructed = cv2.dilate(marker, kernel_recon, iterations=2)
            reconstructed = cv2.bitwise_and(reconstructed, low_mask)
            
            # Include all high confidence regions
            result_for_scale = cv2.bitwise_or(reconstructed, high_mask)
        else:
            # Single threshold (fallback)
            threshold = median_val + adaptive_gamma * std_robust
            _, result_for_scale = cv2.threshold(residual_filtered, threshold, 255, cv2.THRESH_BINARY)
            result_for_scale = cv2.bitwise_and(result_for_scale, zone_mask)
        
        # Weight by kernel size (smaller kernels get higher weight for small defects)
        weight = 1.0 / (1 + 0.5 * np.log(kernel_size))
        combined_result += result_for_scale.astype(np.float32) * weight
    
    # Normalize combined result
    if len(kernel_sizes) > 0:
        combined_result = combined_result / sum(1.0 / (1 + 0.5 * np.log(k)) for k in kernel_sizes)
    
    # Final thresholding
    _, defect_mask = cv2.threshold(combined_result, 127, 255, cv2.THRESH_BINARY)
    defect_mask = defect_mask.astype(np.uint8)
    
    # Apply zone mask to final result
    defect_mask = cv2.bitwise_and(defect_mask, zone_mask)
    
    # Morphological cleanup with zone-specific parameters
    min_defect_size = global_algo_params.get(f"min_defect_area_{zone_name.lower()}_px", 
                                            3 if zone_name == "Core" else 5)
    
    # Opening to remove noise
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel_clean)
    
    # Remove very small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_defect_size:
            defect_mask[labels == i] = 0
    
    # Create confidence map normalized to [0, 1]
    confidence_map = combined_result / 255.0
    confidence_map = np.clip(confidence_map, 0, 1)
    
    # Mask confidence map to zone
    zone_mask_float = zone_mask.astype(np.float32) / 255.0
    confidence_map = confidence_map * zone_mask_float
    
    return defect_mask, confidence_map


def lei_scratch_detection(image: np.ndarray, zone_mask: np.ndarray,
                         global_algo_params: Dict[str, Any]) -> np.ndarray:
    """
    Enhanced LEI with multi-scale and directional filtering
    """
    # Extended parameters for better coverage
    kernel_lengths = global_algo_params.get("lei_kernel_lengths_extended", 
                                           global_algo_params.get("lei_kernel_lengths", [7, 11, 15, 21, 31]))
    angle_step = global_algo_params.get("lei_angle_step_deg", 10)  # Finer angular resolution
    
    # Apply zone mask
    masked_image = cv2.bitwise_and(image, image, mask=zone_mask)
    
    # Multi-scale preprocessing
    scales = [1.0, 0.75, 1.25]
    all_scratch_maps = []
    
    for scale in scales:
        # Resize image
        if scale != 1.0:
            scaled_h = int(image.shape[0] * scale)
            scaled_w = int(image.shape[1] * scale)
            scaled_image = cv2.resize(masked_image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            scaled_mask = cv2.resize(zone_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
        else:
            scaled_image = masked_image
            scaled_mask = zone_mask
        
        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled_image)
        
        # Apply top-hat transform to enhance linear structures
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_tophat)
        
        # Initialize scratch map for this scale
        scratch_map = np.zeros_like(enhanced, dtype=np.float32)
        
        # Enhanced directional filtering
        for angle in range(0, 180, angle_step):
            angle_rad = np.deg2rad(angle)
            
            for kernel_length in kernel_lengths:
                # Create enhanced linear kernel with Gaussian profile
                kernel_size = kernel_length + 4  # Padding for rotation
                kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
                center = kernel_size // 2
                
                # Create Gaussian-weighted line
                sigma = kernel_length / 6.0
                for i in range(kernel_length):
                    pos = i - kernel_length // 2
                    weight = np.exp(-pos**2 / (2 * sigma**2))
                    x = center
                    y = center + pos
                    if 0 <= y < kernel_size:
                        kernel[y, x] = weight
                
                # Rotate kernel
                M = cv2.getRotationMatrix2D((center, center), angle, 1)
                rotated_kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
                
                # Normalize kernel
                kernel_sum = np.sum(rotated_kernel)
                if kernel_sum > 0:
                    rotated_kernel = rotated_kernel / kernel_sum
                
                # Apply directional filter
                response = cv2.filter2D(tophat, cv2.CV_32F, rotated_kernel)
                
                # Non-maximum suppression in perpendicular direction
                nms_kernel = np.zeros((5, 5), dtype=np.float32)
                nms_kernel[2, :] = 1.0
                M_nms = cv2.getRotationMatrix2D((2, 2), angle + 90, 1)
                nms_kernel_rot = cv2.warpAffine(nms_kernel, M_nms, (5, 5))
                nms_response = cv2.filter2D(response, cv2.CV_32F, nms_kernel_rot)
                
                # Update scratch map with maximum response
                scratch_map = np.maximum(scratch_map, response)
        
        # Resize back to original size if needed
        if scale != 1.0:
            scratch_map = cv2.resize(scratch_map, (image.shape[1], image.shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
        
        all_scratch_maps.append(scratch_map)
    
    # Combine multi-scale results
    combined_scratch_map = np.mean(all_scratch_maps, axis=0)
    
    # Normalize
    combined_scratch_map = cv2.normalize(combined_scratch_map, None, 0, 255, 
                                       cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Advanced thresholding with Otsu's method
    _, otsu_thresh = cv2.threshold(combined_scratch_map, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Adaptive threshold for local variations
    adaptive_thresh = cv2.adaptiveThreshold(combined_scratch_map, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 15, -2)
    
    # Combine both thresholding methods
    combined_binary = cv2.bitwise_or(otsu_thresh, adaptive_thresh)
    
    # Apply zone mask
    result = cv2.bitwise_and(combined_binary, zone_mask)
    
    # Morphological operations to connect scratch fragments
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_connect)
    
    # Remove small non-linear components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        
        # Keep only linear structures
        if area < 10 or aspect_ratio < 2.5:
            result[labels == i] = 0
    
    return result

def validate_defects(defect_mask: np.ndarray, original_image: np.ndarray, 
                    zone_mask: np.ndarray, min_contrast: float = 10,
                    zone_name: str = "") -> np.ndarray:
    """
    Enhanced defect validation with zone-specific rules and texture analysis
    """
    validated_mask = np.zeros_like(defect_mask)
    
    # Create zone boundary exclusion mask
    boundary_exclusion_mask = create_boundary_exclusion_mask(zone_mask, localization_data)
    
    # Remove defects on zone boundaries
    defect_mask_cleaned = cv2.bitwise_and(defect_mask, cv2.bitwise_not(boundary_exclusion_mask))
    # Zone-specific validation parameters
    zone_params = {
        "Core": {"min_contrast": 15, "min_area": 3, "texture_threshold": 0.8},
        "Cladding": {"min_contrast": 10, "min_area": 5, "texture_threshold": 0.6},
        "default": {"min_contrast": 10, "min_area": 5, "texture_threshold": 0.5}
    }
    
    params = zone_params.get(zone_name, zone_params["default"])
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask_cleaned, connectivity=8)
    
    for i in range(1, num_labels):
        # Get component mask
        component_mask = (labels == i).astype(np.uint8) * 255
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Skip very small components
        if area < params["min_area"]:
            continue
            
        # Calculate local contrast
        defect_pixels = original_image[component_mask > 0]
        if len(defect_pixels) == 0:
            continue
            
        # Enhanced surrounding region analysis
        kernel_size = max(5, int(np.sqrt(area) / 2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(component_mask, kernel)
        surrounding_mask = cv2.bitwise_and(dilated - component_mask, zone_mask)
        surrounding_pixels = original_image[surrounding_mask > 0]
        
        if len(surrounding_pixels) < 10:
            continue
            
        # Calculate multiple validation metrics
        defect_mean = np.mean(defect_pixels)
        defect_std = np.std(defect_pixels)
        surrounding_mean = np.mean(surrounding_pixels)
        surrounding_std = np.std(surrounding_pixels)
        
        # 1. Contrast validation
        contrast = abs(defect_mean - surrounding_mean)
        
        # 2. Texture consistency check
        texture_similarity = 1.0 - abs(defect_std - surrounding_std) / (max(defect_std, surrounding_std) + 1e-6)
        
        # 3. Statistical significance test
        if len(defect_pixels) > 5 and len(surrounding_pixels) > 5:
            # Simple t-test approximation
            pooled_std = np.sqrt((defect_std**2 + surrounding_std**2) / 2)
            t_statistic = abs(defect_mean - surrounding_mean) / (pooled_std + 1e-6)
            is_significant = t_statistic > 2.0  # Approximately 95% confidence
        else:
            is_significant = contrast > params["min_contrast"]
        
        # 4. Shape analysis for scratches vs noise
        if area > 10:
            perimeter = cv2.arcLength(np.argwhere(component_mask > 0), True)
            compactness = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            is_scratch_like = compactness < 0.3  # Elongated shape
        else:
            is_scratch_like = False
        
        # Validate based on combined criteria
        if zone_name == "Core":
            # Stricter validation for core zone
            is_valid = (contrast >= params["min_contrast"] * 1.5 and is_significant) or \
                      (is_scratch_like and contrast >= params["min_contrast"])
        else:
            # Regular validation for other zones
            is_valid = (contrast >= params["min_contrast"] and 
                       (is_significant or texture_similarity < params["texture_threshold"])) or \
                      is_scratch_like
        
        if is_valid:
            validated_mask = cv2.bitwise_or(validated_mask, component_mask)
            
    return validated_mask

def create_boundary_exclusion_mask(zone_mask: np.ndarray, 
                                  localization_data: Optional[Dict[str, Any]] = None,
                                  boundary_width: int = 3) -> np.ndarray:
    """
    Creates a mask to exclude zone boundaries from defect detection.
    """
    if localization_data is None:
        return np.zeros_like(zone_mask)
    
    h, w = zone_mask.shape
    exclusion_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Get core and cladding parameters
    cladding_center = localization_data.get('cladding_center_xy')
    core_radius = localization_data.get('core_radius_px', 0)
    cladding_radius = localization_data.get('cladding_radius_px', 0)
    
    if cladding_center and core_radius > 0:
        cx, cy = cladding_center
        
        # Create exclusion rings around zone boundaries
        # Core-Cladding boundary
        cv2.circle(exclusion_mask, (int(cx), int(cy)), 
                  int(core_radius + boundary_width), 255, -1)
        cv2.circle(exclusion_mask, (int(cx), int(cy)), 
                  int(max(0, core_radius - boundary_width)), 0, -1)
        
        # Cladding outer boundary
        cv2.circle(exclusion_mask, (int(cx), int(cy)), 
                  int(cladding_radius + boundary_width), 255, -1)
        cv2.circle(exclusion_mask, (int(cx), int(cy)), 
                  int(max(0, cladding_radius - boundary_width)), 0, -1)
    
    return exclusion_mask

def matrix_variance_detection(image: np.ndarray, zone_mask: np.ndarray,
                            variance_threshold: float = 15.0,
                            local_window_size: int = 3) -> np.ndarray:
    """
    Divides image into 9 segments and detects anomalies based on local pixel variance.
    """
    h, w = image.shape[:2]
    result_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define 9 segments (3x3 grid)
    segment_h = h // 3
    segment_w = w // 3
    
    for row in range(3):
        for col in range(3):
            # Calculate segment boundaries
            y_start = row * segment_h
            y_end = (row + 1) * segment_h if row < 2 else h
            x_start = col * segment_w
            x_end = (col + 1) * segment_w if col < 2 else w
            
            # Extract segment
            segment = image[y_start:y_end, x_start:x_end]
            segment_zone_mask = zone_mask[y_start:y_end, x_start:x_end]
            
            # Only process pixels within the zone
            if np.sum(segment_zone_mask) == 0:
                continue
            
            # Analyze each pixel in the segment
            seg_h, seg_w = segment.shape
            half_window = local_window_size // 2
            
            for y in range(half_window, seg_h - half_window):
                for x in range(half_window, seg_w - half_window):
                    if segment_zone_mask[y, x] == 0:
                        continue
                    
                    # Get local neighborhood
                    local_region = segment[y-half_window:y+half_window+1,
                                         x-half_window:x+half_window+1]
                    
                    # Calculate local statistics
                    center_value = float(segment[y, x])
                    local_mean = np.mean(local_region)
                    local_std = np.std(local_region)
                    
                    # Check for significant variance
                    if local_std > 0:
                        # Calculate how many standard deviations the center pixel is from local mean
                        z_score = abs(center_value - local_mean) / local_std
                        
                        # Also check absolute difference
                        abs_diff = abs(center_value - local_mean)
                        
                        # Mark as anomaly if high variance
                        if z_score > 2.0 or abs_diff > variance_threshold:
                            # Convert back to full image coordinates
                            full_y = y_start + y
                            full_x = x_start + x
                            result_mask[full_y, full_x] = 255
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel)
    
    return result_mask

def detect_defects(
    processed_image: np.ndarray,
    zone_mask: np.ndarray,
    zone_name: str, 
    profile_config: Dict[str, Any],
    global_algo_params: Dict[str, Any],
    localization_data: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advanced defect detection with intelligent fusion and validation
    """
    if processed_image is None or zone_mask is None:
        return np.zeros_like(processed_image), np.zeros_like(processed_image, dtype=np.float32)
    
    if np.sum(zone_mask) == 0:
        return np.zeros_like(processed_image), np.zeros_like(processed_image, dtype=np.float32)

    defect_config = profile_config.get("defect_detection", {})
    
    # Get algorithms for this profile
    region_algorithms = defect_config.get("region_algorithms", ["do2mr"])
    linear_algorithms = defect_config.get("linear_algorithms", ["lei_simple"])
    
    # Store individual algorithm results for intelligent fusion
    algorithm_results = {}
    algorithm_confidences = {}
    
    # Apply pre-filtering to reduce noise
    preprocessed = cv2.bilateralFilter(processed_image, 5, 50, 50)
    
    # Initialize combined results
    combined_mask = np.zeros_like(processed_image, dtype=np.uint8)
    combined_confidence = np.zeros_like(processed_image, dtype=np.float32)
    
    # Run region-based algorithms
    for algo in region_algorithms:
        if algo == "do2mr":
            mask, conf = do2mr_detection(preprocessed, zone_mask, zone_name, global_algo_params)
            algorithm_results[algo] = mask
            algorithm_confidences[algo] = conf
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            combined_confidence = np.maximum(combined_confidence, conf)
    
    # Run scratch detection algorithms
    for algo in linear_algorithms:
        if algo in ["lei_simple", "lei_advanced"]:
            scratch_mask = lei_scratch_detection(preprocessed, zone_mask, global_algo_params)
            algorithm_results[algo] = scratch_mask
            algorithm_confidences[algo] = scratch_mask.astype(np.float32) / 255.0
            combined_mask = cv2.bitwise_or(combined_mask, scratch_mask)
            combined_confidence = np.maximum(combined_confidence, scratch_mask.astype(np.float32) / 255.0)
    
    # Intelligent fusion based on consensus
    num_algorithms = len(algorithm_results)
    if num_algorithms == 0:
        return np.zeros_like(processed_image), np.zeros_like(processed_image, dtype=np.float32)
    
    # Create consensus map
    consensus_map = np.zeros_like(processed_image, dtype=np.float32)
    
    for algo_name, result in algorithm_results.items():
        weight = defect_config.get("algorithm_weights", {}).get(algo_name, 1.0)
        consensus_map += (result > 0).astype(np.float32) * weight
    
    # Normalize consensus
    total_weight = sum(defect_config.get("algorithm_weights", {}).get(algo, 1.0) 
                      for algo in algorithm_results.keys())
    if total_weight > 0:
        consensus_map = consensus_map / total_weight
    
    # Dynamic thresholding based on zone
    if zone_name == "Core":
        # More conservative for core zone
        consensus_threshold = 0.6
    else:
        consensus_threshold = 0.4
    
    # Apply consensus threshold
    consensus_mask = (consensus_map >= consensus_threshold).astype(np.uint8) * 255
    
    # Post-processing to remove artifacts
    # Remove isolated pixels
    kernel_median = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    consensus_mask = cv2.medianBlur(consensus_mask, 3)
    
    # Fill small holes in defects
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    consensus_mask = cv2.morphologyEx(consensus_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Enhanced validation with zone context
    validated_mask = validate_defects(consensus_mask, processed_image, zone_mask, 
                                    min_contrast=10, zone_name=zone_name)
    
    # Create final confidence map
    final_confidence = np.zeros_like(processed_image, dtype=np.float32)
    for algo_name, conf in algorithm_confidences.items():
        weight = defect_config.get("algorithm_weights", {}).get(algo_name, 1.0)
        final_confidence += conf * weight
    if total_weight > 0:
        final_confidence = final_confidence / total_weight
    
    # Apply validation mask to confidence
    final_confidence = final_confidence * (validated_mask > 0).astype(np.float32)
    
    matrix_mask = matrix_variance_detection(processed_image, zone_mask,
                                          variance_threshold=global_algo_params.get('matrix_variance_threshold', 15.0))
    combined_mask = cv2.bitwise_or(combined_mask, matrix_mask)
    matrix_confidence = matrix_mask.astype(np.float32) / 255.0 * 0.8  # 80% confidence weight
    combined_confidence = np.maximum(combined_confidence, matrix_confidence)
    
    # Validate defects before returning with zone boundary exclusion
    validated_mask = validate_defects(combined_mask, processed_image, zone_mask,
                                    localization_data=localization_data)
    
    return validated_mask, combined_confidence



# Test function for module validation
def run_basic_tests():
    """
    Runs basic tests to validate the image processing functions.
    """
    logging.info("\n=== Running Image Processing Module Tests ===")
    
    # Create a dummy test image
    test_image_path_str = "sample_fiber_image.png"
    if not Path(test_image_path_str).exists():
        # Create a simple test image
        dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(dummy_image, (100, 100), 80, (128, 128, 128), -1)  # Cladding
        cv2.circle(dummy_image, (100, 100), 6, (64, 64, 64), -1)     # Core (smaller for single-mode)
        cv2.imwrite(test_image_path_str, dummy_image)
        logging.info(f"Created dummy test image: {test_image_path_str}")
    
    # Create dummy configuration
    try:
        dummy_profile_config_main_test = get_config()["processing_profiles"]["deep_inspection"]
    except:
        dummy_profile_config_main_test = {
            "preprocessing": {
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": [8, 8],
                "gaussian_blur_kernel_size": [5, 5]
            },
            "localization": {
                "hough_dp": 1.2,
                "hough_min_dist_factor": 0.15,
                "hough_param1": 70,
                "hough_param2": 35,
                "hough_min_radius_factor": 0.08,
                "hough_max_radius_factor": 0.45
            },
            "defect_detection": {
                "region_algorithms": ["do2mr"],
                "linear_algorithms": ["lei_simple"],
                "min_defect_area_px": 5
            }
        }
    
    dummy_global_algo_params_main_test = {
        "do2mr_kernel_size": 5,
        "do2mr_gamma_default": 1.5,
        "lei_kernel_lengths": [11, 17, 23],
        "lei_angle_step_deg": 15
    }
    
    # Test preprocessing
    logging.info(f"\n--- Test Case 1: Load and Preprocess Image: {test_image_path_str} ---")
    preprocess_result = load_and_preprocess_image(test_image_path_str, dummy_profile_config_main_test) 
    
    if preprocess_result: 
        original_bgr_test, gray_test, processed_test = preprocess_result 
        logging.info(f"Image preprocessed. Shape of processed image: {processed_test.shape}")
        
        # Test fiber localization
        logging.info("\n--- Test Case 2: Locate Fiber Structure ---")
        localization = locate_fiber_structure(processed_test, dummy_profile_config_main_test, original_gray_image=gray_test) 
        
        if localization: 
            logging.info(f"Fiber Localization: {localization}")
            
            # Test zone mask generation
            logging.info("\n--- Test Case 3: Generate Zone Masks ---")
            dummy_zone_defs_main_test = [
                {"name": "Core", "type": "core"},
                {"name": "Cladding", "type": "cladding"}
            ]
            
            um_per_px_test = 0.5 
            user_core_diam_test = 9.0 
            user_cladding_diam_test = 125.0 
            
            zone_masks_generated = generate_zone_masks( 
                processed_test.shape, localization, dummy_zone_defs_main_test,
                um_per_px=um_per_px_test, 
                user_core_diameter_um=user_core_diam_test, 
                user_cladding_diameter_um=user_cladding_diam_test
            )
            
            if zone_masks_generated: 
                logging.info(f"Generated masks for zones: {list(zone_masks_generated.keys())}")
                
                # Test defect detection
                logging.info("\n--- Test Case 4: Detect Defects (Iterating Zones) ---")
                
                for zone_name_test, zone_mask_test in zone_masks_generated.items():
                    if np.sum(zone_mask_test) == 0:
                        logging.info(f"Skipping defect detection for empty zone: {zone_name_test}")
                        continue
                    
                    logging.info(f"--- Detecting defects in Zone: {zone_name_test} ---")
                    defects_mask, conf_map = detect_defects( 
                        processed_test, zone_mask_test, zone_name_test, 
                        dummy_profile_config_main_test, dummy_global_algo_params_main_test
                    )
                    logging.info(f"Defect detection in '{zone_name_test}' zone complete. Found {np.sum(defects_mask > 0)} defect pixels.")
            else: 
                logging.warning("Zone mask generation failed for defect detection test.")
        else: 
            logging.error("Fiber localization failed. Cannot proceed with further tests.")
    else: 
        logging.error("Image preprocessing failed.")

    # Clean up test image
    if Path(test_image_path_str).exists() and test_image_path_str == "sample_fiber_image.png":
        try:
            Path(test_image_path_str).unlink()
            logging.info(f"Cleaned up dummy image: {test_image_path_str}")
        except OSError as e_os_error:
            logging.error(f"Error removing dummy image {test_image_path_str}: {e_os_error}")

    logging.info("=== Image Processing Module Tests Complete ===\n")

if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    # Run tests
    run_basic_tests()