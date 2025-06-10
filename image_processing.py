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
    Improved core detection with multiple fallback methods
    """
    import cv2
    import numpy as np
    
    # Create cladding mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, tuple(map(int, cladding_center)), int(cladding_radius * 0.8), 255, -1)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Method 1: Adaptive thresholding
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
    
    # Method 2: Edge-based detection
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
    DO2MR (Difference of Min-Max Ranking) detection implementation.
    """
    if CPP_ACCELERATOR_AVAILABLE:
        try:
            # Try C++ version first
            kernel_size = global_algo_params.get("do2mr_kernel_size", 5)
            gamma = global_algo_params.get(f"do2mr_gamma_{zone_name.lower()}", 
                                          global_algo_params.get("do2mr_gamma_default", 1.5))
            result = accelerator.do2mr_detection(image, kernel_size, gamma)
            confidence = result.astype(np.float32) / 255.0
            return result, confidence
        except:
            pass
    
    # Python implementation
    kernel_size = global_algo_params.get("do2mr_kernel_size", 5)
    gamma = global_algo_params.get(f"do2mr_gamma_{zone_name.lower()}", 
                                  global_algo_params.get("do2mr_gamma_default", 1.5))
    
    # Apply zone mask
    masked_image = cv2.bitwise_and(image, image, mask=zone_mask)
    
    # Create kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Max and min filtering
    max_filtered = cv2.dilate(masked_image, kernel)
    min_filtered = cv2.erode(masked_image, kernel)
    
    # Calculate residual
    residual = cv2.subtract(max_filtered, min_filtered)
    
    # Apply median blur to reduce noise
    residual_filtered = cv2.medianBlur(residual, 3)
    
    # Calculate statistics within zone
    zone_pixels = residual_filtered[zone_mask > 0]
    if len(zone_pixels) == 0:
        return np.zeros_like(image), np.zeros_like(image, dtype=np.float32)
    
    mean_val = np.mean(zone_pixels)
    std_val = np.std(zone_pixels)
    
    # Threshold
    threshold = mean_val + gamma * std_val
    _, defect_mask = cv2.threshold(residual_filtered, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply zone mask to result
    defect_mask = cv2.bitwise_and(defect_mask, zone_mask)
    
    # Morphological cleanup
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel_clean)
    
    # Create confidence map
    confidence_map = residual_filtered.astype(np.float32) / 255.0
    confidence_map = cv2.bitwise_and(confidence_map, confidence_map, 
                                    mask=(zone_mask.astype(np.float32) / 255.0).astype(np.uint8))
    
    return defect_mask, confidence_map

def lei_scratch_detection(image: np.ndarray, zone_mask: np.ndarray,
                         global_algo_params: Dict[str, Any]) -> np.ndarray:
    """
    LEI (Linear Enhancement Inspector) scratch detection implementation.
    """
    # Parameters
    kernel_lengths = global_algo_params.get("lei_kernel_lengths", [11, 17, 23])
    angle_step = global_algo_params.get("lei_angle_step_deg", 15)
    
    # Apply zone mask
    masked_image = cv2.bitwise_and(image, image, mask=zone_mask)
    
    # Enhance image contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked_image)
    
    # Initialize result
    scratch_map = np.zeros_like(enhanced, dtype=np.float32)
    
    # Search at multiple orientations
    for angle in range(0, 180, angle_step):
        angle_rad = np.deg2rad(angle)
        
        for kernel_length in kernel_lengths:
            # Create oriented kernel
            kernel = np.zeros((kernel_length, kernel_length), dtype=np.float32)
            center = kernel_length // 2
            
            # Draw line in kernel
            for i in range(kernel_length):
                x = int(center + (i - center) * np.cos(angle_rad))
                y = int(center + (i - center) * np.sin(angle_rad))
                if 0 <= x < kernel_length and 0 <= y < kernel_length:
                    kernel[y, x] = 1.0
                    
            # Normalize kernel
            kernel = kernel / (np.sum(kernel) + 1e-6)
            
            # Apply filter
            response = cv2.filter2D(enhanced, cv2.CV_32F, kernel)
            
            # Update maximum response
            scratch_map = np.maximum(scratch_map, response)
    
    # Normalize and threshold
    scratch_map = cv2.normalize(scratch_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(scratch_map, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Apply zone mask
    result = cv2.bitwise_and(binary, zone_mask)
    
    # Clean up
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_clean)
    
    return result

def validate_defects(defect_mask: np.ndarray, original_image: np.ndarray, 
                    zone_mask: np.ndarray, min_contrast: float = 10) -> np.ndarray:
    """
    Validate detected defects to reduce false positives.
    """
    validated_mask = np.zeros_like(defect_mask)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    
    for i in range(1, num_labels):
        # Get component mask
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # Calculate local contrast
        defect_pixels = original_image[component_mask > 0]
        if len(defect_pixels) == 0:
            continue
            
        # Get surrounding pixels
        dilated = cv2.dilate(component_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        surrounding_mask = cv2.bitwise_and(dilated - component_mask, zone_mask)
        surrounding_pixels = original_image[surrounding_mask > 0]
        
        if len(surrounding_pixels) == 0:
            continue
            
        # Calculate contrast
        defect_mean = np.mean(defect_pixels)
        surrounding_mean = np.mean(surrounding_pixels)
        contrast = abs(defect_mean - surrounding_mean)
        
        # Validate based on contrast
        if contrast >= min_contrast:
            validated_mask = cv2.bitwise_or(validated_mask, component_mask)
            
    return validated_mask

def detect_defects(
    processed_image: np.ndarray,
    zone_mask: np.ndarray,
    zone_name: str, 
    profile_config: Dict[str, Any],
    global_algo_params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced defect detection using multi-algorithm fusion approach.
    """
    if processed_image is None or zone_mask is None:
        return np.zeros_like(processed_image), np.zeros_like(processed_image, dtype=np.float32)
    
    if np.sum(zone_mask) == 0:
        return np.zeros_like(processed_image), np.zeros_like(processed_image, dtype=np.float32)

    defect_config = profile_config.get("defect_detection", {})
    
    # Get algorithms for this profile
    region_algorithms = defect_config.get("region_algorithms", ["do2mr"])
    linear_algorithms = defect_config.get("linear_algorithms", ["lei_simple"])
    
    # Initialize combined results
    combined_mask = np.zeros_like(processed_image, dtype=np.uint8)
    combined_confidence = np.zeros_like(processed_image, dtype=np.float32)
    
    # Run region-based algorithms
    for algo in region_algorithms:
        if algo == "do2mr":
            mask, conf = do2mr_detection(processed_image, zone_mask, zone_name, global_algo_params)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            combined_confidence = np.maximum(combined_confidence, conf)
    
    # Run scratch detection algorithms
    for algo in linear_algorithms:
        if algo in ["lei_simple", "lei_advanced"]:
            scratch_mask = lei_scratch_detection(processed_image, zone_mask, global_algo_params)
            combined_mask = cv2.bitwise_or(combined_mask, scratch_mask)
            combined_confidence = np.maximum(combined_confidence, scratch_mask.astype(np.float32) / 255.0)
    
    # Validate defects before returning
    validated_mask = validate_defects(combined_mask, processed_image, zone_mask)
    
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