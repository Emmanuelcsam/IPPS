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
# Attempt to import the compiled C++ accelerator module.
# If it's not found, the pure Python implementations will be used as a fallback.
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


# Stub functions for unavailable algorithms
def _anomalib_full_detection_stub(image, zone_mask, profile_config):
    """Stub for unavailable Anomalib integration."""
    return np.zeros_like(image, dtype=np.uint8), np.zeros_like(image, dtype=np.float32)

def _padim_specific_detection_stub(image, zone_mask, profile_config):
    """Stub for unavailable PaDiM specific detection."""
    return np.zeros_like(image, dtype=np.uint8), np.zeros_like(image, dtype=np.float32)

def _segdecnet_detection_stub(image, zone_mask, profile_config):
    """Stub for unavailable SegDecNet detection."""
    return np.zeros_like(image, dtype=np.uint8), np.zeros_like(image, dtype=np.float32)

def _advanced_scratch_detection_stub(image, zone_mask, profile_config):
    """Stub for unavailable advanced scratch detection."""
    return np.zeros_like(image, dtype=np.uint8)

def _anomaly_detection_stub(image, zone_mask, profile_config):
    """Stub for unavailable anomaly detection."""
    return np.zeros_like(image, dtype=np.uint8), np.zeros_like(image, dtype=np.float32)

def _do2mr_detection_stub(image, zone_mask, zone_name, global_algo_params):
    """Stub for DO2MR detection when accelerator is not available."""
    return np.zeros_like(image, dtype=np.uint8), np.zeros_like(image, dtype=np.float32)

def _gabor_defect_detection_stub(image, zone_mask, global_algo_params):
    """Stub for Gabor defect detection."""
    return np.zeros_like(image, dtype=np.uint8)

def _multiscale_defect_detection_stub(image, zone_mask, global_algo_params):
    """Stub for multi-scale defect detection."""
    return np.zeros_like(image, dtype=np.uint8)

def _lei_scratch_detection_stub(image, zone_mask, global_algo_params):
    """Stub for LEI scratch detection."""
    return np.zeros_like(image, dtype=np.uint8)

def _wavelet_defect_detection_stub(image, zone_mask, global_algo_params):
    """Stub for wavelet defect detection."""
    return np.zeros_like(image, dtype=np.uint8)


# Assign detection functions based on availability
if ANOMALIB_FULL_AVAILABLE:
    _anomalib_full_detection = lambda img, mask, cfg: AnomalibDefectDetector().detect_defects(img, mask, cfg)
else:
    _anomalib_full_detection = _anomalib_full_detection_stub

if PADIM_SPECIFIC_AVAILABLE:
    _padim_specific_detection = lambda img, mask, cfg: FiberPaDiM().detect_defects(img, mask, cfg)
else:
    _padim_specific_detection = _padim_specific_detection_stub

if SEGDECNET_AVAILABLE:
    _segdecnet_detection = lambda img, mask, cfg: FiberSegDecNet().detect_defects(img, mask, cfg)
else:
    _segdecnet_detection = _segdecnet_detection_stub

if ADVANCED_SCRATCH_AVAILABLE:
    _advanced_scratch_detection = lambda img, mask, cfg: AdvancedScratchDetector().detect_scratches(img, mask, cfg)
else:
    _advanced_scratch_detection = _advanced_scratch_detection_stub

if ANOMALY_DETECTION_AVAILABLE:
    _anomaly_detection = lambda img, mask, cfg: AnomalyDetector().detect_anomalies(img, mask, cfg)
else:
    _anomaly_detection = _anomaly_detection_stub

# Assign remaining algorithm stubs regardless of availability
if CPP_ACCELERATOR_AVAILABLE:
    # Use C++ accelerated implementations when available
    def _do2mr_detection(image, zone_mask, zone_name, global_algo_params):
        """C++ accelerated DO2MR detection."""
        try:
            return accelerator.do2mr_detect(image, zone_mask, zone_name, global_algo_params)
        except Exception as e:
            logging.warning(f"C++ DO2MR failed, falling back to stub: {e}")
            return _do2mr_detection_stub(image, zone_mask, zone_name, global_algo_params)
    
    # Assign Gabor detection C++ function if available
    _gabor_defect_detection = lambda img, mask, params: accelerator.gabor_detect(img, mask, params)
    # Assign multi-scale detection C++ function if available  
    _multiscale_defect_detection = lambda img, mask, params: accelerator.multiscale_detect(img, mask, params)
    # Assign LEI scratch detection C++ function if available
    _lei_scratch_detection = lambda img, mask, params: accelerator.lei_scratch_detect(img, mask, params)
    # Assign wavelet detection C++ function if available
    _wavelet_defect_detection = lambda img, mask, params: accelerator.wavelet_detect(img, mask, params)
else:
    # Assign DO2MR stub function to module-level name
    _do2mr_detection = _do2mr_detection_stub
    # Assign Gabor detection stub to module-level name
    _gabor_defect_detection = _gabor_defect_detection_stub
    # Assign multi-scale detection stub to module-level name
    _multiscale_defect_detection = _multiscale_defect_detection_stub
    # Assign LEI scratch detection stub to module-level name
    _lei_scratch_detection = _lei_scratch_detection_stub
    # Assign advanced scratch detection stub to module-level name
    _advanced_scratch_detection = _advanced_scratch_detection_stub
    # Assign wavelet detection stub to module-level name
    _wavelet_defect_detection = _wavelet_defect_detection_stub

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
    # Corrected cv2.add with NumPy addition for safety as per Probs.txt
    corrected_int16 = cv2.subtract(gray_image.astype(np.int16), background.astype(np.int16))
    corrected_int16 = corrected_int16 + 128  # Shift to mid-gray
    # Clip and convert back to original dtype (passed or default uint8)
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
        # Better estimation based on fiber type
        core_radius = cladding_radius * 0.35  # More realistic ratio for single-mode
    
    return tuple(cladding_center), core_radius * 2


def locate_fiber_structure(
    processed_image: np.ndarray,
    profile_config: Dict[str, Any],
    original_gray_image: Optional[np.ndarray] = None
) -> Optional[Dict[str, Any]]:
    """
    Locates the fiber cladding and core using HoughCircles, contour fitting, or circle-fit library.
    Enhanced with improved core detection methods.

    Args:
        processed_image: The preprocessed grayscale image (e.g., after CLAHE and Gaussian blur).
        profile_config: The specific processing profile sub-dictionary from the main config.
        original_gray_image: The original grayscale image, primarily for core detection if available.

    Returns:
        A dictionary containing localization data or None if localization fails.
    """
    # Get localization parameters from the profile configuration.
    loc_params = profile_config.get("localization", {})
    # Get image height (h) and width (w).
    h, w = processed_image.shape[:2]
    # Determine the smaller dimension of the image.
    min_img_dim = min(h, w)

    # --- Initialize Parameters for HoughCircles ---
    # dp: Inverse ratio of accumulator resolution.
    dp = loc_params.get("hough_dp", 1.2)
    # minDist: Minimum distance between centers of detected circles (factor of min_img_dim).
    min_dist_circles = int(min_img_dim * loc_params.get("hough_min_dist_factor", 0.15))
    # param1: Upper Canny threshold for internal edge detection in HoughCircles.
    param1 = loc_params.get("hough_param1", 70)
    # param2: Accumulator threshold for circle centers at the detection stage.
    param2 = loc_params.get("hough_param2", 35)
    # minRadius: Minimum circle radius to detect (factor of min_img_dim).
    min_radius_hough = int(min_img_dim * loc_params.get("hough_min_radius_factor", 0.08))
    # maxRadius: Maximum circle radius to detect (factor of min_img_dim).
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
        localization_result['core_radius_px'] = localization_result['cladding_radius_px'] * 0.35
        logging.warning("Core detection failed, using fallback estimation")

    # --- Adhesive Layer Detection ---
    if 'core_center_xy' in localization_result and 'cladding_center_xy' in localization_result:
        try:
            # Detect adhesive layer between core and cladding
            cl_cx_core, cl_cy_core = localization_result['cladding_center_xy']
            core_radius = localization_result['core_radius_px']
            
            # Create mask for the region between core and cladding
            adhesive_search_mask = np.zeros_like(image_for_core_detect, dtype=np.uint8)
            cv2.circle(adhesive_search_mask, (cl_cx_core, cl_cy_core), int(cladding_radius * 0.95), 255, -1)
            cv2.circle(adhesive_search_mask, (cl_cx_core, cl_cy_core), int(core_radius * 1.05), 0, -1)
            
            # Look for adhesive layer characteristics
            masked_adhesive_region = cv2.bitwise_and(image_for_core_detect, image_for_core_detect, mask=adhesive_search_mask)
            
            # Adhesive often appears as a ring with different intensity
            hist = cv2.calcHist([masked_adhesive_region], [0], adhesive_search_mask, [256], [0, 256])
            
            # Find peaks in histogram (adhesive layer often has distinct intensity)
            if hist is not None and len(hist) > 0:
                # Simple peak detection for adhesive layer
                adhesive_intensity_peaks = []
                for i in range(10, 246):  # Avoid edges
                    if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist) * 0.5:
                        adhesive_intensity_peaks.append(i)
                
                if adhesive_intensity_peaks:
                    # Store adhesive layer information
                    localization_result['adhesive_detected'] = True
                    localization_result['adhesive_intensity_range'] = adhesive_intensity_peaks
                    logging.info(f"Adhesive layer detected with intensity peaks at: {adhesive_intensity_peaks}")
        except Exception as e:
            logging.warning(f"Adhesive layer detection failed: {e}")

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
    Generates binary masks for each inspection zone based on IEC standards and detected fiber.

    Args:
        image_shape: (height, width) of the image.
        localization_data: Dictionary from locate_fiber_structure.
        zone_definitions: List of zone definition dicts from config (e.g., for 'single_mode_pc').
        um_per_px: Current image's microns-per-pixel scale, if available.
        user_core_diameter_um: User-provided core diameter (for scaling relative zones).
        user_cladding_diameter_um: User-provided cladding diameter (for scaling relative zones).

    Returns:
        A dictionary where keys are zone names and values are binary mask (np.ndarray).
    """
    masks: Dict[str, np.ndarray] = {}
    h, w = image_shape[:2]
    Y, X = np.ogrid[:h, :w]

    # Get detected fiber parameters
    cladding_center = localization_data.get('cladding_center_xy')
    core_center = localization_data.get('core_center_xy', cladding_center)
    core_radius_px_detected = localization_data.get('core_radius_px')
    
    detected_cladding_radius_px = localization_data.get('cladding_radius_px')
    cladding_ellipse_params = localization_data.get('cladding_ellipse_params')

    if cladding_center is None:
        logging.error("Cannot generate zone masks: Cladding center not localized.")
        return masks

    cx, cy = cladding_center
    core_cx, core_cy = core_center if core_center else (cx, cy)

    # Calculate distance maps
    dist_sq_from_cladding = (X - cx)**2 + (Y - cy)**2
    dist_sq_from_core = (X - core_cx)**2 + (Y - core_cy)**2

    for zone_def in zone_definitions:
        zone_name = zone_def["name"]
        zone_type = zone_def["type"]
        
        try:
            if zone_type == "core":
                # Core zone mask
                if user_core_diameter_um and um_per_px:
                    # Use user-provided core diameter
                    core_radius_px = (user_core_diameter_um / 2) / um_per_px
                elif core_radius_px_detected and core_radius_px_detected > 0:
                    # Use detected core radius
                    core_radius_px = core_radius_px_detected
                else:
                    # Fallback: estimate from cladding
                    core_radius_px = detected_cladding_radius_px * 0.35 if detected_cladding_radius_px else 20
                
                masks[zone_name] = (dist_sq_from_core <= core_radius_px**2).astype(np.uint8) * 255
                logging.debug(f"Core zone '{zone_name}' created with radius {core_radius_px:.1f}px")

            elif zone_type == "cladding":
                # Cladding zone (excluding core)
                if user_cladding_diameter_um and um_per_px:
                    cladding_radius_px = (user_cladding_diameter_um / 2) / um_per_px
                elif detected_cladding_radius_px:
                    cladding_radius_px = detected_cladding_radius_px
                else:
                    logging.error(f"Cannot create cladding zone '{zone_name}': No cladding radius available")
                    continue

                # Get core radius for exclusion
                if user_core_diameter_um and um_per_px:
                    core_radius_px = (user_core_diameter_um / 2) / um_per_px
                elif core_radius_px_detected and core_radius_px_detected > 0:
                    core_radius_px = core_radius_px_detected
                else:
                    core_radius_px = cladding_radius_px * 0.35

                # Create annular mask (cladding - core)
                cladding_mask = (dist_sq_from_cladding <= cladding_radius_px**2).astype(np.uint8)
                core_mask = (dist_sq_from_core <= core_radius_px**2).astype(np.uint8)
                masks[zone_name] = (cladding_mask - core_mask) * 255
                logging.debug(f"Cladding zone '{zone_name}' created as annulus: outer={cladding_radius_px:.1f}px, inner={core_radius_px:.1f}px")

            elif zone_type == "adhesive":
                # Adhesive zone around cladding
                inner_factor = zone_def.get("inner_radius_factor", 1.0)
                outer_factor = zone_def.get("outer_radius_factor", 1.15)
                
                if detected_cladding_radius_px:
                    inner_radius_px = detected_cladding_radius_px * inner_factor
                    outer_radius_px = detected_cladding_radius_px * outer_factor
                    
                    outer_mask = (dist_sq_from_cladding <= outer_radius_px**2).astype(np.uint8)
                    inner_mask = (dist_sq_from_cladding <= inner_radius_px**2).astype(np.uint8)
                    masks[zone_name] = (outer_mask - inner_mask) * 255
                    logging.debug(f"Adhesive zone '{zone_name}' created: inner={inner_radius_px:.1f}px, outer={outer_radius_px:.1f}px")
                else:
                    logging.error(f"Cannot create adhesive zone '{zone_name}': No cladding radius available")

            elif zone_type == "custom_radius":
                # Custom radius zone
                radius_um = zone_def.get("radius_um")
                if radius_um and um_per_px:
                    radius_px = radius_um / um_per_px
                    masks[zone_name] = (dist_sq_from_cladding <= radius_px**2).astype(np.uint8) * 255
                    logging.debug(f"Custom radius zone '{zone_name}' created with {radius_px:.1f}px")
                else:
                    logging.error(f"Cannot create custom radius zone '{zone_name}': Missing radius_um or um_per_px")

        except Exception as e:
            logging.error(f"Error creating zone mask '{zone_name}': {e}")
            continue

    return masks


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
    # Validate inputs
    if processed_image is None or zone_mask is None:
        logging.error(f"Invalid input to detect_defects: processed_image or zone_mask is None")
        return np.zeros_like(processed_image), np.zeros_like(processed_image, dtype=np.float32)
    
    if np.sum(zone_mask) == 0:
        logging.debug(f"Defect detection skipped for empty zone mask in zone '{zone_name}'.")
        return np.zeros_like(processed_image), np.zeros_like(processed_image, dtype=np.float32)

    defect_config = profile_config.get("defect_detection", {})
    enabled_algorithms = defect_config.get("enabled_algorithms", ["basic_threshold"])
    
    # Initialize result arrays
    combined_defect_mask = np.zeros_like(processed_image, dtype=np.uint8)
    combined_confidence_map = np.zeros_like(processed_image, dtype=np.float32)
    
    # Apply zone mask to input image
    masked_image = cv2.bitwise_and(processed_image, processed_image, mask=zone_mask)
    
    logging.debug(f"Running defect detection in zone '{zone_name}' with algorithms: {enabled_algorithms}")
    
    # Algorithm execution
    for algorithm in enabled_algorithms:
        try:
            if algorithm == "basic_threshold":
                defect_mask, confidence = _basic_threshold_detection(masked_image, defect_config)
            elif algorithm == "do2mr" and CPP_ACCELERATOR_AVAILABLE:
                defect_mask, confidence = _do2mr_detection(masked_image, zone_mask, zone_name, global_algo_params)
            elif algorithm == "gabor":
                defect_mask = _gabor_defect_detection(masked_image, zone_mask, global_algo_params)
                confidence = defect_mask.astype(np.float32) / 255.0
            elif algorithm == "multiscale":
                defect_mask = _multiscale_defect_detection(masked_image, zone_mask, global_algo_params)
                confidence = defect_mask.astype(np.float32) / 255.0
            elif algorithm == "lei_scratch":
                defect_mask = _lei_scratch_detection(masked_image, zone_mask, global_algo_params)
                confidence = defect_mask.astype(np.float32) / 255.0
            elif algorithm == "advanced_scratch":
                defect_mask = _advanced_scratch_detection(masked_image, zone_mask, defect_config)
                confidence = defect_mask.astype(np.float32) / 255.0
            elif algorithm == "wavelet":
                defect_mask = _wavelet_defect_detection(masked_image, zone_mask, global_algo_params)
                confidence = defect_mask.astype(np.float32) / 255.0
            elif algorithm == "anomalib_full":
                defect_mask, confidence = _anomalib_full_detection(masked_image, zone_mask, defect_config)
            elif algorithm == "padim_specific":
                defect_mask, confidence = _padim_specific_detection(masked_image, zone_mask, defect_config)
            elif algorithm == "segdecnet":
                defect_mask, confidence = _segdecnet_detection(masked_image, zone_mask, defect_config)
            elif algorithm == "anomaly_detection":
                defect_mask, confidence = _anomaly_detection(masked_image, zone_mask, defect_config)
            else:
                logging.warning(f"Unknown algorithm '{algorithm}' in zone '{zone_name}', skipping")
                continue
            
            # Combine results using logical OR for masks and maximum for confidence
            combined_defect_mask = cv2.bitwise_or(combined_defect_mask, defect_mask)
            combined_confidence_map = np.maximum(combined_confidence_map, confidence)
            
            logging.debug(f"Algorithm '{algorithm}' found {np.sum(defect_mask > 0)} defect pixels in zone '{zone_name}'")
            
        except Exception as e:
            logging.error(f"Error in algorithm '{algorithm}' for zone '{zone_name}': {e}")
            continue
    
    # Apply zone mask to final results
    combined_defect_mask = cv2.bitwise_and(combined_defect_mask, zone_mask)
    combined_confidence_map = combined_confidence_map * (zone_mask.astype(np.float32) / 255.0)
    
    logging.debug(f"Combined defect detection in zone '{zone_name}' found {np.sum(combined_defect_mask > 0)} total defect pixels")
    
    return combined_defect_mask, combined_confidence_map


def _basic_threshold_detection(image: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basic threshold-based defect detection.
    """
    threshold_value = config.get("basic_threshold_value", 0)
    threshold_type = config.get("basic_threshold_type", "auto")
    
    if threshold_type == "auto":
        # Use Otsu's method for automatic thresholding
        _, defect_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert if needed (defects are typically darker)
        if np.mean(image[defect_mask > 0]) > np.mean(image[defect_mask == 0]):
            defect_mask = cv2.bitwise_not(defect_mask)
    else:
        # Use manual threshold
        _, defect_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Create confidence map based on distance from threshold
    confidence_map = np.abs(image.astype(np.float32) - threshold_value) / 255.0
    confidence_map = np.clip(confidence_map, 0, 1)
    
    return defect_mask, confidence_map


def _scratch_detection_lei(image: np.ndarray) -> np.ndarray:
    """
    Linear Edge Intensity (LEI) based scratch detection.
    """
    # Apply different morphological operations to detect linear features
    scratch_map_combined = np.zeros_like(image, dtype=np.uint8)
    
    # Ridge filter for detecting linear features
    try:
        from skimage.filters import sato, frangi, hessian
        ridge_response = sato(image, sigmas=range(1, 4), black_ridges=True)
        if np.any(ridge_response):
            ridge_response_norm = cv2.normalize(ridge_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            _, ridge_mask = cv2.threshold(ridge_response_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            scratch_map_combined = cv2.bitwise_or(scratch_map_combined, ridge_mask)
    except ImportError:
        # Fallback to basic morphological operations
        pass
    except np.linalg.LinAlgError:
        pass 

    if np.any(ridge_response):
        ridge_response_norm = cv2.normalize(ridge_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, ridge_mask = cv2.threshold(ridge_response_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        scratch_map_combined = cv2.bitwise_or(scratch_map_combined, ridge_mask)
    
    # Black-hat morphology for line detection
    kernel_bh_rect_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)) 
    kernel_bh_rect_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)) 
    blackhat_v = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel_bh_rect_vertical)
    blackhat_h = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel_bh_rect_horizontal)
    blackhat_combined = np.maximum(blackhat_v, blackhat_h)

    if np.any(blackhat_combined):
        _, bh_thresh = cv2.threshold(blackhat_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        scratch_map_combined = cv2.bitwise_or(scratch_map_combined, bh_thresh)
    
    # Hough line detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3) 
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=7) 
    
    if lines is not None:
        line_mask = np.zeros_like(image, dtype=np.uint8)
        for line_segment in lines:
            x1, y1, x2, y2 = line_segment[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1) 
        scratch_map_combined = cv2.bitwise_or(scratch_map_combined, line_mask)
    
    # Clean up results
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    scratch_map_combined = cv2.morphologyEx(scratch_map_combined, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    scratch_map_combined = cv2.morphologyEx(scratch_map_combined, cv2.MORPH_CLOSE, kernel_clean, iterations=1)
    
    return scratch_map_combined


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
        cv2.circle(dummy_image, (100, 100), 30, (64, 64, 64), -1)     # Core
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
                "enabled_algorithms": ["basic_threshold"],
                "basic_threshold_value": 100
            }
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
                {"name": "Cladding", "type": "cladding"},
                {"name": "Adhesive", "type": "adhesive", "inner_radius_factor": 1.0, "outer_radius_factor": 1.15}
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
                dummy_global_algo_params_main_test = {"test_param": "test_value"}
                
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