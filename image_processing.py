#!/usr/bin/env python3
# image_processing.py

"""
D-Scope Blink: Image Processing Engine
======================================
This module contains the core logic for processing fiber optic end face images.
It includes functions for preprocessing, fiber localization (cladding and core),
zone mask generation, and the multi-algorithm defect detection engine with fusion.
"""
# Missing imports - add these
import pywt  # For wavelet transform
from scipy import ndimage
from skimage import morphology, filters
import cv2 # OpenCV for all core image processing tasks.
import numpy as np # NumPy for numerical and array operations.
from typing import Dict, Any, Optional, List, Tuple # Standard library for type hinting.
import logging # Standard library for logging events.
from pathlib import Path # Standard library for object-oriented path manipulation.
from skimage.feature import local_binary_pattern

# Attempt to import functions from other D-Scope Blink modules.
# These will be fully available when the whole system is assembled.
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
    # Assuming config_loader.py is in the same directory or Python path.
    from config_loader import get_config # Function to access the global configuration.
except ImportError:
    # Fallback for standalone testing if config_loader is not directly available.
    # In a full project, this might load a default or raise a more critical error.
    logging.warning("Could not import get_config from config_loader. Using dummy config for standalone testing.")
    def get_config() -> Dict[str, Any]: # Define a dummy get_config for standalone testing.
        """Returns a dummy configuration for standalone testing."""
        # This is a simplified dummy config. In reality, it would load from config_loader.
        # For testing image_processing.py, ensure relevant keys are present.
        return {
            "algorithm_parameters": {
                "flat_field_image_path": None,
                "morph_gradient_kernel_size": [5,5],
                "black_hat_kernel_size": [11,11],
                "lei_kernel_lengths": [11,17],
                "lei_angle_step_deg": 15,
                "sobel_scharr_ksize": 3, # Used if skeletonization relies on Canny with Sobel/Scharr implicitly
                "skeletonization_dilation_kernel_size": [3,3]
            },
            # Add other keys as needed by functions in this module for standalone testing
        }
    # --- Helper stubs/implementations for all missing functions ---
    # These stubs are defined if config_loader import fails.
    # Later, more complete local versions of these functions are defined,
    # which will overwrite these stubs if this script is run standalone.

    def _do2mr_detection_stub(gray_img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Difference of min-max ranking filtering (DO2MR) to detect region defects.
        Returns a binary mask (0/255). (STUB VERSION)
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        min_filt = cv2.erode(gray_img, kernel)
        max_filt = cv2.dilate(gray_img, kernel)
        residual = cv2.subtract(max_filt, min_filt)
        # Sigma/mean threshold
        zone_vals = residual[residual > 0]
        if zone_vals.size == 0:
            return np.zeros_like(gray_img, dtype=np.uint8)
        mean_res = np.mean(zone_vals)
        std_res = np.std(zone_vals)
        gamma = 1.5
        mask = np.zeros_like(gray_img, dtype=np.uint8)
        mask[(residual - mean_res) > (gamma * std_res)] = 255
        mask = cv2.medianBlur(mask, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return mask

    def _gabor_defect_detection_stub(gray_img: np.ndarray) -> np.ndarray:
        """
        Use Gabor filters to highlight region irregularities.
        Returns a binary mask. (STUB VERSION)
        """
        h, w = gray_img.shape
        accum = np.zeros((h, w), dtype=np.float32)
        for theta in np.arange(0, np.pi, np.pi / 4):
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray_img, cv2.CV_32F, kernel)
            accum = np.maximum(accum, filtered)
        # Use Otsu threshold on accumulated response
        accum_uint8 = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(accum_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    def _multiscale_defect_detection_stub(gray_img: np.ndarray, scales: List[float]) -> np.ndarray:
        """
        Run a simple blob detection at multiple scales (Gaussian pyramid) to detect regions.
        Returns a binary mask where any scale detected a candidate. (STUB VERSION)
        """
        accum = np.zeros_like(gray_img, dtype=np.uint8)
        for s in scales:
            resized = cv2.resize(gray_img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            blurred = cv2.GaussianBlur(resized, (5, 5), 0)
            # Use simple threshold in scaled space
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Upsample back to original
            up = cv2.resize(thresh, (gray_img.shape[1], gray_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            accum = cv2.bitwise_or(accum, up)
        return accum

    def _lei_scratch_detection_stub(gray_img: np.ndarray, kernel_lengths: List[int], angle_step: int) -> np.ndarray:
        """
        LEI-inspired linear enhancement scratch detector.
        Returns a float32 response map. (STUB VERSION)
        """
        h, w = gray_img.shape
        max_resp = np.zeros((h, w), dtype=np.float32)
        for length in kernel_lengths:
            for theta_deg in range(0, 180, angle_step):
                theta = np.deg2rad(theta_deg)
                # Create a linear kernel: a rotated line of ones of length 'length'
                kern = np.zeros((length, length), dtype=np.float32)
                cv2.line(
                    kern,
                    (length // 2, 0),
                    (length // 2, length - 1),
                    1, thickness=1
                )  # vertical line
                # Rotate kernel
                M = cv2.getRotationMatrix2D((length / 2, length / 2), theta_deg, 1.0)
                kern_rot = cv2.warpAffine(kern, M, (length, length), flags=cv2.INTER_LINEAR)
                resp = cv2.filter2D(gray_img.astype(np.float32), cv2.CV_32F, kern_rot)
                max_resp = np.maximum(max_resp, resp)
        return max_resp

    def _advanced_scratch_detection_stub(gray_img: np.ndarray) -> np.ndarray:
        """
        Example: combination of Canny + Hough to detect line segments.
        Returns binary mask of detected lines. (STUB VERSION)
        """
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
        mask = np.zeros_like(gray_img, dtype=np.uint8)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=15, minLineLength=10, maxLineGap=5
        )
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
        return mask

    def _wavelet_defect_detection_stub(gray_img: np.ndarray) -> np.ndarray:
        """
        Detect defects using wavelet decomposition (e.g., Haar).  
        Returns a binary mask of potential anomalies. (STUB VERSION)
        """
        coeffs = pywt.dwt2(gray_img.astype(np.float32), 'haar')
        cA, (cH, cV, cD) = coeffs
        # Compute magnitude of detail coefficients
        mag = np.sqrt(cH**2 + cV**2 + cD**2)
        mag_resized = cv2.resize(mag, (gray_img.shape[1], gray_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mag_uint8 = cv2.normalize(mag_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(mag_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    # Assign stubs to the names that will be later (potentially) overwritten by detailed implementations
    # This ensures that if this module is run standalone and config_loader is missing,
    # the _multiscale_defect_detection (stub or later detailed one) can call _do2mr_detection (stub).
    _do2mr_detection = _do2mr_detection_stub
    _gabor_defect_detection = _gabor_defect_detection_stub
    _multiscale_defect_detection = _multiscale_defect_detection_stub
    _lei_scratch_detection = _lei_scratch_detection_stub
    _advanced_scratch_detection = _advanced_scratch_detection_stub
    _wavelet_defect_detection = _wavelet_defect_detection_stub

# --- Image Loading and Preprocessing ---
def load_and_preprocess_image(image_path_str: str, profile_config: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Loads an image, converts it to grayscale, and applies configured preprocessing steps.

    Args:
        image_path_str: Path to the image file.
        profile_config: The specific processing profile sub-dictionary from the main config,
                        containing preprocessing parameters.

    Returns:
        A tuple containing:
            - original_bgr: The original loaded BGR image (for annotation).
            - gray_image: The initial grayscale image.
            - processed_image: The grayscale image after all preprocessing steps.
        Returns None if the image cannot be loaded.
    """
    image_path = Path(image_path_str) # Convert string path to Path object.
    if not image_path.exists() or not image_path.is_file(): # Check if the path is a valid file.
        logging.error(f"Image file not found or is not a file: {image_path}")
        return None # Return None if image not found or not a file.

    original_bgr = cv2.imread(str(image_path)) # Read the image using OpenCV.
    if original_bgr is None: # Check if image loading failed.
        logging.error(f"Failed to load image: {image_path}")
        return None # Return None if loading failed.
    logging.info(f"Image '{image_path.name}' loaded successfully.")

    gray_image = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY) # Convert BGR image to 8-bit grayscale.
    logging.debug("Image converted to grayscale.")

    # --- Illumination Correction (CLAHE) ---
    # Get CLAHE parameters from the profile config.
    clahe_clip_limit = profile_config.get("preprocessing", {}).get("clahe_clip_limit", 2.0)
    clahe_tile_size_list = profile_config.get("preprocessing", {}).get("clahe_tile_grid_size", [8, 8])
    clahe_tile_grid_size = tuple(clahe_tile_size_list) if isinstance(clahe_tile_size_list, list) and len(clahe_tile_size_list) == 2 else (8,8)
        
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size) # Create CLAHE object.
    # The paper mentions histogram equalization for LEI. CLAHE is generally more robust for varying illumination.
    illum_corrected_image = clahe.apply(gray_image) # Apply CLAHE to the grayscale image.
    logging.debug(f"CLAHE applied with clipLimit={clahe_clip_limit}, tileGridSize={clahe_tile_grid_size}.")

    #  --- Advanced Illumination Correction (if enabled) ---
    if profile_config.get("preprocessing", {}).get("enable_illumination_correction", False):
        illum_corrected_image = _correct_illumination(illum_corrected_image)
        logging.debug("Applied advanced illumination correction.")

    # --- Noise Reduction (Gaussian Blur) ---
    # Get Gaussian blur parameters from the profile config.
    blur_kernel_list = profile_config.get("preprocessing", {}).get("gaussian_blur_kernel_size", [5, 5])
    gaussian_blur_kernel_size = tuple(blur_kernel_list) if isinstance(blur_kernel_list, list) and len(blur_kernel_list) == 2 else (5,5)
    # Ensure kernel dimensions are odd.
    tmp = []
    for k in gaussian_blur_kernel_size:
        tmp.append(k if (k % 2 == 1) else (k + 1))
    gaussian_blur_kernel_size = tuple(tmp)


    # The paper uses Gaussian filtering before DO2MR
    processed_image = cv2.GaussianBlur(illum_corrected_image, gaussian_blur_kernel_size, 0) # Apply Gaussian blur.
    logging.debug(f"Gaussian blur applied with kernel size {gaussian_blur_kernel_size}.")

    return original_bgr, gray_image, processed_image # Return original, grayscale, and processed images.


def _correct_illumination(gray_image: np.ndarray) -> np.ndarray:
    """
    Performs advanced illumination correction using rolling ball algorithm.
    """
    # Estimate background using morphological closing with large kernel
    kernel_size = 50
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    
    # Subtract background
    corrected = cv2.subtract(gray_image, background)
    corrected = cv2.add(corrected, 128)  # Shift to mid-gray
    
    return corrected

def locate_fiber_structure(
    processed_image: np.ndarray,
    profile_config: Dict[str, Any],
    original_gray_image: Optional[np.ndarray] = None # Added for core detection
) -> Optional[Dict[str, Any]]:
    """
    Locates the fiber cladding and core using HoughCircles, contour fitting, or circle-fit library.

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

    # Initialize dictionary to store localization results.
    localization_result = {}
    # Log the parameters being used for HoughCircles.
    logging.debug(f"Attempting HoughCircles with dp={dp}, minDist={min_dist_circles}, p1={param1}, p2={param2}, minR={min_radius_hough}, maxR={max_radius_hough}")
    
    # --- Primary Method: HoughCircles for Cladding Detection ---
    # Detect circles in the processed image.
    circles = cv2.HoughCircles(
        processed_image, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist_circles,
        param1=param1, param2=param2, minRadius=min_radius_hough, maxRadius=max_radius_hough
    )

# Enhanced multi-method circle detection
    if circles is None or 'cladding_center_xy' not in localization_result:
        logging.info("Attempting enhanced multi-method circle detection")
        
        # Method 1: Template matching for circular patterns
        if processed_image.shape[0] > 100 and processed_image.shape[1] > 100: # Ensure image is large enough
            # Create circular template
            template_radius = int(min_img_dim * 0.3)
            if template_radius > 1: # Ensure template radius is valid
                template = np.zeros((template_radius*2, template_radius*2), dtype=np.uint8)
                cv2.circle(template, (template_radius, template_radius), template_radius, 255, -1)
                
                # Match template at multiple scales
                best_match_val = 0
                best_match_loc = None
                best_match_scale = 1.0
                
                for scale in np.linspace(0.5, 1.5, 11): # 11 scales from 0.5x to 1.5x
                    if template_radius * scale < 1: # Skip if scaled template is too small
                        continue
                    scaled_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    if scaled_template.shape[0] == 0 or scaled_template.shape[1] == 0: # ensure scaled template is not empty
                        continue
                    if scaled_template.shape[0] > processed_image.shape[0] or scaled_template.shape[1] > processed_image.shape[1]:
                        continue
                        
                    result = cv2.matchTemplate(processed_image, scaled_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > best_match_val:
                        best_match_val = max_val
                        best_match_loc = max_loc
                        best_match_scale = scale
                
                if best_match_val > 0.6 and best_match_loc: # Threshold for good match
                    detected_radius = int(template_radius * best_match_scale)
                    detected_center = (best_match_loc[0] + detected_radius, best_match_loc[1] + detected_radius)
                    
                    localization_result['cladding_center_xy'] = detected_center
                    localization_result['cladding_radius_px'] = float(detected_radius)
                    localization_result['localization_method'] = 'TemplateMatching'
                    logging.info(f"Cladding detected via template matching: Center={detected_center}, Radius={detected_radius}px")

    # Check if any circles were found by HoughCircles.
    if circles is not None:
        # Log the number of circles detected.
        logging.info(f"HoughCircles detected {circles.shape[1]} circle(s).")
        # Convert circle parameters (x, y, radius) to integers.
        circles_int = np.uint16(np.around(circles))
        # Initialize variables to select the best circle.
        best_circle_hough = None
        # Initialize max radius found so far.
        max_r_hough_found = 0
        # Calculate image center coordinates.
        img_center_x, img_center_y = w // 2, h // 2
        # Initialize minimum distance to image center.
        min_dist_to_img_center = float('inf')

        # Iterate through all detected circles to find the best candidate for cladding.
        for c_hough in circles_int[0, :]:
            # Extract center coordinates (cx, cy) and radius (r).
            cx_h, cy_h, r_h = int(c_hough[0]), int(c_hough[1]), int(c_hough[2])
            # Calculate distance of circle center from image center.
            dist_h = np.sqrt((cx_h - img_center_x)**2 + (cy_h - img_center_y)**2)
            
            # Heuristic: Prefer larger circles closer to the image center.
            # This condition checks if the current circle is "better" than previously found ones.
            if r_h > max_r_hough_found - 20 and dist_h < min_dist_to_img_center + 20 : # Allow some tolerance.
                 if r_h > max_r_hough_found or dist_h < min_dist_to_img_center: # Prioritize radius then centrality.
                    max_r_hough_found = r_h # Update max radius.
                    min_dist_to_img_center = dist_h # Update min distance to center.
                    best_circle_hough = c_hough # Update best circle.
        
        # If no specific "best" circle was selected through scoring, and circles were found, pick the first one as a fallback.
        if best_circle_hough is None and len(circles_int[0,:]) > 0:
            best_circle_hough = circles_int[0,0] # Select the first detected circle.
            logging.warning("Multiple circles from Hough; heuristic didn't pinpoint one, took the first as cladding.")

        # If a best circle was determined by Hough method.
        if best_circle_hough is not None:
            # Extract parameters of the best circle.
            cladding_cx, cladding_cy, cladding_r = int(best_circle_hough[0]), int(best_circle_hough[1]), int(best_circle_hough[2])
            # Store cladding center coordinates.
            localization_result['cladding_center_xy'] = (cladding_cx, cladding_cy)
            # Store cladding radius in pixels.
            localization_result['cladding_radius_px'] = float(cladding_r)
            # Store the method used for localization.
            localization_result['localization_method'] = 'HoughCircles'
            # Log the detected cladding parameters.
            logging.info(f"Cladding (Hough): Center=({cladding_cx},{cladding_cy}), Radius={cladding_r}px")
        else:
            # If HoughCircles detected circles but failed to select a best one (e.g. all too small/off-center).
            logging.warning("HoughCircles detected circles, but failed to select a suitable cladding circle.")
            # Ensure circles is None to trigger fallback if this path is taken.
            circles = None 
    else:
        # This log occurs if cv2.HoughCircles itself returns None.
        if 'cladding_center_xy' not in localization_result: # Only log if template matching also failed
            logging.warning("HoughCircles found no circles initially (and template matching did not yield a result).")


    # --- Fallback Method 1: Adaptive Thresholding + Contour Fitting ---
    # This block is executed if the primary HoughCircles method failed to identify a suitable cladding.
    if 'cladding_center_xy' not in localization_result:
        # Log that the system is attempting the first fallback method.
        logging.warning("Attempting adaptive threshold contour fitting fallback for cladding detection.")
        
        # Get adaptive thresholding parameters from the profile configuration.
        adaptive_thresh_block_size = loc_params.get("adaptive_thresh_block_size", 31) # Block size for adaptive threshold.
        adaptive_thresh_C = loc_params.get("adaptive_thresh_C", 5) # Constant subtracted from the mean.
        # Ensure block size is odd, as required by OpenCV.
        if adaptive_thresh_block_size % 2 == 0: adaptive_thresh_block_size +=1

        # Determine which image to use for thresholding.
        # 'original_gray_image' (if available and less processed) might be better than 'processed_image'.
        image_for_thresh = original_gray_image if original_gray_image is not None else processed_image

        # Apply adaptive thresholding. THRESH_BINARY_INV is used if the fiber is darker than background.
        # If fiber is brighter, THRESH_BINARY should be used.
        thresh_img_adaptive = cv2.adaptiveThreshold(
            image_for_thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adaptive_thresh_block_size, adaptive_thresh_C
        )
        logging.debug("Adaptive threshold applied for contour fallback.")
        
        # --- Enhanced Morphological Operations for Fallback ---
        # Close small gaps in the fiber structure.
        kernel_close_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)) # Kernel for closing.
        closed_adaptive = cv2.morphologyEx(thresh_img_adaptive, cv2.MORPH_CLOSE, kernel_close_large, iterations=2) # Apply closing.
        logging.debug("Applied large closing operation to adaptive threshold result.")
        
        # Fill holes within the identified fiber structure.
        # binary_fill_holes expects a binary image (0 or 1).
        closed_adaptive_binary = (closed_adaptive // 255).astype(np.uint8) # Convert to 0/1.
        try:
            filled_adaptive = ndimage.binary_fill_holes(closed_adaptive_binary).astype(np.uint8) * 255 # Fill holes.
            logging.debug("Applied hole filling to adaptive threshold result.")
        except Exception as e_fill: # Handle potential errors in binary_fill_holes.
            logging.warning(f"Hole filling failed: {e_fill}. Proceeding with un-filled image.")
            filled_adaptive = closed_adaptive # Use the image before hole filling.
        
        # Open to remove small noise or protrusions after filling.
        kernel_open_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)) # Smaller kernel for opening.
        opened_adaptive = cv2.morphologyEx(filled_adaptive, cv2.MORPH_OPEN, kernel_open_small, iterations=1) # Apply opening.
        logging.debug("Applied small opening operation to adaptive threshold result.")
        
        # Find contours on the cleaned binary image.
        contours_adaptive, _ = cv2.findContours(opened_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # List to store valid fiber contours found by this fallback method.
        valid_fiber_contours = []
        if contours_adaptive: # If any contours were found.
            logging.debug(f"Found {len(contours_adaptive)} contours after adaptive thresholding and morphology.")
            for c_adap in contours_adaptive: # Iterate through each contour.
                area = cv2.contourArea(c_adap) # Calculate contour area.
                
                # Filter by area: contour must be reasonably large.
                # These are relative to min/max radius expected by Hough, providing some bounds.
                min_area_expected = (np.pi * (min_radius_hough**2)) * 0.3 # Heuristic: e.g. 30% of min Hough area.
                max_area_expected = (np.pi * (max_radius_hough**2)) * 2.0 # Heuristic: e.g. 200% of max Hough area.
                if not (min_area_expected < area < max_area_expected): # If area is outside expected range.
                    logging.debug(f"Contour skipped: Area {area:.1f}px outside range ({min_area_expected:.1f}-{max_area_expected:.1f})px.")
                    continue # Skip this contour.

                perimeter = cv2.arcLength(c_adap, True) # Calculate contour perimeter.
                if perimeter == 0: continue # Avoid division by zero if perimeter is zero.
                circularity = 4 * np.pi * area / (perimeter**2) # Calculate circularity.
                
                # Filter by circularity: fiber end face should be somewhat circular.
                # A perfect circle has circularity 1.0.
                if circularity < 0.5: # Adjust this threshold based on expected fiber shape.
                    logging.debug(f"Contour skipped: Circularity {circularity:.2f} < 0.5.")
                    continue # Skip this contour.
                
                valid_fiber_contours.append(c_adap) # Add valid contour to list.

            # If valid fiber contours were found by adaptive thresholding.
            if valid_fiber_contours:
                # Select the largest valid contour as the best candidate for the fiber.
                fiber_contour_adaptive = max(valid_fiber_contours, key=cv2.contourArea)
                logging.info(f"Selected largest valid contour (Area: {cv2.contourArea(fiber_contour_adaptive):.1f}px, Circularity: {4 * np.pi * cv2.contourArea(fiber_contour_adaptive) / (cv2.arcLength(fiber_contour_adaptive, True)**2):.2f}) for fitting.")
                
                # Check if the contour has enough points for ellipse fitting.
                if len(fiber_contour_adaptive) >= 5:
                    # Check config if ellipse fitting is preferred for this profile.
                    if loc_params.get("use_ellipse_detection", True): # Default to True if not specified
                        # Fit an ellipse to the contour.
                        ellipse_params = cv2.fitEllipse(fiber_contour_adaptive)
                        # Extract ellipse parameters: center (cx, cy), axes (minor, major), angle.
                        cladding_cx, cladding_cy = int(ellipse_params[0][0]), int(ellipse_params[0][1])
                        cladding_minor_axis = ellipse_params[1][0] # Minor axis.
                        cladding_major_axis = ellipse_params[1][1] # Major axis.
                        # Store ellipse parameters in the localization result.
                        localization_result['cladding_center_xy'] = (cladding_cx, cladding_cy)
                        # Calculate average radius from major and minor axes.
                        localization_result['cladding_radius_px'] = (cladding_major_axis + cladding_minor_axis) / 4.0
                        localization_result['cladding_ellipse_params'] = ellipse_params # Store full ellipse parameters.
                        localization_result['localization_method'] = 'ContourFitEllipse' # Mark method.
                        logging.info(f"Cladding (ContourFitEllipse): Center=({cladding_cx},{cladding_cy}), Axes=({cladding_minor_axis:.1f},{cladding_major_axis:.1f})px, Angle={ellipse_params[2]:.1f}deg")
                    else: # If ellipse detection is disabled, fit a minimum enclosing circle.
                        (cx_circ, cy_circ), r_circ = cv2.minEnclosingCircle(fiber_contour_adaptive) # Fit circle.
                        localization_result['cladding_center_xy'] = (int(cx_circ), int(cy_circ)) # Store center.
                        localization_result['cladding_radius_px'] = float(r_circ) # Store radius.
                        localization_result['localization_method'] = 'ContourFitCircle' # Mark method.
                        logging.info(f"Cladding (ContourFitCircle): Center=({int(cx_circ)},{int(cy_circ)}), Radius={r_circ:.1f}px")
                else:
                    # Log if the largest contour is too small for fitting.
                    logging.warning("Adaptive contour found, but too small for robust ellipse/circle fitting (less than 5 points).")
            else:
                # Log if no suitable contours were found after filtering.
                logging.warning("Adaptive thresholding did not yield any suitable fiber contours after filtering.")
        else:
            # Log if no contours were found at all by adaptive thresholding.
            logging.warning("No contours found after adaptive thresholding and initial morphological operations.")


    # --- Fallback Method 2: Circle-Fit library (if enabled and previous methods failed) ---
    # This block is executed if cladding_center_xy is still not found and circle-fit is enabled and available.
    if 'cladding_center_xy' not in localization_result and loc_params.get("use_circle_fit", True) and CIRCLE_FIT_AVAILABLE:
        # Log attempt to use circle-fit library.
        logging.info("Attempting circle-fit library method as a further fallback for cladding detection.")
        try:
            # Using 'processed_image' for Canny to get edge points for circle_fit.
            # Alternative: use 'original_gray_image' if it provides cleaner edges for this specific method.
            edges_for_circle_fit = cv2.Canny(processed_image, 50, 150) # Standard Canny parameters.
            
            # Find contours on these Canny edges.
            contours_cf, _ = cv2.findContours(edges_for_circle_fit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours_cf: # If contours are found.
                # Concatenate points from all reasonably large contours for a more robust fit.
                # This helps if the fiber edge is broken into multiple segments by Canny.
                all_points_for_cf = []
                for c_cf in contours_cf:
                    if cv2.contourArea(c_cf) > 50: # Heuristic: filter out very small noisy contours.
                        all_points_for_cf.extend(c_cf.reshape(-1,2))
                
                if len(all_points_for_cf) > 20 : # Ensure enough points for a reliable fit.
                    points_for_cf_np = np.array(all_points_for_cf) # Convert list of points to NumPy array.
                    
                    # Define fitting methods from circle_fit library to try.
                    fit_methods_cf = [
                        ('algebraic', cf.least_squares_circle), # Fast, but can be sensitive to outliers.
                        ('hyper', cf.hyper_fit),             # Generally more robust.
                        ('taubin', cf.taubin_svd)            # Robust, often considered good.
                    ]
                    
                    best_fit_circle_cf = None # Initialize best fit circle.
                    best_residual_cf = float('inf') # Initialize best residual.
                    
                    # Iterate through defined fitting methods.
                    for method_name_cf, fit_func_cf in fit_methods_cf:
                        try:
                            # Perform circle fitting.
                            xc_cf, yc_cf, r_cf, residual_cf = fit_func_cf(points_for_cf_np)
                            # Sanity checks for the fitted circle:
                            # - Radius within expected Hough bounds (loosened slightly).
                            # - Center within image boundaries.
                            if min_radius_hough * 0.7 < r_cf < max_radius_hough * 1.3 and \
                               0 < xc_cf < w and 0 < yc_cf < h:
                                if residual_cf < best_residual_cf: # If current fit is better.
                                    best_fit_circle_cf = (xc_cf, yc_cf, r_cf) # Update best fit.
                                    best_residual_cf = residual_cf # Update best residual.
                                    logging.debug(f"Circle-fit ({method_name_cf}): Center=({xc_cf:.1f},{yc_cf:.1f}), R={r_cf:.1f}px, Residual={residual_cf:.3f}")
                        except Exception as e_cf_fit: # Handle errors during fitting.
                            logging.debug(f"Circle-fit method {method_name_cf} failed: {e_cf_fit}")
                    
                    if best_fit_circle_cf: # If a best fit was found.
                        xc_final_cf, yc_final_cf, r_final_cf = best_fit_circle_cf # Unpack best fit parameters.
                        # Store results.
                        localization_result['cladding_center_xy'] = (int(xc_final_cf), int(yc_final_cf))
                        localization_result['cladding_radius_px'] = float(r_final_cf)
                        localization_result['localization_method'] = 'CircleFitLib' # Mark method.
                        localization_result['fit_residual'] = best_residual_cf # Store fit residual.
                        logging.info(f"Cladding (CircleFitLib best): Center=({int(xc_final_cf)},{int(yc_final_cf)}), Radius={r_final_cf:.1f}px, Residual={best_residual_cf:.3f}")
                    else: # If no suitable circle found by circle_fit.
                        logging.warning("Circle-fit library methods did not yield a suitable circle.")
                else: # If not enough points for fitting.
                    logging.warning(f"Not enough contour points ({len(all_points_for_cf)}) for robust circle-fit library method.")
            else: # If no contours found for circle-fit.
                logging.warning("No contours found from Canny edges for circle-fit library method.")
        except ImportError: # Handle if circle_fit library is not actually available.
            logging.error("circle_fit library was marked as available in config but failed to import.")
            # Ensure CIRCLE_FIT_AVAILABLE is False if it fails here to prevent repeated attempts.
            # global CIRCLE_FIT_AVAILABLE # (if it's a global flag, this would be needed)
            # CIRCLE_FIT_AVAILABLE = False # This should ideally modify the global flag if this function can be re-entered.
        except Exception as e_circle_fit_main: # Handle other errors during circle-fit process.
            logging.error(f"An error occurred during the circle-fit library attempt: {e_circle_fit_main}")

    # --- After all attempts, check if cladding was found ---
    if 'cladding_center_xy' not in localization_result: # If cladding center is still not found.
        logging.error("Failed to localize fiber cladding by any method.")
        return None # Critical failure, return None.

    # --- Core Detection (Proceeds if cladding was successfully found) ---
    # Ensure original_gray_image is used for better intensity distinction if available.
    image_for_core_detect = original_gray_image if original_gray_image is not None else processed_image
    
    # Create a mask for the cladding area to search for the core.
    cladding_mask_for_core_det = np.zeros_like(image_for_core_detect, dtype=np.uint8)
    cl_cx_core, cl_cy_core = localization_result['cladding_center_xy'] # Get cladding center.

    # Use the determined localization method to create the search mask for the core.
    # Reduce search radius slightly (e.g., 90-95% of cladding) to avoid cladding edge effects.
    search_radius_factor = 0.90 
    if localization_result.get('localization_method') in ['HoughCircles', 'CircleFitLib', 'ContourFitCircle', 'TemplateMatching']:
        cl_r_core_search = int(localization_result['cladding_radius_px'] * search_radius_factor)
        cv2.circle(cladding_mask_for_core_det, (cl_cx_core, cl_cy_core), cl_r_core_search, 255, -1)
    elif localization_result.get('cladding_ellipse_params'): # If cladding was an ellipse.
        ellipse_p_core = localization_result['cladding_ellipse_params']
        # Scale down ellipse axes for core search.
        scaled_axes_core = (ellipse_p_core[1][0] * search_radius_factor, ellipse_p_core[1][1] * search_radius_factor)
        cv2.ellipse(cladding_mask_for_core_det, (ellipse_p_core[0], scaled_axes_core, ellipse_p_core[2]), 255, -1)
    else: # Should not happen if cladding_center_xy is present, but as a safeguard.
        logging.error("Cladding localization method unknown for core detection masking. Cannot proceed with core detection.")
        # Return with at least cladding info, core will be marked as not found or estimated.
        localization_result['core_center_xy'] = localization_result['cladding_center_xy'] # Default to cladding center.
        localization_result['core_radius_px'] = localization_result['cladding_radius_px'] * 0.4 # Default typical ratio.
        logging.warning(f"Core detection failed due to masking issue, defaulting to 0.4 * cladding radius.")
        return localization_result

    # Apply the cladding mask to the image chosen for core detection.
    masked_for_core = cv2.bitwise_and(image_for_core_detect, image_for_core_detect, mask=cladding_mask_for_core_det)

    # Otsu's thresholding: Core is darker, so THRESH_BINARY_INV makes core white.
    _, core_thresh_inv_otsu = cv2.threshold(masked_for_core, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Re-mask to ensure it's strictly within the search area.
    core_thresh_inv_otsu = cv2.bitwise_and(core_thresh_inv_otsu, core_thresh_inv_otsu, mask=cladding_mask_for_core_det)
    
    # Morphological opening to remove small noise from core thresholding.
    kernel_core_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    core_thresh_inv_otsu_opened = cv2.morphologyEx(core_thresh_inv_otsu, cv2.MORPH_OPEN, kernel_core_open, iterations=1)

    # Find contours of potential core regions.
    core_contours, _ = cv2.findContours(core_thresh_inv_otsu_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if core_contours: # If core contours are found.
        best_core_contour = None # Initialize best core contour.
        min_core_dist_to_cl_center = float('inf') # Initialize min distance to cladding center.
        max_core_area = 0 # Initialize max core area (alternative selection criteria).

        # Iterate through found core contours.
        for c_core_contour in core_contours:
            area_core = cv2.contourArea(c_core_contour) # Calculate area.
            # Min area for core (e.g., related to min_radius_hough, but for core which is smaller).
            if area_core < np.pi * (min_radius_hough * 0.1)**2 : continue 
            
            M_core = cv2.moments(c_core_contour) # Calculate moments.
            if M_core["m00"] == 0: continue # Skip if area is zero.
            core_cx_cand, core_cy_cand = int(M_core["m10"] / M_core["m00"]), int(M_core["m01"] / M_core["m00"]) # Centroid.
            
            # Distance from this candidate core center to the established cladding center.
            dist_to_cladding_center = np.sqrt((core_cx_cand - cl_cx_core)**2 + (core_cy_cand - cl_cy_core)**2)
            
            # Core should be very close to the cladding center.
            # Max allowed offset could be a small fraction of cladding radius.
            max_offset_allowed = localization_result['cladding_radius_px'] * 0.2 # e.g., 20% of cladding radius.

            if dist_to_cladding_center < max_offset_allowed: # If core is reasonably centered.
                # Prefer the largest valid contour that is well-centered.
                if area_core > max_core_area:
                    max_core_area = area_core
                    best_core_contour = c_core_contour
                    min_core_dist_to_cl_center = dist_to_cladding_center # Also track its distance
        
        if best_core_contour is not None: # If a best core contour was selected.
            (core_cx_fit, core_cy_fit), core_r_fit = cv2.minEnclosingCircle(best_core_contour) # Fit circle to contour.
            # Store core parameters.
            localization_result['core_center_xy'] = (int(core_cx_fit), int(core_cy_fit))
            localization_result['core_radius_px'] = float(core_r_fit)
            logging.info(f"Core (ContourFit): Center=({int(core_cx_fit)},{int(cy_fit)}), Radius={core_r_fit:.1f}px")
        else: # If no suitable core contour found.
            logging.warning("Could not identify a distinct core contour within the cladding using current criteria.")
            # Fallback: estimate core based on typical ratio to cladding.
            localization_result['core_center_xy'] = localization_result['cladding_center_xy'] # Default to cladding center.
            localization_result['core_radius_px'] = localization_result['cladding_radius_px'] * 0.4 # Default typical ratio.
            logging.warning(f"Core detection failed, defaulting to 0.4 * cladding radius.")
    else: # If no core contours found by Otsu.
        logging.warning("No core contours found using Otsu within cladding mask.")
        localization_result['core_center_xy'] = localization_result['cladding_center_xy'] # Default to cladding center.
        localization_result['core_radius_px'] = localization_result['cladding_radius_px'] * 0.4 # Default typical ratio.
        logging.warning(f"Core detection defaulting to 0.4 * cladding radius.")

    return localization_result # Return all localization data.


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
    masks: Dict[str, np.ndarray] = {} # Initialize dictionary for masks.
    h, w = image_shape[:2] # Get image height and width.
    Y, X = np.ogrid[:h, :w] # Create Y and X coordinate grids.

    # Get detected fiber parameters
    cladding_center = localization_data.get('cladding_center_xy') # Get cladding center.
    # Core parameters might be used if zones are relative to core for some fiber types
    core_center = localization_data.get('core_center_xy', cladding_center) # Default core center to cladding center.
    core_radius_px = localization_data.get('core_radius_px', 0) # Get core radius.
    
    # Detected cladding dimensions
    detected_cladding_radius_px = localization_data.get('cladding_radius_px') # Get cladding radius.
    cladding_ellipse_params = localization_data.get('cladding_ellipse_params') # Get cladding ellipse parameters.


    if cladding_center is None: # If cladding center not found.
        logging.error("Cannot generate zone masks: Cladding center not localized.")
        return masks # Return empty masks.

    # The main reference for zones is typically the cladding outer diameter.
    # If user provided cladding diameter, use it as the reference for scaling factors.
    # Otherwise, use the detected cladding radius in pixels.
    reference_cladding_diameter_um = user_cladding_diameter_um # Use user-provided cladding diameter.
    reference_core_diameter_um = user_core_diameter_um # Use user-provided core diameter.

    for zone_def in zone_definitions: # Iterate through each zone definition.
        name = zone_def["name"] # Get zone name.
        # Zone radii factors can be relative to core or cladding diameter from config.
        # e.g., r_max_factor_core_relative, r_max_factor_cladding_relative
        
        r_min_px: float = 0.0 # Initialize min radius in pixels.
        r_max_px: float = 0.0 # Initialize max radius in pixels.

        # Determine r_min_px and r_max_px based on config and detected/provided dimensions.
        # This logic needs to be robust for different factor types in zone_def.
        # Example: if 'r_max_factor_core_relative' is used, it's relative to core diameter.
        # If 'r_max_factor_cladding_relative' is used, it's relative to cladding diameter.
        # If absolute 'r_min_um', 'r_max_um' are in config, those take precedence if um_per_px is known.

        # Simplified logic based on prompt's example config structure:
        # Factors like "r_max_factor_core_relative" would apply to the *known* core diameter
        # to get a micron value, then convert to pixels.
        # If operating in pixel mode, these factors might apply directly to detected pixel radii.
        
        # This section requires careful mapping from config structure to pixel radii.
        # For the given example config:
        # "Core": "r_max_factor_core_relative": 1.0 means r_max for core IS the core radius.
        # "Cladding": "r_max_factor_cladding_relative": 1.0 means r_max for cladding IS the cladding radius.
        # The r_min for Cladding needs to be Core's r_max.
        # Adhesive/Contact are relative to Cladding's outer radius.
        
        # --- Determine pixel radii for the current zone ---
        # This logic assumes the factors are relative to a specific feature's *actual* radius (core or cladding)
        # or are absolute micron values that need conversion.
        
        current_zone_center = cladding_center # Default center for most zones.
        is_elliptical_zone = False # Flag for elliptical zones.
        # current_zone_ellipse_params = None # Store ellipse parameters if needed. # Not used

        if um_per_px and reference_cladding_diameter_um: # Micron mode with reference dimensions.
            # Convert um definitions from config (if present) or calculate from factors.
            # This example assumes factors define radii that become absolute after applying to a reference.
            # Example: Core zone's r_max_factor_core_relative=1 means its r_max_um is reference_core_diameter_um / 2.
            if name == "Core" and reference_core_diameter_um: # If zone is Core and core diameter known.
                r_min_um = zone_def.get("r_min_factor", 0.0) * (reference_core_diameter_um / 2.0) # Apply factor to radius.
                r_max_um = zone_def.get("r_max_factor_core_relative", 1.0) * (reference_core_diameter_um / 2.0) # Apply factor.
                current_zone_center = core_center # Use detected core center for core zone.
            elif name == "Cladding" and reference_cladding_diameter_um and reference_core_diameter_um: # If zone is Cladding and cladding diameter known.
                # Cladding r_min is core's r_max.
                core_def_temp = next((zd for zd in zone_definitions if zd["name"] == "Core"), None) # Get core definition.
                r_min_um_cladding_start = 0.0 # Initialize.
                if core_def_temp: # If core definition exists.
                     r_min_um_cladding_start = core_def_temp.get("r_max_factor_core_relative",1.0) * (reference_core_diameter_um / 2.0)
                
                r_min_um_from_factor = zone_def.get("r_min_factor_cladding_relative", 0.0) * (reference_cladding_diameter_um / 2.0) # Factor relative to cladding itself (e.g. 0.0 for start of cladding annulus).
                r_min_um = max(r_min_um_from_factor, r_min_um_cladding_start) # Ensure it starts after core.

                r_max_um = zone_def.get("r_max_factor_cladding_relative", 1.0) * (reference_cladding_diameter_um / 2.0) # Factor relative to cladding.
            elif name == "Cladding" and reference_cladding_diameter_um and not reference_core_diameter_um: # Cladding, but no core diameter
                logging.warning(f"Zone '{name}': Missing reference core diameter for precise r_min_um. Assuming r_min starts at cladding center for this factor.")
                r_min_um = zone_def.get("r_min_factor_cladding_relative", 0.0) * (reference_cladding_diameter_um / 2.0)
                r_max_um = zone_def.get("r_max_factor_cladding_relative", 1.0) * (reference_cladding_diameter_um / 2.0)
            else: # For Adhesive, Contact, etc., typically relative to cladding outer diameter.
                # These r_min/max factors are assumed to be multipliers of the cladding radius.
                cladding_outer_r_um = reference_cladding_diameter_um / 2.0 # Cladding outer radius.
                r_min_um = zone_def.get("r_min_factor_cladding_relative", 1.0) * cladding_outer_r_um # e.g. 1.0 for start of adhesive.
                r_max_um = zone_def.get("r_max_factor_cladding_relative", 1.15) * cladding_outer_r_um # e.g. 1.15 for end of adhesive.

            r_min_px = r_min_um / um_per_px # Convert min um to px.
            r_max_px = r_max_um / um_per_px # Convert max um to px.
        
        elif detected_cladding_radius_px: # Pixel mode, using detected cladding as reference.
            # This mode is used if um_per_px or reference_cladding_diameter_um is not available.
            # Factors in config are applied to *detected* pixel radii.
            if name == "Core": # For Core zone.
                # Core radius in config might be relative to its own detected size, or a factor of cladding.
                # The example config implies "r_max_factor_core_relative" is 1.0 to its own radius.
                r_min_px = 0.0 # Core starts at center.
                r_max_px = core_radius_px if core_radius_px > 0 else detected_cladding_radius_px * 0.4 # Use detected core radius or fallback.
                current_zone_center = core_center # Use detected core center.
            elif name == "Cladding": # For Cladding zone.
                r_min_px = core_radius_px if core_radius_px > 0 else detected_cladding_radius_px * 0.4 # Starts after core.
                r_max_px = detected_cladding_radius_px # Ends at detected cladding radius.
            else: # For Adhesive, Contact.
                # Factors are relative to detected cladding radius.
                r_min_px = zone_def.get("r_min_factor_cladding_relative", 1.0) * detected_cladding_radius_px
                r_max_px = zone_def.get("r_max_factor_cladding_relative", 1.15) * detected_cladding_radius_px
        else: # Fallback if no scale and no detected cladding radius.
            logging.error(f"Cannot define zone '{name}' due to missing scale and localization data.")
            continue # Skip this zone.

        # Create mask for the current zone (annulus).
        zone_mask_np = np.zeros((h, w), dtype=np.uint8) # Initialize zone mask.
        cx_zone, cy_zone = current_zone_center # Get current zone center. Note: Renamed from cx, cy to avoid clash with outer scope in some editors

        # Use ellipse for non-core zones if cladding was elliptical,
        # or for core zone if core itself was not distinctly found (and thus defaults to cladding shape).
        use_ellipse_for_zone = cladding_ellipse_params and \
                               (name != "Core" or (name == "Core" and core_radius_px <=0) or \
                                (name=="Core" and localization_data.get('core_center_xy') == localization_data.get('cladding_center_xy')))


        if use_ellipse_for_zone:
            base_center_ell = (int(cladding_ellipse_params[0][0]), int(cladding_ellipse_params[0][1]))
            # Use core_center if the zone IS Core and a distinct core_center was found.
            # Otherwise, stick to the cladding's ellipse center.
            if name == "Core" and core_center != cladding_center: # If core has its own center
                 # This case is tricky: Core zone but using cladding's ellipticity.
                 # For simplicity, if core has its own center, assume it's circular unless core_ellipse_params are also found.
                 # The current `use_ellipse_for_zone` logic might need refinement for this specific sub-case.
                 # Defaulting to cladding's ellipse params but potentially centered at core_center if different.
                 # This could be complex if the core center is significantly offset from elliptical cladding center.
                 # For now, if using ellipse for core, it's based on cladding's ellipse data.
                 # If core_center is different, it implies core was found circularly, so this branch might not be hit for core.
                 pass # Sticking to base_center_ell which is cladding_ellipse_params[0]

            base_minor_axis = cladding_ellipse_params[1][0] # Full minor axis length
            base_major_axis = cladding_ellipse_params[1][1] # Full major axis length
            base_angle = cladding_ellipse_params[2] # Cladding angle.

            # Average radius of the fitted cladding ellipse
            avg_cladding_ellipse_radius = (base_major_axis + base_minor_axis) / 4.0

            if avg_cladding_ellipse_radius > 0: # Avoid division by zero
                # Scale factors based on target pixel radii for the zone vs. avg cladding ellipse radius
                scale_factor_max = r_max_px / avg_cladding_ellipse_radius
                scale_factor_min = r_min_px / avg_cladding_ellipse_radius
            else: # Should not happen if detected_cladding_radius_px was valid for ellipse
                scale_factor_max = 1.0
                scale_factor_min = 0.0
            
            # Calculate scaled axes for the current zone's ellipse (these are full axis lengths)
            outer_ellipse_axes = (int(base_minor_axis * scale_factor_max), int(base_major_axis * scale_factor_max))
            inner_ellipse_axes = (int(base_minor_axis * scale_factor_min), int(base_major_axis * scale_factor_min))
            
            # Draw outer ellipse (filled)
            if r_max_px > 0 and outer_ellipse_axes[0] > 0 and outer_ellipse_axes[1] > 0:
                 cv2.ellipse(zone_mask_np, (base_center_ell, outer_ellipse_axes, base_angle), 255, -1)
            
            # Subtract inner ellipse (draw filled black ellipse on a temp mask, then subtract)
            if r_min_px > 0 and inner_ellipse_axes[0] > 0 and inner_ellipse_axes[1] > 0:
                 temp_inner_mask = np.zeros_like(zone_mask_np)
                 cv2.ellipse(temp_inner_mask, (base_center_ell, inner_ellipse_axes, base_angle), 255, -1)
                 zone_mask_np = cv2.subtract(zone_mask_np, temp_inner_mask)
            is_elliptical_zone = True

        else: # Circular zones.
            dist_sq_map = (X - cx_zone)**2 + (Y - cy_zone)**2 # Squared distance from center map.
            # Mask is 1 where r_min_px^2 <= dist_sq < r_max_px^2.
            zone_mask_np = ((dist_sq_map >= r_min_px**2) & (dist_sq_map < r_max_px**2)).astype(np.uint8) * 255

        masks[name] = zone_mask_np # Store the generated mask.
        logging.debug(f"Generated mask for zone '{name}': Center={current_zone_center}, Rmin_px={r_min_px:.1f}, Rmax_px={r_max_px:.1f}, Elliptical={is_elliptical_zone}")

    return masks # Return dictionary of zone masks.

# --- DETAILED ALGORITHM IMPLEMENTATIONS ---
# These will overwrite stubs if stubs were defined (i.e., if config_loader import failed)

def _lei_scratch_detection(enhanced_image: np.ndarray, kernel_lengths: List[int], angle_step: int = 15) -> np.ndarray:
    """
    Complete LEI implementation with dual-branch approach from paper Section 3.2
    """
    h, w = enhanced_image.shape[:2]
    max_response_map = np.zeros((h, w), dtype=np.float32)
    
    # Pre-compute sin/cos for efficiency
    angles_rad = np.deg2rad(np.arange(0, 180, angle_step))
    
    for length in kernel_lengths:
        # Half-width for dual branches (paper specifies 5 pixels total width)
        branch_offset = 2  # Distance from center to branch
        
        for angle_rad in angles_rad:
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            # Create accumulator for this orientation
            response_at_angle = np.zeros((h, w), dtype=np.float32) # Renamed from 'response' to avoid conflict
            
            # Optimized sampling (conceptual, direct pixel access is still iterative)
            # A fully vectorized version would be significantly more complex with custom kernel generation or generalized Hough
            for y_center in range(h): # y_center instead of y
                for x_center in range(w): # x_center instead of x
                    red_sum = 0.0
                    red_count = 0
                    gray_sum = 0.0
                    gray_count = 0
                    
                    for t in range(-length//2, length//2 + 1):
                        # Center line (red branch)
                        cx = int(round(x_center + t * cos_a)) # round for better accuracy
                        cy = int(round(y_center + t * sin_a))
                        
                        if 0 <= cx < w and 0 <= cy < h:
                            red_sum += enhanced_image[cy, cx]
                            red_count += 1
                        
                        # Side branches (gray) - perpendicular offset
                        for side in [-1, 1]:
                            # Perpendicular vector to (cos_a, sin_a) is (-sin_a, cos_a) or (sin_a, -cos_a)
                            gx = int(round(x_center + t * cos_a + side * branch_offset * (-sin_a)))
                            gy = int(round(y_center + t * sin_a + side * branch_offset * cos_a))
                            
                            if 0 <= gx < w and 0 <= gy < h:
                                gray_sum += enhanced_image[gy, gx]
                                gray_count += 1
                    
                    if red_count > length * 0.7 and gray_count > 0:  # Ensure sufficient sampling
                        f_r = red_sum / red_count
                        f_g = gray_sum / gray_count
                        response_at_angle[y_center, x_center] = max(0, f_r - f_g) # Paper's formula is often R = Fr - Fg or R = 2Fr - Fg (original had 2Fr)
                                                                        # Using Fr - Fg as a common variant, adjust if 2Fr - Fg from specific paper is needed.
                                                                        # The prompt's original detailed _lei_scratch_detection had max(0, 2 * f_r - f_g)
                                                                        # Reverting to that for consistency with prompt.
                        response_at_angle[y_center, x_center] = max(0, 2 * f_r - f_g)


            # Apply Gaussian smoothing to reduce noise (original paper might apply this per orientation or at end)
            # response_at_angle = cv2.GaussianBlur(response_at_angle, (3, 3), 0.5) # Smoothing per angle
            
            # Update maximum response
            max_response_map = np.maximum(max_response_map, response_at_angle)
    
    # Global smoothing after all angles
    max_response_map = cv2.GaussianBlur(max_response_map, (3,3), 0.5)

    # Post-process to enhance linear structures
    # Apply morphological top-hat to enhance bright linear features
    # Using a small kernel as response map should already highlight lines
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)) # Or (3,1) depending on typical scratch orientation relative to image axes after max accumulation
    max_response_map_uint8 = cv2.normalize(max_response_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    max_response_map_processed = cv2.morphologyEx(max_response_map_uint8, cv2.MORPH_TOPHAT, kernel_tophat) 
    # The function should return float32 map as per its original stub, so normalize and return float
    # However, downstream usually expects uint8 for thresholding. Let's return float for now.
    # Normalizing the final output before returning float
    cv2.normalize(max_response_map_processed, max_response_map_processed, 0, 1.0, cv2.NORM_MINMAX) # Normalize to 0-1 float

    return max_response_map_processed.astype(np.float32)


def _gabor_defect_detection(image: np.ndarray) -> np.ndarray:
    """
    Uses Gabor filters for texture-based defect detection.
    Particularly good for detecting periodic defects and scratches.
    """
    gabor_filters = [] # Renamed from 'filters'
    ksize = 31 # Kernel size
    sigma = 4.0 # Sigma for Gaussian envelope
    lambd = 10.0 # Wavelength of the sinusoidal factor
    gamma = 0.5 # Spatial aspect ratio (ellipticity)
    psi = 0 # Phase offset (phi in OpenCV)
    
    # Create Gabor filters at different orientations
    for theta in np.arange(0, np.pi, np.pi / 8): # Iterate 8 orientations
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        gabor_filters.append(kern)
    
    # Apply filters and combine responses
    responses = []
    for kern in gabor_filters:
        filtered = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kern) # Ensure input is float
        responses.append(np.abs(filtered)) # Magnitude of response
    
    # Combine responses - use maximum response across all orientations
    gabor_response = np.max(np.array(responses), axis=0) # Convert list to np.array before np.max
    
    # Normalize to 0-255 uint8 for thresholding
    gabor_response_norm = cv2.normalize(gabor_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold to get defect mask
    _, defect_mask = cv2.threshold(gabor_response_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return defect_mask

def _wavelet_defect_detection(image: np.ndarray) -> np.ndarray:
    """
    Uses wavelet transform for multi-resolution defect detection.
    Effective for detecting defects at different scales.
    """
    # Perform 2D discrete wavelet transform
    # Ensure image is float32 for pywt
    coeffs = pywt.dwt2(image.astype(np.float32), 'db4') # Using Daubechies 4
    cA, (cH, cV, cD) = coeffs # cA: approximation, cH/cV/cD: horizontal/vertical/diagonal details
    
    # Combine detail coefficients (energy of details)
    details_magnitude = np.sqrt(cH**2 + cV**2 + cD**2)
    
    # Reconstruct at original size (Resize detail magnitude map)
    details_resized = cv2.resize(details_magnitude, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to 0-255 uint8 for thresholding
    details_norm = cv2.normalize(details_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold to get defects
    _, defect_mask = cv2.threshold(details_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return defect_mask

# IMPORTANT REORDERING: Detailed _do2mr_detection now comes BEFORE _multiscale_defect_detection
def _do2mr_detection(masked_zone_image: np.ndarray, kernel_size: int = 5, gamma: float = 1.5) -> np.ndarray:
    """
    Enhanced DO2MR implementation following sensors-18-01408-v2.pdf Section 3.1
    """
    # Ensure input is 8-bit grayscale
    if masked_zone_image.dtype != np.uint8:
        masked_zone_image = cv2.normalize(masked_zone_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Multiple kernel sizes for multi-scale detection as per paper
    # If a specific kernel_size is given, use it. If default (5) is used, use multi-scale.
    kernel_sizes_to_use = [3, 5, 7] if kernel_size == 5 else [kernel_size] 
    combined_result = np.zeros_like(masked_zone_image, dtype=np.float32)
    
    for k_size in kernel_sizes_to_use:
        # Square structuring element as per paper
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        
        # Min-Max filtering (Equations 1-3 in paper)
        min_filtered = cv2.erode(masked_zone_image, kernel, iterations=1)
        max_filtered = cv2.dilate(masked_zone_image, kernel, iterations=1)
        
        # Residual calculation (Equation 4)
        # Ensure subtraction is safe (e.g. convert to float or use cv2.subtract)
        residual_current = cv2.subtract(max_filtered.astype(np.int16), min_filtered.astype(np.int16)).astype(np.float32)
        # residual_current = max_filtered.astype(np.float32) - min_filtered.astype(np.float32)
        
        # Apply weight based on kernel size (smaller kernels for fine details)
        weight = 1.0 / k_size if k_size > 0 else 1.0
        combined_result += residual_current * weight
    
    # Normalize combined result to range 0-255
    if np.any(combined_result): # Avoid normalization if all zeros
        cv2.normalize(combined_result, combined_result, 0, 255, cv2.NORM_MINMAX)
    residual_normalized = combined_result.astype(np.uint8)
    
    # Sigma-based thresholding (Equation 6 from paper)
    # Apply threshold only within the actual masked zone (where masked_zone_image was > 0 initially)
    # However, _do2mr_detection is called with an already masked image.
    # So, we analyze pixels within the *current* image that are non-zero.
    active_pixels_mask = masked_zone_image > 0 # Use original mask for stats if available, else current non-zero
    if np.sum(active_pixels_mask) == 0: # if the input masked_zone_image is all black
        return np.zeros_like(masked_zone_image, dtype=np.uint8)
    
    # Calculate mean and std on the *residual_normalized* values within the active_pixels_mask
    mean_res = np.mean(residual_normalized[active_pixels_mask])
    std_res = np.std(residual_normalized[active_pixels_mask])
    
    # Dynamic gamma adjustment placeholder (as in original code)
    # A better way would be to pass zone_name or a pre-calculated gamma
    # if 'core' in str(masked_zone_image.shape).lower(): # This is a fragile check
    #     gamma_local = 1.2  # More sensitive for core
    # else:
    #     gamma_local = gamma # Use passed or default gamma
    gamma_local = gamma # Using passed gamma

    thresh_value = mean_res + gamma_local * std_res
    # Ensure thresh_value is within 0-255 range for uint8 image
    thresh_value = np.clip(thresh_value, 0, 255)

    _, defect_binary = cv2.threshold(residual_normalized, thresh_value, 255, cv2.THRESH_BINARY)
    
    # Post-processing exactly as paper specifies
    defect_binary = cv2.medianBlur(defect_binary, 3) # Median filter with 3x3 kernel
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # 3x3 elliptical SE for opening
    defect_binary = cv2.morphologyEx(defect_binary, cv2.MORPH_OPEN, kernel_open)
    
    # Additional connected component filtering (filter small defects)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_binary, connectivity=8)
    filtered_mask = np.zeros_like(defect_binary)
    
    min_defect_area_px = 5 # Minimum area threshold (from paper or config)
    for i in range(1, num_labels): # Skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_defect_area_px:
            filtered_mask[labels == i] = 255
    
    return filtered_mask

def _multiscale_defect_detection(image: np.ndarray, scales: List[float] = [0.5, 1.0, 1.5, 2.0]) -> np.ndarray:
    """
    Performs multi-scale defect detection for improved accuracy.
    Based on scale-space theory for robust defect detection.
    This version uses the (now detailed) _do2mr_detection internally.
    """
    h, w = image.shape[:2]
    combined_map_float = np.zeros((h, w), dtype=np.float32) # Accumulate float results
    
    for scale in scales:
        # Resize image
        scaled_image = image.copy() # Default to original image if scale is 1.0
        if scale != 1.0:
            if scale <= 0: continue # Skip invalid scales
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            if scaled_h <=0 or scaled_w <=0: continue # Skip if scaled dimensions are zero/negative
            scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply DO2MR at this scale
        # The detailed _do2mr_detection has default kernel_size=5 and gamma=1.5
        # We might want to scale the kernel_size for _do2mr_detection based on the image scale
        do2mr_kernel_size_at_scale = max(3, int(5 * scale)) # Ensure kernel size is at least 3 and odd
        if do2mr_kernel_size_at_scale % 2 == 0: do2mr_kernel_size_at_scale +=1
        
        # _do2mr_detection expects an already masked image. 'scaled_image' here is from 'image' which is already masked_zone_image
        do2mr_result_at_scale = _do2mr_detection(scaled_image, kernel_size=do2mr_kernel_size_at_scale) # Using detailed _do2mr
        
        # Resize result back to original size
        do2mr_result_resized = do2mr_result_at_scale # Default if scale is 1.0
        if scale != 1.0:
            do2mr_result_resized = cv2.resize(do2mr_result_at_scale, (w, h), interpolation=cv2.INTER_NEAREST) # Use NEAREST for binary mask
        
        # Weight by scale (smaller scales for fine details, larger for bigger defects)
        # This weighting logic might need tuning.
        weight = 1.0 / scale if scale > 1 else scale if scale > 0 else 1.0
        combined_map_float += do2mr_result_resized.astype(np.float32) * weight
    
    # Normalize the combined float map to 0-255 and convert to uint8
    # This map represents a kind of confidence or accumulated detection score.
    if np.any(combined_map_float): # Avoid normalization if all zeros
        cv2.normalize(combined_map_float, combined_map_float, 0, 255, cv2.NORM_MINMAX)
    
    # The function is expected to return a binary mask by some callers/original stubs.
    # For now, let's threshold this combined map using Otsu to get a final binary mask.
    # Or, it could return the float map if fusion logic expects float inputs.
    # The original stub returned a binary mask. Let's stick to that.
    combined_map_uint8 = combined_map_float.astype(np.uint8)
    _, final_binary_mask = cv2.threshold(combined_map_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return final_binary_mask


def _advanced_scratch_detection(image: np.ndarray) -> np.ndarray:
    """
    Advanced scratch detection using multiple techniques.
    Combines Hessian-based ridge detection, morphological operations, and Hough lines.
    Input 'image' should be 8-bit grayscale.
    """
    if image.dtype != np.uint8: # Ensure input is uint8
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = image.shape[:2]
    scratch_map_combined = np.zeros((h, w), dtype=np.uint8) # Combined binary mask
    
    # 1. Ridge detection using Hessian eigenvalues (Frangi filter is more robust but complex)
    # This basic Hessian approach can be noisy.
    # Sobel derivatives
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobelxx = cv2.Sobel(sobelx, cv2.CV_64F, 1, 0, ksize=5) # cv2.Sobel(image, cv2.CV_64F, 2, 0, ksize=5)
    sobelyy = cv2.Sobel(sobely, cv2.CV_64F, 0, 1, ksize=5) # cv2.Sobel(image, cv2.CV_64F, 0, 2, ksize=5)
    sobelxy = cv2.Sobel(sobelx, cv2.CV_64F, 0, 1, ksize=5) # cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)

    ridge_response = np.zeros_like(image, dtype=np.float64)
    # Eigenvalue calculation (can be slow in Python loop)
    # Consider skimage.feature.hessian_matrix and hessian_matrix_eigvals for optimized version
    for r in range(h): # Renamed y to r to avoid confusion
        for c in range(w): # Renamed x to c
            hessian_matrix = np.array([[sobelxx[r,c], sobelxy[r,c]], 
                                       [sobelxy[r,c], sobelyy[r,c]]])
            try:
                eigenvalues, _ = np.linalg.eig(hessian_matrix) # Use eig for general matrices
                # For dark lines (ridges in inverted image or valleys in original)
                # we look for a large positive eigenvalue and a small eigenvalue for the other.
                # For bright lines, large negative eigenvalue. Scratches are usually dark.
                # Assuming scratches are darker: one large positive eigenvalue after inversion, or large negative here.
                # The original code looked for eigenvalues.min() < -10 (for bright lines/valleys).
                # Let's stick to detecting dark scratches. A dark scratch is a valley.
                # A valley has one small (close to zero) eigenvalue and one large positive eigenvalue for Dyy (if horizontal)
                # This part needs careful formulation of ridge/valley response from Hessian.
                # The provided code used `eigenvalues.min() < -10` which implies bright line detection.
                # For dark lines on bright background, one eigenvalue will be large positive, other small.
                # Let lambda1, lambda2 be eigenvalues, lambda1 <= lambda2.
                # For dark line: lambda2 is large positive, lambda1 is small (close to 0).
                # For bright line: lambda1 is large negative, lambda2 is small (close to 0).
                # The original used `eigenvalues.min() < -10`, so it was for bright lines.
                # We'll keep this logic, assuming it was intended.
                if eigenvalues.min() < -50: # Adjusted threshold, highly empirical
                    ridge_response[r, c] = np.abs(eigenvalues.min())
            except np.linalg.LinAlgError:
                pass # Skip if eigenvalue computation fails

    if np.any(ridge_response):
        ridge_response_norm = cv2.normalize(ridge_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, ridge_mask = cv2.threshold(ridge_response_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        scratch_map_combined = cv2.bitwise_or(scratch_map_combined, ridge_mask)
    
    # 2. Morphological black-hat for dark scratches
    # This is effective for enhancing dark features on a lighter background.
    kernel_bh_rect_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)) # For vertical scratches
    kernel_bh_rect_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)) # For horizontal scratches
    
    blackhat_v = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel_bh_rect_vertical)
    blackhat_h = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel_bh_rect_horizontal)
    
    # Combine responses from different orientations
    blackhat_combined = np.maximum(blackhat_v, blackhat_h)
    # Add other angles if needed, e.g., 45 degrees
    # kernel_bh_diag = cv2.getStructuringElement(cv2.MORPH_CROSS, (9,9)) # Example for general small dark spots
    # blackhat_diag = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel_bh_diag)
    # blackhat_combined = np.maximum(blackhat_combined, blackhat_diag)

    if np.any(blackhat_combined):
        _, bh_thresh = cv2.threshold(blackhat_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        scratch_map_combined = cv2.bitwise_or(scratch_map_combined, bh_thresh)
    
    # 3. Line segment detection using Canny + HoughLinesP
    edges = cv2.Canny(image, 50, 150, apertureSize=3) # Standard Canny
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=7) # Adjusted params
    
    if lines is not None:
        line_mask = np.zeros_like(image, dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1) # Thinner lines
        scratch_map_combined = cv2.bitwise_or(scratch_map_combined, line_mask)
    
    # Clean up noise from combined map
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Open to remove small noise, Close to connect nearby segments
    scratch_map_combined = cv2.morphologyEx(scratch_map_combined, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    scratch_map_combined = cv2.morphologyEx(scratch_map_combined, cv2.MORPH_CLOSE, kernel_clean, iterations=1)
    
    return scratch_map_combined


def detect_defects(
    processed_image: np.ndarray,
    zone_mask: np.ndarray,
    zone_name: str,  # Added zone_name here
    profile_config: Dict[str, Any],
    global_algo_params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced defect detection using multi-algorithm fusion approach.
    
    Returns:
        Tuple of (final_defect_mask, confidence_map)
    """
    if np.sum(zone_mask) == 0:
        logging.debug(f"Defect detection skipped for empty zone mask in zone '{zone_name}'.")
        return np.zeros_like(processed_image, dtype=np.uint8), np.zeros_like(processed_image, dtype=np.float32)

    h, w = processed_image.shape[:2]
    confidence_map = np.zeros((h, w), dtype=np.float32)

    # working_image will be used by all algorithms after potential zone-specific preprocessing
    working_image = cv2.bitwise_and(processed_image, processed_image, mask=zone_mask)

    # Apply zone-specific preprocessing based on the passed zone_name
    if zone_name == "Core":
        logging.debug(f"Applying {zone_name}-specific preprocessing.")
        working_image_original_dtype = working_image.dtype
        
        # Median blur
        working_image = cv2.medianBlur(working_image, 3)
        
        # CLAHE for contrast enhancement in the core
        if np.any(working_image[zone_mask > 0]): # Ensure image is not all black before CLAHE
            clahe_core = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            
            # Apply CLAHE only to the relevant parts of the image to avoid issues with all-zero inputs
            # This is a bit more robust if working_image can be all zeros within the mask
            temp_img_for_clahe = working_image.copy()
            # Check if the masked area has content
            if np.any(temp_img_for_clahe[zone_mask > 0]):
                # Apply CLAHE. CLAHE works on uint8 or uint16.
                if temp_img_for_clahe.dtype not in [np.uint8, np.uint16]:
                    temp_img_for_clahe = cv2.normalize(temp_img_for_clahe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                enhanced_region = clahe_core.apply(temp_img_for_clahe) # Apply to whole image, then re-mask
                working_image = cv2.bitwise_and(enhanced_region, enhanced_region, mask=zone_mask)
            #else: working_image remains as is (after median blur)
        
        # Ensure dtype consistency if necessary, though most OpenCV functions handle uint8
        if working_image.dtype != working_image_original_dtype and working_image_original_dtype == np.uint8:
             working_image = cv2.normalize(working_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    elif zone_name == "Cladding":
        logging.debug(f"Applying {zone_name}-specific preprocessing.")
        # Bilateral filter for cladding
        working_image = cv2.bilateralFilter(working_image, d=5, sigmaColor=50, sigmaSpace=50)
        # Re-mask to ensure filter didn't affect outside areas (important for bilateral)
        working_image = cv2.bitwise_and(working_image, working_image, mask=zone_mask)
    
    # Add other 'elif zone_name == "XYZ":' blocks here if other zones need specific preprocessing

    logging.debug(f"Proceeding with defect detection for zone: '{zone_name}' using specifically preprocessed image.")


    detection_cfg = profile_config.get("defect_detection", {})
    region_algos = detection_cfg.get("region_algorithms", [])
    linear_algos = detection_cfg.get("linear_algorithms", [])
    optional_algos = detection_cfg.get("optional_algorithms", [])
    algo_weights = detection_cfg.get("algorithm_weights", {})

    # A. Region Defect Analysis
    if "do2mr" in region_algos:
        # Determine gamma for DO2MR based on zone or use a default
        current_do2mr_gamma = global_algo_params.get("do2mr_gamma_default", 1.5)
        if zone_name == "Core":
            current_do2mr_gamma = global_algo_params.get("do2mr_gamma_core", 1.2)
        
        do2mr_result = _do2mr_detection(working_image, kernel_size=5, gamma=current_do2mr_gamma)
        confidence_map[do2mr_result > 0] += algo_weights.get("do2mr", 0.8)
        logging.debug("Applied DO2MR for region defects.")

    if "morph_gradient" in region_algos:
        kernel_size_list_mg = global_algo_params.get("morph_gradient_kernel_size", [5,5]) # Renamed var
        kernel_mg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(kernel_size_list_mg))
        morph_gradient_img = cv2.morphologyEx(working_image, cv2.MORPH_GRADIENT, kernel_mg)
        _, thresh_mg = cv2.threshold(morph_gradient_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        confidence_map[thresh_mg > 0] += algo_weights.get("morph_gradient", 0.4)
        logging.debug("Applied Morphological Gradient for region defects.")

    if "black_hat" in region_algos:
        kernel_size_list_bh = global_algo_params.get("black_hat_kernel_size", [11,11]) # Renamed var
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(kernel_size_list_bh))
        black_hat_img = cv2.morphologyEx(working_image, cv2.MORPH_BLACKHAT, kernel_bh)
        _, thresh_bh = cv2.threshold(black_hat_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        confidence_map[thresh_bh > 0] += algo_weights.get("black_hat", 0.6)
        logging.debug("Applied Black-Hat Transform for region defects.")
    
    if "gabor" in region_algos:
        gabor_result = _gabor_defect_detection(working_image)
        confidence_map[gabor_result > 0] += algo_weights.get("gabor", 0.4)
        logging.debug("Applied Gabor filters for region defects.")
    
    if "multiscale" in region_algos:
        scales_ms = global_algo_params.get("multiscale_factors", [0.5, 1.0, 1.5, 2.0]) # Renamed var
        multiscale_result = _multiscale_defect_detection(working_image, scales_ms)
        confidence_map[multiscale_result > 0] += algo_weights.get("multiscale", 0.6)
        logging.debug("Applied multi-scale detection for region defects.")

    if "lbp" in region_algos: # _lbp_defect_detection defined at end of file
        lbp_result = _lbp_defect_detection(working_image)
        confidence_map[lbp_result > 0] += algo_weights.get("lbp", 0.3)
        logging.debug("Applied LBP texture analysis for defects.")
    
    # B. Linear Defect Analysis (Scratches)
    if "lei_advanced" in linear_algos:
        # Apply histogram equalization for LEI if working_image is uint8
        enhanced_for_lei = working_image
        if working_image.dtype == np.uint8:
            enhanced_for_lei = cv2.equalizeHist(working_image)
        else: # If not uint8, normalize and convert for equalizeHist
            logging.warning("LEI enhancement: working_image not uint8, normalizing for equalizeHist.")
            norm_for_lei = cv2.normalize(working_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            enhanced_for_lei = cv2.equalizeHist(norm_for_lei)

        lei_kernels = global_algo_params.get("lei_kernel_lengths", [11, 17, 23])
        angle_step_lei = global_algo_params.get("lei_angle_step_deg", 15) # Renamed var
        
        lei_response_float = _lei_scratch_detection(enhanced_for_lei, lei_kernels, angle_step_lei) # Returns float 0-1
        
        # Convert float response (0-1) to uint8 (0-255) for thresholding
        lei_response_uint8 = (lei_response_float * 255).astype(np.uint8)
        _, thresh_lei = cv2.threshold(lei_response_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up
        kernel_open_lei = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)) # For vertical-ish lines
        thresh_lei = cv2.morphologyEx(thresh_lei, cv2.MORPH_OPEN, kernel_open_lei)
        
        confidence_map[thresh_lei > 0] += algo_weights.get("lei_advanced", 0.8)
        logging.debug("Applied LEI-advanced method for linear defects.")
    
    if "advanced_scratch" in linear_algos:
        advanced_scratch_result = _advanced_scratch_detection(working_image)
        confidence_map[advanced_scratch_result > 0] += algo_weights.get("advanced_scratch", 0.7)
        logging.debug("Applied advanced scratch detection.")

    if "skeletonization" in linear_algos:
        # Ensure working_image is uint8 for Canny
        img_for_canny = working_image
        if working_image.dtype != np.uint8:
            img_for_canny = cv2.normalize(working_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        edges_skel = cv2.Canny(img_for_canny, 50, 150, apertureSize=global_algo_params.get("sobel_scharr_ksize",3)) # Renamed var
        try:
            thinned_edges = cv2.ximgproc.thinning(edges_skel, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            dilation_kernel_size_list_skel = global_algo_params.get("skeletonization_dilation_kernel_size",[3,3]) # Renamed var
            dilation_kernel_skel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(dilation_kernel_size_list_skel)) # Renamed var
            thinned_edges_dilated = cv2.dilate(thinned_edges, dilation_kernel_skel, iterations=1)
            confidence_map[thinned_edges_dilated > 0] += algo_weights.get("skeletonization", 0.3)
            logging.debug("Applied Canny + Skeletonization for linear defects.")
        except AttributeError:
            logging.warning("cv2.ximgproc.thinning not available (opencv-contrib-python needed). Skipping skeletonization.")
        except cv2.error as e:
            logging.warning(f"OpenCV error during skeletonization (thinning): {e}. Skipping.")


    # C. Optional Advanced Methods
    if "wavelet" in optional_algos:
        wavelet_result = _wavelet_defect_detection(working_image)
        confidence_map[wavelet_result > 0] += algo_weights.get("wavelet", 0.4)
        logging.debug("Applied wavelet transform for defect detection.")
    
    # D. Scratch Dataset Integration
    if global_algo_params.get("scratch_dataset_path") and "dataset_scratch" in optional_algos:
        try:
            from scratch_dataset_handler import ScratchDatasetHandler # Assuming this exists elsewhere
            dataset_handler = ScratchDatasetHandler(global_algo_params["scratch_dataset_path"])
            scratch_prob = dataset_handler.augment_scratch_detection(working_image) # Ensure working_image is suitable for this
            confidence_map += scratch_prob * algo_weights.get("dataset_scratch", 0.5)
            logging.debug("Applied scratch dataset augmentation.")
        except ImportError:
            logging.warning("ScratchDatasetHandler module not found. Skipping scratch dataset integration.")
        except Exception as e:
            logging.warning(f"Scratch dataset integration failed: {e}")

    # E. Anomaly Detection
    if "anomaly" in optional_algos and ANOMALY_DETECTION_AVAILABLE: # Check ANOMALY_DETECTION_AVAILABLE flag
        try:
            # from anomaly_detection import AnomalyDetector # Already imported or attempted at top
            anomaly_detector = AnomalyDetector(global_algo_params.get("anomaly_model_path"))
            anomaly_mask = anomaly_detector.detect_anomalies(working_image) # Ensure working_image is suitable
            if anomaly_mask is not None:
                confidence_map[anomaly_mask > 0] += algo_weights.get("anomaly", 0.5)
                logging.debug("Applied anomaly detection for defects.")
        # Removed ImportError here as it's handled by ANOMALY_DETECTION_AVAILABLE
        except Exception as e:
            logging.warning(f"Anomaly detection failed: {e}")
    elif "anomaly" in optional_algos and not ANOMALY_DETECTION_AVAILABLE:
        logging.warning("Anomaly detection algorithm specified, but AnomalyDetector module is not available.")


    confidence_threshold_from_config = detection_cfg.get("confidence_threshold", 0.9) 
    # Enhanced fusion with adaptive thresholding based on zone
    zone_adaptive_threshold_map = { # Renamed var
        "Core": 0.7,      # Lower threshold for critical core zone
        "Cladding": 0.9,  # Standard threshold
        "Adhesive": 1.1,  # Higher threshold for less critical zones
        "Contact": 1.2
    }

    # Apply zone-specific threshold if identified, otherwise use the general confidence_threshold
    adaptive_threshold_val = zone_adaptive_threshold_map.get(zone_name, confidence_threshold_from_config) # Renamed var

    # Multi-level thresholding for better defect separation
    high_confidence_mask = (confidence_map >= adaptive_threshold_val).astype(np.uint8) * 255
    medium_confidence_mask = ((confidence_map >= adaptive_threshold_val * 0.7) &
                              (confidence_map < adaptive_threshold_val)).astype(np.uint8) * 128

    # Combine masks with morphological operations
    combined_defect_mask = cv2.bitwise_or(high_confidence_mask, medium_confidence_mask) # Renamed var

    # Size-based filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_defect_mask, connectivity=8) # Use combined_defect_mask
    final_defect_mask_in_zone = np.zeros_like(combined_defect_mask, dtype=np.uint8) # Use combined_defect_mask shape

    min_area_by_confidence_map = { # Renamed var
        255: detection_cfg.get("min_defect_area_px_high_conf", 5), # High confidence min area
        128: detection_cfg.get("min_defect_area_px_med_conf", 10)  # Medium confidence needs larger area
    }
    default_min_area = detection_cfg.get("min_defect_area_px", 5)


    for i in range(1, num_labels): # Iterate through detected components (label 0 is background)
        area = stats[i, cv2.CC_STAT_AREA]
        component_mask = (labels == i)
        
        # Ensure the region is not empty before calling max() (though component_mask should not be empty here)
        if np.any(component_mask):
            mask_val = combined_defect_mask[component_mask].max() # Get max value (255 or 128) in this component
            min_area = min_area_by_confidence_map.get(mask_val, default_min_area)

            if area >= min_area:
                final_defect_mask_in_zone[component_mask] = 255 # Mark accepted defects as 255
        else:
            logging.debug(f"Skipping empty labeled region {i} during size-based filtering.")


    # Ensure defects are strictly within the original zone_mask bounds
    final_defect_mask_in_zone = cv2.bitwise_and(final_defect_mask_in_zone, final_defect_mask_in_zone, mask=zone_mask)
    # Stray hyphen was here, removed.

    # Final morphological cleaning
    kernel_clean_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) # Renamed var
    final_defect_mask_in_zone = cv2.morphologyEx(final_defect_mask_in_zone, cv2.MORPH_OPEN, kernel_clean_final)
    final_defect_mask_in_zone = cv2.morphologyEx(final_defect_mask_in_zone, cv2.MORPH_CLOSE, kernel_clean_final)

    logging.debug(f"Defect detection fusion complete for zone '{zone_name}'. Adaptive threshold: {adaptive_threshold_val:.2f}. Fallback config threshold: {confidence_threshold_from_config:.2f}.")
    return final_defect_mask_in_zone, confidence_map

def _lbp_defect_detection(gray_img: np.ndarray) -> np.ndarray:
    """
    Local Binary Pattern detection for texture-based defects
    Input gray_img should be 8-bit.
    """
    # Ensure input is uint8
    if gray_img.dtype != np.uint8:
        gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    from skimage.feature import local_binary_pattern # Keep import local if only used here
    
    # LBP parameters
    radius = 1
    n_points = 8 * radius # Number of points in a circularly symmetric neighborhood
    METHOD = 'uniform' # 'uniform' is robust to rotation and has fewer patterns
    
    # Compute LBP
    lbp = local_binary_pattern(gray_img, n_points, radius, method=METHOD)
    
    # LBP result needs to be scaled to 0-255 for visualization/thresholding
    # The range of LBP values depends on n_points and method. For 'uniform', it's n_points + 2.
    lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply adaptive threshold to find anomalous LBP regions
    # Parameters for adaptiveThreshold might need tuning
    thresh = cv2.adaptiveThreshold(lbp_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2) # INV if defects are different texture
    
    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return thresh

# --- Main function for testing this module (optional) ---
if __name__ == "__main__":
    # This block is for testing the image_processing module independently.
    # It requires a sample image and a dummy config or access to config_loader.
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s') # Basic logging config.

    # --- Dummy Configuration for Testing ---
    # In a real scenario, this would come from config_loader.py
    dummy_profile_config_main = { # Renamed to avoid conflict if other dummy_profile_config exists
        "preprocessing": {
            "clahe_clip_limit": 2.0, 
            "clahe_tile_grid_size": [8, 8], 
            "gaussian_blur_kernel_size": [5, 5],
            "enable_illumination_correction": False 
        },
        "localization": {
            "hough_dp": 1.2, "hough_min_dist_factor": 0.15, 
            "hough_param1": 70, "hough_param2": 35, 
            "hough_min_radius_factor": 0.08, "hough_max_radius_factor": 0.45,
            "use_ellipse_detection": True,
            "use_circle_fit": True 
        },
        "defect_detection": {
            "region_algorithms": ["do2mr", "morph_gradient", "black_hat", "gabor", "multiscale", "lbp"], 
            "linear_algorithms": ["lei_advanced", "advanced_scratch", "skeletonization"],
            "optional_algorithms": ["wavelet"], # Add "anomaly", "dataset_scratch" if stubs/handlers are available
            "confidence_threshold": 0.8, # General fallback threshold
            "min_defect_area_px_high_conf": 3, # Min area for high confidence defects
            "min_defect_area_px_med_conf": 6,  # Min area for medium confidence defects
            "algorithm_weights": { # Example weights
                "do2mr": 0.7, "morph_gradient": 0.4, "black_hat": 0.6, 
                "gabor": 0.5, "multiscale": 0.6, "lbp": 0.3,
                "lei_advanced": 0.8, "advanced_scratch": 0.7, "skeletonization": 0.3,
                "wavelet": 0.4 
            }
        }
    }
    dummy_global_algo_params_main = get_config().get("algorithm_parameters", {}) # Get global algo params from dummy config.
    # Add more specific params to dummy_global_algo_params_main if needed by algos:
    dummy_global_algo_params_main.update({
        "do2mr_gamma_default": 1.5,
        "do2mr_gamma_core": 1.2,
        "multiscale_factors": [0.5, 1.0, 1.5], # Example for multiscale
        # "anomaly_model_path": None, # Path for anomaly detection model
        # "scratch_dataset_path": None # Path for scratch dataset
    })
    
    # --- Dummy Zone Definitions for Testing (replace with actual config loading) ---
    dummy_zone_defs_main = [ # Renamed var
        {"name": "Core", "r_min_factor": 0.0, "r_max_factor_core_relative": 1.0, "color_bgr": [255,0,0]},
        {"name": "Cladding", "r_min_factor_cladding_relative": 0.0, "r_max_factor_cladding_relative": 1.0, "color_bgr": [0,255,0]},
        {"name": "Adhesive", "r_min_factor_cladding_relative": 1.0, "r_max_factor_cladding_relative": 1.15, "color_bgr": [0,0,255]},
    ]

    # --- Test Case: Load and Preprocess an Image ---
    # Replace "path/to/your/sample_fiber_image.png" with an actual image path for testing.
    # Create a dummy image if you don't have one readily available.
    test_image_path_str = "sample_fiber_image.png" # Placeholder path.
    # Create a dummy image for testing if it doesn't exist
    if not Path(test_image_path_str).exists(): # Check if dummy image exists.
        dummy_img_arr_h, dummy_img_arr_w = 600, 800
        dummy_img_arr = np.full((dummy_img_arr_h, dummy_img_arr_w), 128, dtype=np.uint8) # Create dummy array.
        # Draw "cladding" (brighter) and "core" (darker)
        cv2.circle(dummy_img_arr, (dummy_img_arr_w//2, dummy_img_arr_h//2), 150, 200, -1) 
        cv2.circle(dummy_img_arr, (dummy_img_arr_w//2, dummy_img_arr_h//2), 60, 50, -1)   
        # Draw a "scratch" (dark line)
        cv2.line(dummy_img_arr, (dummy_img_arr_w//2 - 100, dummy_img_arr_h//2 - 50), 
                 (dummy_img_arr_w//2 + 100, dummy_img_arr_h//2 + 50), 10, 3) 
        # Draw a "pit" (dark spot)
        cv2.circle(dummy_img_arr, (dummy_img_arr_w//2 + 50, dummy_img_arr_h//2 - 30), 15, 20, -1) 
        # Add some noise
        noise = np.random.randint(0, 15, (dummy_img_arr_h, dummy_img_arr_w), dtype=np.uint8)
        dummy_img_arr = cv2.add(dummy_img_arr, noise)
        dummy_img_arr_bgr = cv2.cvtColor(dummy_img_arr, cv2.COLOR_GRAY2BGR) # Convert to BGR for imwrite
        cv2.imwrite(test_image_path_str, dummy_img_arr_bgr) 
        logging.info(f"Created a dummy image at {test_image_path_str} for testing.")

    logging.info(f"\n--- Test Case 1: Load and Preprocess Image: {test_image_path_str} ---")
    preprocess_result = load_and_preprocess_image(test_image_path_str, dummy_profile_config_main) # Load and preprocess.
    
    if preprocess_result: # If preprocessing successful.
        original_bgr_test, gray_test, processed_test = preprocess_result # Unpack results.
        logging.info(f"Image preprocessed. Shape of processed image: {processed_test.shape}")
        # cv2.imshow("Processed Test Image", processed_test); cv2.waitKey(1); # Optional: display.

        # --- Test Case 2: Locate Fiber Structure ---
        logging.info("\n--- Test Case 2: Locate Fiber Structure ---")
        # Pass original_gray_image for potentially better core/adaptive threshold localization
        localization = locate_fiber_structure(processed_test, dummy_profile_config_main, original_gray_image=gray_test) 
        
        if localization: # If localization successful.
            logging.info(f"Fiber Localization: {localization}")
            # Draw localization on original image for verification
            viz_loc_img = original_bgr_test.copy()
            if 'cladding_center_xy' in localization and 'cladding_radius_px' in localization:
                cc = localization['cladding_center_xy']
                cr = int(localization['cladding_radius_px'])
                cv2.circle(viz_loc_img, cc, cr, (0,255,0), 2) # Green for cladding
                if 'cladding_ellipse_params' in localization:
                     cv2.ellipse(viz_loc_img, localization['cladding_ellipse_params'], (0,255,255), 2) # Yellow for ellipse
            if 'core_center_xy' in localization and 'core_radius_px' in localization:
                coc = localization['core_center_xy']
                cor = int(localization['core_radius_px'])
                cv2.circle(viz_loc_img, coc, cor, (255,0,0), 2) # Blue for core
            # cv2.imshow("Localization Result", viz_loc_img); cv2.waitKey(1);


            # --- Test Case 3: Generate Zone Masks ---
            logging.info("\n--- Test Case 3: Generate Zone Masks ---")
            # For testing, can use pixel mode (um_per_px=None) or provide dummy scale.
            # Providing dummy user diameters to test that logic path if um_per_px is also given.
            um_per_px_test = 0.5 # Example: 0.5 microns per pixel
            user_core_diam_test = 9.0 # Example: 9um core
            user_cladding_diam_test = 125.0 # Example: 125um cladding
            
            zone_masks_generated = generate_zone_masks( # Generate zone masks.
                processed_test.shape, localization, dummy_zone_defs_main,
                um_per_px=um_per_px_test, 
                user_core_diameter_um=user_core_diam_test, 
                user_cladding_diameter_um=user_cladding_diam_test
            )
            if zone_masks_generated: # If zone masks generated.
                logging.info(f"Generated masks for zones: {list(zone_masks_generated.keys())}")
                # Example: Display the core mask overlaid (optional)
                # if "Core" in zone_masks_generated:
                #    core_mask_display = cv2.bitwise_and(original_bgr_test, original_bgr_test, mask=zone_masks_generated["Core"])
                #    cv2.imshow("Core Mask Area", core_mask_display); cv2.waitKey(1);

                # --- Test Case 4: Detect Defects in each Zone ---
                logging.info("\n--- Test Case 4: Detect Defects (Iterating Zones) ---")
                combined_defects_viz = np.zeros_like(processed_test, dtype=np.uint8)

                for zone_n, zone_m in zone_masks_generated.items(): # zone_n is name, zone_m is mask
                    if np.sum(zone_m) == 0:
                        logging.info(f"Skipping defect detection for empty zone: {zone_n}")
                        continue
                    
                    logging.info(f"--- Detecting defects in Zone: {zone_n} ---")
                    defects_mask, conf_map = detect_defects( 
                        processed_test, zone_m, zone_n, # Pass zone_name
                        dummy_profile_config_main, dummy_global_algo_params_main
                    )
                    logging.info(f"Defect detection in '{zone_n}' zone complete. Found {np.sum(defects_mask > 0)} defect pixels.")
                    if np.any(defects_mask):
                        combined_defects_viz = cv2.bitwise_or(combined_defects_viz, defects_mask)
                        # cv2.imshow(f"Defects in {zone_n}", defects_mask); cv2.waitKey(1);
                
                # Display combined defects on original image
                final_viz_img = original_bgr_test.copy()
                final_viz_img[combined_defects_viz > 0] = [0,0,255] # Mark defects in Red
                # cv2.imshow("All Detected Defects", final_viz_img); cv2.waitKey(0)

            else: # If zone mask generation failed.
                logging.warning("Zone mask generation failed for defect detection test.")
        else: # If localization failed.
            logging.error("Fiber localization failed. Cannot proceed with further tests.")
    else: # If preprocessing failed.
        logging.error("Image preprocessing failed.")

    # cv2.destroyAllWindows() # Clean up OpenCV windows if any were left open by waitKey(0)

    # Clean up dummy image if it was created by this script
    if Path(test_image_path_str).exists() and test_image_path_str == "sample_fiber_image.png":
        try:
            Path(test_image_path_str).unlink()
            logging.info(f"Cleaned up dummy image: {test_image_path_str}")
        except OSError as e:
            logging.error(f"Error removing dummy image {test_image_path_str}: {e}")