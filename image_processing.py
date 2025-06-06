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
    # Parameters fine-tuned for fiber optic end face detection:
    # - dp: Inverse ratio of accumulator resolution to image resolution (1.0 = same, 2.0 = half)
    # - minDist: Minimum distance between detected circle centers (prevents multiple detections of same fiber)
    # - param1: Upper threshold for Canny edge detector (higher = fewer edges)
    # - param2: Accumulator threshold for circle centers (lower = more circles detected)
    # - minRadius/maxRadius: Expected fiber size range in pixels
    
    logging.debug(f"HoughCircles parameters: dp={dp}, minDist={min_dist_circles}, "
                  f"param1={param1}, param2={param2}, minRadius={min_radius_hough}, maxRadius={max_radius_hough}")
    
    circles = cv2.HoughCircles(
        processed_image, 
        cv2.HOUGH_GRADIENT, 
        dp=dp,                    # Typical range: 1.0-2.0
        minDist=min_dist_circles, # Typical: 0.1-0.3 * image dimension
        param1=param1,            # Typical range: 50-150
        param2=param2,            # Typical range: 20-50
        minRadius=min_radius_hough,
        maxRadius=max_radius_hough
    )

# Enhanced multi-method circle detection
# Enhanced multi-method circle detection
    if circles is None or 'cladding_center_xy' not in localization_result:
        logging.info("Attempting enhanced multi-method circle detection")
        
        # Method 1: Template matching for circular patterns
        if processed_image.shape[0] > 100 and processed_image.shape[1] > 100:
            # Create circular template
            template_radius = int(min_img_dim * 0.3)
            template = np.zeros((template_radius*2, template_radius*2), dtype=np.uint8)
            cv2.circle(template, (template_radius, template_radius), template_radius, 255, -1)
            
            # Match template at multiple scales
            best_match_val = 0
            best_match_loc = None
            best_match_scale = 1.0
            
            for scale in np.linspace(0.5, 1.5, 11):
                scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
                if scaled_template.shape[0] > processed_image.shape[0] or scaled_template.shape[1] > processed_image.shape[1]:
                    continue
                    
                result = cv2.matchTemplate(processed_image, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_match_val:
                    best_match_val = max_val
                    best_match_loc = max_loc
                    best_match_scale = scale
            
            if best_match_val > 0.6:  # Threshold for good match
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
            # Indexing c_hough[0] etc. is correct, Pylance warning is a false positive
            cx_h, cy_h, r_h = int(c_hough[0]), int(c_hough[1]), int(c_hough[2]) # type: ignore #
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
            cladding_cx, cladding_cy, cladding_r = int(best_circle_hough[0]), int(best_circle_hough[1]), int(best_circle_hough[2]) # type: ignore #
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
        
        # Corrected handling of ndimage.binary_fill_holes
        filled_adaptive = closed_adaptive # Default to pre-fill image
        try:
            fill_result = ndimage.binary_fill_holes(closed_adaptive_binary)
            if fill_result is not None:
                filled_adaptive = fill_result.astype(np.uint8) * 255 # Fill holes. # Corrected from dtype=np.uint8
                logging.debug("Applied hole filling to adaptive threshold result.")
            else:
                logging.warning("ndimage.binary_fill_holes returned None. Using pre-fill image.")
                # filled_adaptive remains closed_adaptive (already set as default)
        except Exception as e_fill: # Handle potential errors in binary_fill_holes.
            logging.warning(f"Hole filling failed: {e_fill}. Proceeding with un-filled image.")
            # filled_adaptive remains closed_adaptive

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
                        # Indexing ellipse_params is correct, Pylance warning is likely false positive
                        cladding_cx, cladding_cy = int(ellipse_params[0][0]), int(ellipse_params[0][1])
                        cladding_minor_axis = ellipse_params[1][0] # Minor axis.
                        cladding_major_axis = ellipse_params[1][1] # Major axis.
                        # Store ellipse parameters in the localization result.
                        localization_result['cladding_center_xy'] = (cladding_cx, cladding_cy)
                        # Calculate average radius from major and minor axes.
                        localization_result['cladding_radius_px'] = (cladding_major_axis + cladding_minor_axis) / 4.0 # Using /4 for radius from two axes
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
                        # ('taubin', cf.taubin_svd) # Commented out due to Pylance error: "taubin_svd" is not a known attribute
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
    # After core detection, add adhesive layer detection
    
    if 'core_center_xy' in localization_result and 'cladding_center_xy' in localization_result:
        # Detect adhesive layer between core and cladding
        cladding_radius = localization_result['cladding_radius_px']
        core_radius = localization_result['core_radius_px']
        
        # Create mask for the region between core and cladding
        adhesive_search_mask = np.zeros_like(image_for_core_detect, dtype=np.uint8)
        cv2.circle(adhesive_search_mask, cl_cx_core, int(cladding_radius * 0.95), 255, -1)
        cv2.circle(adhesive_search_mask, cl_cx_core, int(core_radius * 1.05), 0, -1)
        
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
                
            
    # Create a mask for the cladding area to search for the core.
    cladding_mask_for_core_det = np.zeros_like(image_for_core_detect, dtype=np.uint8)
    cl_cx_core, cl_cy_core = localization_result['cladding_center_xy'] # Get cladding center.

    # Use the determined localization method to create the search mask for the core.
    # Reduce search radius slightly (e.g., 90-95% of cladding) to avoid cladding edge effects.
    search_radius_factor = 0.90 
    if localization_result.get('localization_method') in ['HoughCircles', 'CircleFitLib', 'ContourFitCircle', 'TemplateMatching']:
        cl_r_core_search = int(localization_result['cladding_radius_px'] * search_radius_factor)
        # Corrected color for cv2.circle
        cv2.circle(cladding_mask_for_core_det, (cl_cx_core, cl_cy_core), cl_r_core_search, (255,), -1)
    elif localization_result.get('cladding_ellipse_params'): # If cladding was an ellipse.
        ellipse_p_core = localization_result['cladding_ellipse_params']
        # Scale down ellipse axes for core search.
        scaled_axes_core = (ellipse_p_core[1][0] * search_radius_factor, ellipse_p_core[1][1] * search_radius_factor)
        # Corrected color for cv2.ellipse
        cv2.ellipse(cladding_mask_for_core_det, (ellipse_p_core[0], scaled_axes_core, ellipse_p_core[2]), (255,), -1)
    else: # Should not happen if cladding_center_xy is present, but as a safeguard.
        logging.error("Cladding localization method unknown for core detection masking. Cannot proceed with core detection.")
        # Return with at least cladding info, core will be marked as not found or estimated.
        localization_result['core_center_xy'] = localization_result['cladding_center_xy'] # Default to cladding center.
        localization_result['core_radius_px'] = localization_result['cladding_radius_px'] * 0.4 # Default typical ratio.
        logging.warning(f"Core detection failed due to masking issue, defaulting to 0.4 * cladding radius.")
        return localization_result

    # Apply the cladding mask to the image chosen for core detection.
    masked_for_core = cv2.bitwise_and(image_for_core_detect, image_for_core_detect, mask=cladding_mask_for_core_det)


    # Enhanced core detection with multiple methods
    
    # Method 1: Adaptive threshold for better local contrast handling
    adaptive_core = cv2.adaptiveThreshold(masked_for_core, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 31, 5)
    
    # Method 2: Otsu's thresholding
    _, otsu_core = cv2.threshold(masked_for_core, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Method 3: Gradient-based detection
    gradient_x = cv2.Sobel(masked_for_core, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(masked_for_core, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_mag_norm = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, gradient_core = cv2.threshold(gradient_mag_norm, 30, 255, cv2.THRESH_BINARY)
    
    # Combine methods using voting
    combined_core = np.zeros_like(masked_for_core, dtype=np.uint8)
    combined_core[(adaptive_core > 0) & (otsu_core > 0)] = 255
    combined_core[(adaptive_core > 0) & (gradient_core > 0)] = 255
    combined_core[(otsu_core > 0) & (gradient_core > 0)] = 255
    
    # Re-mask to ensure it's strictly within the search area
    core_thresh_inv_otsu = cv2.bitwise_and(combined_core, combined_core, mask=cladding_mask_for_core_det)

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
            # Corrected typo cy_fit to core_cy_fit
            logging.info(f"Core (ContourFit): Center=({int(core_cx_fit)},{int(core_cy_fit)}), Radius={core_r_fit:.1f}px")
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
    core_center = localization_data.get('core_center_xy', cladding_center) # Default core center to cladding center.
    core_radius_px_detected = localization_data.get('core_radius_px') # Get detected core radius (can be None or 0)
    
    detected_cladding_radius_px = localization_data.get('cladding_radius_px') # Get cladding radius.
    cladding_ellipse_params = localization_data.get('cladding_ellipse_params') # Get cladding ellipse parameters.


    if cladding_center is None: # If cladding center not found.
        logging.error("Cannot generate zone masks: Cladding center not localized.")
        return masks # Return empty masks.

    reference_cladding_diameter_um = user_cladding_diameter_um 
    reference_core_diameter_um = user_core_diameter_um

    for zone_def in zone_definitions: # Iterate through each zone definition.
        name = zone_def["name"]
        r_min_px: float = 0.0 
        r_max_px: float = 0.0 
        
        current_zone_center = cladding_center 
        is_elliptical_zone = False

        # Mode determination: Micron-based or Pixel-based
        # Prefer micron mode if all necessary info is present
        micron_mode_possible = um_per_px is not None and um_per_px > 0 and reference_cladding_diameter_um is not None
        
        if micron_mode_possible:
            logging.debug(f"Zone '{name}': Using micron mode for definitions.")
            # All calculations in microns first, then convert to pixels
            r_min_um: Optional[float] = None
            r_max_um: Optional[float] = None

            if name == "Core" and reference_core_diameter_um is not None:
                core_radius_ref_um = reference_core_diameter_um / 2.0
                r_min_um = zone_def.get("r_min_factor", 0.0) * core_radius_ref_um
                r_max_um = zone_def.get("r_max_factor_core_relative", 1.0) * core_radius_ref_um
                current_zone_center = core_center if core_center is not None else cladding_center
            elif name == "Cladding":
                cladding_radius_ref_um = reference_cladding_diameter_um / 2.0
                r_max_um = zone_def.get("r_max_factor_cladding_relative", 1.0) * cladding_radius_ref_um
                # r_min for cladding is core's r_max
                if reference_core_diameter_um is not None:
                    core_def_temp = next((zd for zd in zone_definitions if zd["name"] == "Core"), None)
                    core_radius_ref_um_for_cladding_min = reference_core_diameter_um / 2.0
                    r_min_um_cladding_start = 0.0
                    if core_def_temp:
                         r_min_um_cladding_start = core_def_temp.get("r_max_factor_core_relative",1.0) * core_radius_ref_um_for_cladding_min
                    
                    r_min_um_from_factor = zone_def.get("r_min_factor_cladding_relative", 0.0) * cladding_radius_ref_um
                    r_min_um = max(r_min_um_from_factor, r_min_um_cladding_start)
                else: # No core reference, cladding r_min relative to its own diameter
                    logging.warning(f"Zone '{name}': Missing reference core diameter for precise r_min_um. r_min relative to cladding.")
                    r_min_um = zone_def.get("r_min_factor_cladding_relative", 0.0) * cladding_radius_ref_um
            else: # Other zones (Adhesive, Contact) relative to cladding outer diameter
                cladding_outer_r_um = reference_cladding_diameter_um / 2.0
                r_min_um = zone_def.get("r_min_factor_cladding_relative", 1.0) * cladding_outer_r_um
                r_max_um = zone_def.get("r_max_factor_cladding_relative", 1.15) * cladding_outer_r_um
            
            if r_min_um is not None and r_max_um is not None and um_per_px is not None and um_per_px > 0: # Ensure um_per_px is valid
                r_min_px = r_min_um / um_per_px
                r_max_px = r_max_um / um_per_px
            else:
                logging.error(f"Cannot define zone '{name}' in micron mode due to missing data (r_min/max_um or um_per_px). Falling back.")
                micron_mode_possible = False # Force fallback to pixel mode if critical data missing

        # Pixel mode (either primary choice or fallback from micron mode)
        if not micron_mode_possible:
            logging.debug(f"Zone '{name}': Using pixel mode for definitions.")
            if detected_cladding_radius_px is not None and detected_cladding_radius_px > 0:
                if name == "Core":
                    r_min_px = 0.0
                    # Use detected core radius if valid, else estimate from cladding
                    core_r_px_to_use = core_radius_px_detected if core_radius_px_detected is not None and core_radius_px_detected > 0 else (detected_cladding_radius_px * 0.4)
                    r_max_px = zone_def.get("r_max_factor_core_relative", 1.0) * core_r_px_to_use # Assume factor applies to actual/estimated core radius
                    current_zone_center = core_center if core_center is not None else cladding_center
                elif name == "Cladding":
                    core_r_px_for_cladding_min = core_radius_px_detected if core_radius_px_detected is not None and core_radius_px_detected > 0 else (detected_cladding_radius_px * 0.4)
                    r_min_px = zone_def.get("r_min_factor_cladding_relative", 0.0) * detected_cladding_radius_px # Factor relative to cladding
                    # Ensure r_min for cladding starts after core
                    r_min_px = max(r_min_px, core_r_px_for_cladding_min) 
                    r_max_px = zone_def.get("r_max_factor_cladding_relative", 1.0) * detected_cladding_radius_px
                else: # Adhesive, Contact relative to detected cladding radius
                    r_min_px = zone_def.get("r_min_factor_cladding_relative", 1.0) * detected_cladding_radius_px
                    r_max_px = zone_def.get("r_max_factor_cladding_relative", 1.15) * detected_cladding_radius_px
            else:
                logging.error(f"Cannot define zone '{name}' in pixel mode: detected_cladding_radius_px is missing or invalid.")
                continue # Skip this zone

        # Create mask for the current zone
        zone_mask_np = np.zeros((h, w), dtype=np.uint8)
        
        if current_zone_center is None: # Should be caught by cladding_center check, but safeguard
            logging.error(f"Critical: current_zone_center is None for zone '{name}'. Skipping.")
            continue
        cx_zone, cy_zone = current_zone_center

        # Determine if ellipse should be used for this zone
        # Use ellipse if cladding was found as elliptical AND (it's not Core OR Core wasn't distinctly found/matches cladding shape)
        use_ellipse_for_zone = (cladding_ellipse_params is not None) and \
                               (name != "Core" or \
                                (name == "Core" and (core_radius_px_detected is None or core_radius_px_detected <= 0)) or \
                                (name == "Core" and localization_data.get('core_center_xy') == localization_data.get('cladding_center_xy')))


        if use_ellipse_for_zone and cladding_ellipse_params is not None: # Ensure cladding_ellipse_params is not None
            # Indexing cladding_ellipse_params is correct, Pylance error is likely false positive
            base_center_ell = (int(cladding_ellipse_params[0][0]), int(cladding_ellipse_params[0][1]))
            # Ensure axes are tuple of floats for cv2.ellipse
            base_minor_axis = float(cladding_ellipse_params[1][0]) 
            base_major_axis = float(cladding_ellipse_params[1][1])
            base_angle = float(cladding_ellipse_params[2])

            avg_cladding_ellipse_radius = (base_major_axis + base_minor_axis) / 4.0 # Using /4 for radius from two axes

            if avg_cladding_ellipse_radius > 1e-6: # Avoid division by zero or very small numbers
                assert isinstance(r_max_px, float), "r_max_px should be a float for division" #
                assert isinstance(avg_cladding_ellipse_radius, float) and avg_cladding_ellipse_radius != 0, \
                    "avg_cladding_ellipse_radius must be a non-zero float for division" #
                scale_factor_max = r_max_px / avg_cladding_ellipse_radius #
                assert isinstance(r_min_px, float), "r_min_px should be a float for division" #
                scale_factor_min = r_min_px / avg_cladding_ellipse_radius #
            else:
                scale_factor_max = 1.0
                scale_factor_min = 0.0
            
            # Axes for cv2.ellipse should be (major_axis/2, minor_axis/2) or (width, height)
            # The cv2.fitEllipse returns (minorAxis, majorAxis) which are full lengths.
            # cv2.ellipse expects (majorAxisRadius, minorAxisRadius) or rather (axes_width/2, axes_height/2)
            # Here, base_minor_axis and base_major_axis are full lengths.
            # For cv2.ellipse, the axes tuple is (width, height) which are full lengths.
            outer_ellipse_axes_tuple = (int(base_minor_axis * scale_factor_max), int(base_major_axis * scale_factor_max))
            inner_ellipse_axes_tuple = (int(base_minor_axis * scale_factor_min), int(base_major_axis * scale_factor_min))
            
            if r_max_px > 0 and outer_ellipse_axes_tuple[0] > 0 and outer_ellipse_axes_tuple[1] > 0:
                 # Color for cv2.ellipse
                 cv2.ellipse(zone_mask_np, (base_center_ell, outer_ellipse_axes_tuple, base_angle), (255,), -1)
            
            if r_min_px > 0 and inner_ellipse_axes_tuple[0] > 0 and inner_ellipse_axes_tuple[1] > 0:
                 temp_inner_mask = np.zeros_like(zone_mask_np)
                 # Color for cv2.ellipse
                 cv2.ellipse(temp_inner_mask, (base_center_ell, inner_ellipse_axes_tuple, base_angle), (255,), -1)
                 zone_mask_np = cv2.subtract(zone_mask_np, temp_inner_mask)
            is_elliptical_zone = True

        else: # Circular zones.
            dist_sq_map = (X - cx_zone)**2 + (Y - cy_zone)**2 
            zone_mask_np = ((dist_sq_map >= r_min_px**2) & (dist_sq_map < r_max_px**2)).astype(np.uint8) * 255

        masks[name] = zone_mask_np 
        logging.debug(f"Generated mask for zone '{name}': Center={current_zone_center}, Rmin_px={r_min_px:.1f}, Rmax_px={r_max_px:.1f}, Elliptical={is_elliptical_zone}")

    return masks

# --- DETAILED ALGORITHM IMPLEMENTATIONS ---
# These will overwrite stubs if stubs were defined (i.e., if config_loader import failed)



def _lei_scratch_detection(enhanced_image: np.ndarray, kernel_lengths: List[int], angle_step: int = 15) -> np.ndarray:
    """
    Complete LEI implementation following paper Section 3.2 exactly.
    Uses dual-branch linear detector with proper response calculation.
    Optimized using vectorized operations for better performance.
    """
    h, w = enhanced_image.shape[:2]
    max_response_map = np.zeros((h, w), dtype=np.float32)
    
    # Paper specifies 12 orientations (0° to 165° in 15° steps)
    angles_deg = np.arange(0, 180, angle_step)
    
    # Pre-compute coordinate grids for vectorization
    Y, X = np.ogrid[:h, :w]
    
    for length in kernel_lengths:
        # Paper specifies branch offset of 2 pixels
        branch_offset = 2
        half_length = length // 2
        
        for angle_deg in angles_deg:
            angle_rad = np.deg2rad(angle_deg)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            # Create kernels for red and gray branches
            red_kernel = np.zeros((length, length), dtype=np.float32)
            gray_kernel = np.zeros((length, length), dtype=np.float32)
            
            # Fill kernels based on paper's dual-branch design
            for t in range(-half_length, half_length + 1):
                # Red branch (center line)
                cx = half_length + int(round(t * cos_a))
                cy = half_length + int(round(t * sin_a))
                if 0 <= cx < length and 0 <= cy < length:
                    red_kernel[cy, cx] = 1.0
                
                # Gray branches (parallel lines)
                for side in [-1, 1]:
                    gx = half_length + int(round(t * cos_a + side * branch_offset * (-sin_a)))
                    gy = half_length + int(round(t * sin_a + side * branch_offset * cos_a))
                    if 0 <= gx < length and 0 <= gy < length:
                        gray_kernel[gy, gy] = 1.0
            
            # Normalize kernels
            red_sum = np.sum(red_kernel)
            gray_sum = np.sum(gray_kernel)
            
            if red_sum > 0:
                red_kernel /= red_sum
            if gray_sum > 0:
                gray_kernel /= gray_sum
            
            # Apply filters using convolution
            if red_sum > 0 and gray_sum > 0:
                red_response = cv2.filter2D(enhanced_image.astype(np.float32), cv2.CV_32F, red_kernel)
                gray_response = cv2.filter2D(enhanced_image.astype(np.float32), cv2.CV_32F, gray_kernel)
                
                # Paper's formula: s_θ(x,y) = 2*f_r - f_g
                response = np.maximum(0, 2 * red_response - gray_response)
                max_response_map = np.maximum(max_response_map, response)
    
    # Apply Gaussian smoothing as per paper
    max_response_map = cv2.GaussianBlur(max_response_map, (3, 3), 0.5)
    
    # Normalize to 0-1 range
    if np.max(max_response_map) > 0:
        max_response_map = max_response_map / np.max(max_response_map)
    
    return max_response_map




def _gabor_defect_detection(image: np.ndarray) -> np.ndarray:
    """
    Uses Gabor filters for texture-based defect detection.
    """
    gabor_filters_list = [] # Renamed from filters to gabor_filters_list
    ksize = 31 
    sigma = 4.0 
    lambd = 10.0 
    gamma_gabor = 0.5 # Renamed gamma to gamma_gabor to avoid conflict with other gamma variables
    psi = 0 
    
    for theta in np.arange(0, np.pi, np.pi / 8): 
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma_gabor, psi, ktype=cv2.CV_32F)
        gabor_filters_list.append(kern)
    
    responses = []
    for kern in gabor_filters_list:
        filtered = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kern) 
        responses.append(np.abs(filtered)) 
    
    gabor_response = np.max(np.array(responses), axis=0) 
    gabor_response_norm = cv2.normalize(gabor_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #
    _, defect_mask = cv2.threshold(gabor_response_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return defect_mask

def _wavelet_defect_detection(image: np.ndarray) -> np.ndarray:
    """
    Uses wavelet transform for multi-resolution defect detection.
    """
    coeffs = pywt.dwt2(image.astype(np.float32), 'db4') 
    cA, (cH, cV, cD) = coeffs 
    details_magnitude = np.sqrt(cH**2 + cV**2 + cD**2)
    details_resized = cv2.resize(details_magnitude, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    details_norm = cv2.normalize(details_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #
    _, defect_mask = cv2.threshold(details_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return defect_mask


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



def _multiscale_do2mr_detection(image: np.ndarray, scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5]) -> np.ndarray:
    """
    Multi-scale DO2MR detection as suggested in the paper for improved accuracy.
    Combines results from multiple scales to reduce false positives.
    """
    h, w = image.shape[:2]
    combined_result = np.zeros((h, w), dtype=np.float32)
    
    for scale in scales:
        # Resize image
        if scale != 1.0:
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_image = image.copy()
        
        # Apply DO2MR at this scale
        # Adjust kernel size based on scale
        kernel_size = max(3, int(5 * scale))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        result = _do2mr_detection(scaled_image, kernel_size=kernel_size)
        
        # Resize result back to original size
        if scale != 1.0:
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Weight by scale (smaller scales get higher weight for small defects)
        weight = 1.0 / scale if scale > 0 else 1.0
        combined_result += result.astype(np.float32) * weight
    
    # Normalize and threshold
    combined_result = combined_result / len(scales)
    _, final_result = cv2.threshold(combined_result.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    
    return final_result


def _multiscale_defect_detection(image: np.ndarray, scales: List[float] = [0.5, 1.0, 1.5, 2.0]) -> np.ndarray:
    """
    Performs multi-scale defect detection using detailed _do2mr_detection.
    """
    h, w = image.shape[:2]
    combined_map_float = np.zeros((h, w), dtype=np.float32) 
    
    for scale_ms in scales: # Renamed scale to scale_ms
        scaled_image = image.copy() 
        if scale_ms != 1.0:
            if scale_ms <= 0: continue 
            scaled_h, scaled_w = int(h * scale_ms), int(w * scale_ms)
            if scaled_h <=0 or scaled_w <=0: continue 
            scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        
        do2mr_kernel_size_at_scale = max(3, int(5 * scale_ms)) 
        if do2mr_kernel_size_at_scale % 2 == 0: do2mr_kernel_size_at_scale +=1
        
        do2mr_result_at_scale = _do2mr_detection(scaled_image, kernel_size=do2mr_kernel_size_at_scale) 
        
        do2mr_result_resized = do2mr_result_at_scale 
        if scale_ms != 1.0:
            do2mr_result_resized = cv2.resize(do2mr_result_at_scale, (w, h), interpolation=cv2.INTER_NEAREST) 
        
        weight = 1.0 / scale_ms if scale_ms > 1 else scale_ms if scale_ms > 0 else 1.0
        combined_map_float += do2mr_result_resized.astype(np.float32) * weight
    
    combined_map_uint8 = np.zeros((h,w), dtype=np.uint8)
    if np.any(combined_map_float): 
        combined_map_uint8 = cv2.normalize(combined_map_float, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    _, final_binary_mask = cv2.threshold(combined_map_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return final_binary_mask


def _advanced_scratch_detection(image: np.ndarray) -> np.ndarray:
    """
    Advanced scratch detection using multiple techniques.
    """
    processed_image = image
    if image.dtype != np.uint8: 
        processed_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #

    h, w = processed_image.shape[:2]
    scratch_map_combined = np.zeros((h, w), dtype=np.uint8) 
    
    sobelx = cv2.Sobel(processed_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(processed_image, cv2.CV_64F, 0, 1, ksize=5)
    sobelxx = cv2.Sobel(sobelx, cv2.CV_64F, 1, 0, ksize=5) 
    sobelyy = cv2.Sobel(sobely, cv2.CV_64F, 0, 1, ksize=5) 
    sobelxy = cv2.Sobel(sobelx, cv2.CV_64F, 0, 1, ksize=5) 

    ridge_response = np.zeros_like(processed_image, dtype=np.float64)
    for r_idx in range(h): # Renamed y/r to r_idx
        for c_idx in range(w): # Renamed x/c to c_idx
            hessian_matrix = np.array([[sobelxx[r_idx,c_idx], sobelxy[r_idx,c_idx]], 
                                       [sobelxy[r_idx,c_idx], sobelyy[r_idx,c_idx]]])
            try:
                eigenvalues, _ = np.linalg.eig(hessian_matrix) 
                if eigenvalues.min() < -50: 
                    ridge_response[r_idx, c_idx] = np.abs(eigenvalues.min())
            except np.linalg.LinAlgError:
                pass 

    if np.any(ridge_response):
        ridge_response_norm = cv2.normalize(ridge_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #
        _, ridge_mask = cv2.threshold(ridge_response_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        scratch_map_combined = cv2.bitwise_or(scratch_map_combined, ridge_mask)
    
    kernel_bh_rect_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)) 
    kernel_bh_rect_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)) 
    blackhat_v = cv2.morphologyEx(processed_image, cv2.MORPH_BLACKHAT, kernel_bh_rect_vertical)
    blackhat_h = cv2.morphologyEx(processed_image, cv2.MORPH_BLACKHAT, kernel_bh_rect_horizontal)
    blackhat_combined = np.maximum(blackhat_v, blackhat_h)

    if np.any(blackhat_combined):
        _, bh_thresh = cv2.threshold(blackhat_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        scratch_map_combined = cv2.bitwise_or(scratch_map_combined, bh_thresh)
    
    edges = cv2.Canny(processed_image, 50, 150, apertureSize=3) 
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=7) 
    
    if lines is not None:
        line_mask = np.zeros_like(processed_image, dtype=np.uint8)
        for line_segment in lines: # Renamed line to line_segment
            x1, y1, x2, y2 = line_segment[0]
            # Corrected color for cv2.line
            cv2.line(line_mask, (x1, y1), (x2, y2), (255,), 1) 
        scratch_map_combined = cv2.bitwise_or(scratch_map_combined, line_mask)
    
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    scratch_map_combined = cv2.morphologyEx(scratch_map_combined, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    scratch_map_combined = cv2.morphologyEx(scratch_map_combined, cv2.MORPH_CLOSE, kernel_clean, iterations=1)
    
    return scratch_map_combined


def detect_defects(
    processed_image: np.ndarray,
    zone_mask: np.ndarray,
    zone_name: str, 
    profile_config: Dict[str, Any],
    global_algo_params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced defect detection using multi-algorithm fusion approach.
    Uses zone-specific parameters for better accuracy as per research paper.
    """
    if np.sum(zone_mask) == 0:
        logging.debug(f"Defect detection skipped for empty zone mask in zone '{zone_name}'.")
        return np.zeros_like(processed_image, dtype=np.uint8), np.zeros_like(processed_image, dtype=np.float32)

    h, w = processed_image.shape[:2]
    confidence_map = np.zeros((h, w), dtype=np.float32)
    working_image_input = cv2.bitwise_and(processed_image, processed_image, mask=zone_mask)

    # Ensure working_image is uint8 for certain operations
    if working_image_input.dtype != np.uint8:
        logging.debug(f"Original working_image for zone '{zone_name}' is {working_image_input.dtype}, will normalize to uint8 for some steps.")
        # Keep a float version if needed for some algos, but ensure uint8 for others
        working_image_uint8 = cv2.normalize(working_image_input, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        working_image_uint8 = working_image_input.copy() # Use copy to avoid modifying original processed_image slice

    working_image_for_processing = working_image_uint8 # Default to uint8 version

    if zone_name == "Core":
        logging.debug(f"Applying {zone_name}-specific preprocessing.")
        # Median blur expects uint8
        blurred_core = cv2.medianBlur(working_image_uint8, 3)
        if np.any(blurred_core[zone_mask > 0]): 
            clahe_core = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            # CLAHE expects uint8
            enhanced_region = clahe_core.apply(blurred_core) 
            working_image_for_processing = cv2.bitwise_and(enhanced_region, enhanced_region, mask=zone_mask)
        else:
            working_image_for_processing = blurred_core # Use blurred if CLAHE had no effect or region was blank
    elif zone_name == "Cladding":
        logging.debug(f"Applying {zone_name}-specific preprocessing.")
        # Bilateral filter works on uint8 or float32. If input was float, could use working_image_input
        # For consistency with uint8 path:
        working_image_for_processing = cv2.bilateralFilter(working_image_uint8, d=5, sigmaColor=50, sigmaSpace=50)
        working_image_for_processing = cv2.bitwise_and(working_image_for_processing, working_image_for_processing, mask=zone_mask)
    
    logging.debug(f"Proceeding with defect detection for zone: '{zone_name}' using specifically preprocessed image of type {working_image_for_processing.dtype}.")

    detection_cfg = profile_config.get("defect_detection", {})
    region_algos = detection_cfg.get("region_algorithms", [])
    linear_algos = detection_cfg.get("linear_algorithms", [])
    optional_algos = detection_cfg.get("optional_algorithms", [])
    algo_weights = detection_cfg.get("algorithm_weights", {})

    # Ensure working_image_for_processing is used by subsequent algorithms
    # Some algorithms might prefer float input, others uint8. Adjust as necessary.
    # For now, most stubs and detailed implementations expect uint8 or handle conversion.

    if "do2mr" in region_algos:
        # Use zone-specific gamma values as per paper
        current_do2mr_gamma = global_algo_params.get("do2mr_gamma_default", 1.5)
        if zone_name == "Core":
            current_do2mr_gamma = global_algo_params.get("do2mr_gamma_core", 1.2)
        elif zone_name == "Cladding":
            current_do2mr_gamma = global_algo_params.get("do2mr_gamma_cladding", 1.5)
        elif zone_name == "Adhesive":
            current_do2mr_gamma = global_algo_params.get("do2mr_gamma_adhesive", 2.0)
            
        # Use multi-scale DO2MR for better accuracy
        if "multiscale" in region_algos:
            multiscale_result = _multiscale_do2mr_detection(working_image_for_processing)
            confidence_map[multiscale_result > 0] += algo_weights.get("multiscale_do2mr", 0.9)
        else:
            do2mr_result = _do2mr_detection(working_image_for_processing, kernel_size=5, gamma=current_do2mr_gamma)
            confidence_map[do2mr_result > 0] += algo_weights.get("do2mr", 0.8)
        logging.debug(f"Applied DO2MR with gamma={current_do2mr_gamma} for zone '{zone_name}'")

    if "morph_gradient" in region_algos:
        kernel_size_list_mg_dd = global_algo_params.get("morph_gradient_kernel_size", [5,5]) # Renamed var
        kernel_mg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(kernel_size_list_mg_dd))
        # MORPH_GRADIENT expects single channel uint8, float32, or float64
        morph_gradient_img = cv2.morphologyEx(working_image_for_processing, cv2.MORPH_GRADIENT, kernel_mg)
        _, thresh_mg = cv2.threshold(morph_gradient_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        confidence_map[thresh_mg > 0] += algo_weights.get("morph_gradient", 0.4)
        logging.debug("Applied Morphological Gradient for region defects.")

    if "black_hat" in region_algos:
        kernel_size_list_bh_dd = global_algo_params.get("black_hat_kernel_size", [11,11]) # Renamed var
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(kernel_size_list_bh_dd))
        black_hat_img = cv2.morphologyEx(working_image_for_processing, cv2.MORPH_BLACKHAT, kernel_bh)
        _, thresh_bh = cv2.threshold(black_hat_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        confidence_map[thresh_bh > 0] += algo_weights.get("black_hat", 0.6)
        logging.debug("Applied Black-Hat Transform for region defects.")
    
    if "gabor" in region_algos:
        gabor_result = _gabor_defect_detection(working_image_for_processing) # Expects uint8 or float, handles astype(np.float32)
        confidence_map[gabor_result > 0] += algo_weights.get("gabor", 0.4)
        logging.debug("Applied Gabor filters for region defects.")
    
    if "multiscale" in region_algos:
        scales_ms_dd = global_algo_params.get("multiscale_factors", [0.5, 1.0, 1.5, 2.0]) # Renamed var
        multiscale_result = _multiscale_defect_detection(working_image_for_processing, scales_ms_dd) # _multiscale_defect_detection calls _do2mr_detection
        confidence_map[multiscale_result > 0] += algo_weights.get("multiscale", 0.6)
        logging.debug("Applied multi-scale detection for region defects.")

    if "lbp" in region_algos:
            from skimage.feature import local_binary_pattern
            radius = 1
            n_points = 8 * radius
            lbp = local_binary_pattern(working_image_for_processing, n_points, radius, method='uniform')
            lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            _, lbp_mask = cv2.threshold(lbp_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            confidence_map[lbp_mask > 0] += algo_weights.get("lbp", 0.3)
            logging.debug("Applied LBP texture analysis for defects.")
    
    if "lei_advanced" in linear_algos:
            # Step 1: Image Enhancement using histogram equalization (Paper Section 3.2)
            enhanced_for_lei = cv2.equalizeHist(working_image_for_processing)
            logging.debug("Applied histogram equalization for LEI scratch detection")
            
            # Step 2: Scratch Searching with linear detectors at multiple orientations
            lei_kernel_lengths = global_algo_params.get("lei_kernel_lengths", [11, 17, 23])
            angle_step_deg = global_algo_params.get("lei_angle_step_deg", 15)
            
            # Create response maps for each orientation
            max_response_map = np.zeros_like(enhanced_for_lei, dtype=np.float32)
            
            for kernel_length in lei_kernel_lengths:
                for angle_deg in range(0, 180, angle_step_deg):
                    # Create linear kernel for current orientation
                    angle_rad = np.radians(angle_deg)
                    
                    # Create a line kernel
                    kernel = np.zeros((kernel_length, kernel_length), dtype=np.float32)
                    center = kernel_length // 2
                    
                    # Draw line through center at specified angle
                    for i in range(kernel_length):
                        x = int(center + (i - center) * np.cos(angle_rad))
                        y = int(center + (i - center) * np.sin(angle_rad))
                        if 0 <= x < kernel_length and 0 <= y < kernel_length:
                            kernel[y, x] = 1.0
                    
                    # Normalize kernel
                    kernel_sum = np.sum(kernel)
                    if kernel_sum > 0:
                        kernel /= kernel_sum
                    
                    # Apply filter to get response for this orientation
                    response = cv2.filter2D(enhanced_for_lei.astype(np.float32), cv2.CV_32F, kernel)
                    max_response_map = np.maximum(max_response_map, response)
            
            # Step 3: Scratch Segmentation - threshold the response map
            # Normalize response map to 0-255 range for thresholding
            max_response_uint8 = cv2.normalize(max_response_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            _, scratch_binary = cv2.threshold(max_response_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Step 4: Result Synthesis is handled by the confidence map
            # Clean up the result with morphological operations
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # Horizontal kernel for scratches
            scratch_binary_cleaned = cv2.morphologyEx(scratch_binary, cv2.MORPH_CLOSE, kernel_clean)
            
            confidence_map[scratch_binary_cleaned > 0] += algo_weights.get("lei_advanced", 0.8)
            logging.debug("Completed LEI scratch detection")
    
    if "advanced_scratch" in linear_algos:
        # _advanced_scratch_detection handles internal normalization if not uint8
        advanced_scratch_result = _advanced_scratch_detection(working_image_for_processing)
        confidence_map[advanced_scratch_result > 0] += algo_weights.get("advanced_scratch", 0.7)
        logging.debug("Applied advanced scratch detection.")

    if "skeletonization" in linear_algos:
        img_for_canny_skel = working_image_for_processing # Use the preprocessed image for the zone
        if img_for_canny_skel.dtype != np.uint8: # Canny prefers uint8
            img_for_canny_skel = cv2.normalize(img_for_canny_skel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #

        edges_skel_dd = cv2.Canny(img_for_canny_skel, 50, 150, apertureSize=global_algo_params.get("sobel_scharr_ksize",3)) # Renamed var
        try:
            thinned_edges = cv2.ximgproc.thinning(edges_skel_dd, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            dilation_kernel_size_list_skel_dd = global_algo_params.get("skeletonization_dilation_kernel_size",[3,3]) # Renamed var
            dilation_kernel_skel_dd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(dilation_kernel_size_list_skel_dd)) # Renamed var
            thinned_edges_dilated = cv2.dilate(thinned_edges, dilation_kernel_skel_dd, iterations=1)
            confidence_map[thinned_edges_dilated > 0] += algo_weights.get("skeletonization", 0.3)
            logging.debug("Applied Canny + Skeletonization for linear defects.")
        except AttributeError:
            logging.warning("cv2.ximgproc.thinning not available (opencv-contrib-python needed). Skipping skeletonization.")
        except cv2.error as e_cv_error: # Renamed e to e_cv_error
            logging.warning(f"OpenCV error during skeletonization (thinning): {e_cv_error}. Skipping.")

    if "wavelet" in optional_algos:
        import pywt
        coeffs = pywt.dwt2(working_image_for_processing.astype(np.float32), 'db4')
        cA, (cH, cV, cD) = coeffs
        details_magnitude = np.sqrt(cH**2 + cV**2 + cD**2)
        details_resized = cv2.resize(details_magnitude, working_image_for_processing.shape[::-1])
        details_norm = cv2.normalize(details_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, wavelet_mask = cv2.threshold(details_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        confidence_map[wavelet_mask > 0] += algo_weights.get("wavelet", 0.4)
        logging.debug("Applied wavelet transform for defect detection.")
    
    if global_algo_params.get("scratch_dataset_path") and "dataset_scratch" in optional_algos:
        try:
            from scratch_dataset_handler import ScratchDatasetHandler 
            dataset_handler = ScratchDatasetHandler(global_algo_params["scratch_dataset_path"])
            # Ensure working_image_for_processing is suitable for augment_scratch_detection
            scratch_prob = dataset_handler.augment_scratch_detection(working_image_for_processing) 
            confidence_map += scratch_prob * algo_weights.get("dataset_scratch", 0.5)
            logging.debug("Applied scratch dataset augmentation.")
        except ImportError:
            logging.warning("ScratchDatasetHandler module not found. Skipping scratch dataset integration.")
        except Exception as e_sds: # Renamed e to e_sds
            logging.warning(f"Scratch dataset integration failed: {e_sds}")

    if "anomaly" in optional_algos and ANOMALY_DETECTION_AVAILABLE: 
        try:
            anomaly_detector = AnomalyDetector(global_algo_params.get("anomaly_model_path"))
            # Ensure working_image_for_processing is suitable for detect_anomalies (e.g. BGR or Grayscale as expected by model)
            # AnomalyDetector's detect_anomalies handles BGR/Grayscale conversion to RGB if needed
            anomaly_mask = anomaly_detector.detect_anomalies(working_image_for_processing) 
            if anomaly_mask is not None:
                confidence_map[anomaly_mask > 0] += algo_weights.get("anomaly", 0.5)
                logging.debug("Applied anomaly detection for defects.")
        except Exception as e_anomaly: # Renamed e to e_anomaly
            logging.warning(f"Anomaly detection failed: {e_anomaly}")
    elif "anomaly" in optional_algos and not ANOMALY_DETECTION_AVAILABLE:
        logging.warning("Anomaly detection algorithm specified, but AnomalyDetector module is not available.")

    confidence_threshold_from_config = detection_cfg.get("confidence_threshold", 0.9) 
    zone_adaptive_threshold_map_dd = { # Renamed var
        "Core": 0.7,      
        "Cladding": 0.9,  
        "Adhesive": 1.1,  
        "Contact": 1.2
    }
    adaptive_threshold_val_dd = zone_adaptive_threshold_map_dd.get(zone_name, confidence_threshold_from_config) # Renamed var
    assert isinstance(adaptive_threshold_val_dd, (float, int)), \
        f"adaptive_threshold_val_dd is expected to be a number, got {type(adaptive_threshold_val_dd)}" #

    # Operator issues with adaptive_threshold_val_dd being None are unlikely here due to defaults
    high_confidence_mask = (confidence_map >= adaptive_threshold_val_dd).astype(np.uint8) * 255 #
    medium_confidence_mask = ((confidence_map >= adaptive_threshold_val_dd * 0.7) & #
                              (confidence_map < adaptive_threshold_val_dd)).astype(np.uint8) * 128 #

    combined_defect_mask_dd = cv2.bitwise_or(high_confidence_mask, medium_confidence_mask) # Renamed var
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_defect_mask_dd, connectivity=8) 
    final_defect_mask_in_zone = np.zeros_like(combined_defect_mask_dd, dtype=np.uint8) 

    min_area_by_confidence_map_dd = { # Renamed var
        255: detection_cfg.get("min_defect_area_px_high_conf", 5), 
        128: detection_cfg.get("min_defect_area_px_med_conf", 10)  
    }
    default_min_area = detection_cfg.get("min_defect_area_px", 5)

    for i in range(1, num_labels): 
        area = stats[i, cv2.CC_STAT_AREA]
        component_mask = (labels == i)
        if np.any(component_mask):
            mask_val = combined_defect_mask_dd[component_mask].max() 
            min_area = min_area_by_confidence_map_dd.get(mask_val, default_min_area)
            if area >= min_area:
                final_defect_mask_in_zone[component_mask] = 255 
        else:
            logging.debug(f"Skipping empty labeled region {i} during size-based filtering.")

    def validate_defect_mask(defect_mask, original_image_for_validation, zone_name_for_validation): # Renamed args for clarity
        """Validate defects using additional criteria to reduce false positives."""
        validated_mask = np.zeros_like(defect_mask)
        
        # Find all potential defects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # Extract defect region from original image
            # Ensure ROI coordinates are within image bounds
            y_end, x_end = min(y + h, original_image_for_validation.shape[0]), min(x + w, original_image_for_validation.shape[1])
            defect_roi = original_image_for_validation[y:y_end, x:x_end]
            
            # Ensure labels ROI matches defect_roi shape
            defect_mask_roi = (labels[y:y_end, x:x_end] == i).astype(np.uint8)


            # Calculate contrast with surrounding area
            # Ensure defect_mask_roi has the same dimensions as defect_roi before using it for indexing
            if defect_roi.shape[0] != defect_mask_roi.shape[0] or defect_roi.shape[1] != defect_mask_roi.shape[1]:
                # This case should ideally not happen if stats and labels are consistent.
                # If it does, skip this defect or log an error.
                logging.warning(f"Shape mismatch between defect_roi {defect_roi.shape} and defect_mask_roi {defect_mask_roi.shape} for defect component {i}. Skipping contrast check.")
                if area >= min_area_by_confidence_map_dd.get(defect_mask[labels == i].max() if np.any(labels==i) else default_min_area, default_min_area): # Check area again before adding
                     validated_mask[labels == i] = 255 # Add defect if area is fine but contrast check failed due to shape
                continue


            surrounding_kernel_size = 5 # Define kernel size for dilation
            dilated_defect_mask_roi = cv2.dilate(defect_mask_roi, np.ones((surrounding_kernel_size,surrounding_kernel_size), np.uint8))
            surrounding_mask = dilated_defect_mask_roi - defect_mask_roi
            
            if np.sum(defect_mask_roi) > 0 and np.sum(surrounding_mask) > 0:
                # Ensure defect_roi is not empty before trying to access pixels.
                # Also ensure defect_mask_roi has some True values to avoid errors with empty slices.
                defect_pixels = defect_roi[defect_mask_roi > 0]
                surrounding_pixels = defect_roi[surrounding_mask > 0]

                if defect_pixels.size > 0 and surrounding_pixels.size > 0:
                    defect_mean = np.mean(defect_pixels)
                    surrounding_mean = np.mean(surrounding_pixels)
                    contrast = abs(defect_mean - surrounding_mean)
                else:
                    contrast = 0 # Not enough pixels for contrast calculation
                
                # Zone-specific validation thresholds
                min_contrast = {
                    "Core": 15,      # Core requires higher contrast
                    "Cladding": 10,  # Cladding moderate contrast
                    "Adhesive": 8,   # Adhesive lower contrast
                    "Contact": 5     # Contact lowest contrast
                }.get(zone_name_for_validation, 10) # Use renamed arg
                
                # Default min_area for validation, can be different from initial filtering
                min_validation_area = 5 
                
                # Validate based on contrast and size
                if contrast >= min_contrast and area >= min_validation_area :
                    validated_mask[labels == i] = 255
                elif area >= min_validation_area : # If contrast is low, but area is okay, still consider it if other filters passed
                    # This is a policy decision: what to do if contrast is low.
                    # For now, let's assume if it passed other filters, it's a candidate,
                    # but ideally, contrast should be a strong factor.
                    # If strict contrast is required, remove this elif.
                    # Let's keep it for now to be less aggressive in filtering here.
                    # validated_mask[labels == i] = 255 # (Original logic was to add it)
                    # Let's be stricter: if contrast fails, it fails validation here.
                    logging.debug(f"Defect component {i} failed contrast check ({contrast:.1f} < {min_contrast}). Area: {area}")
                else:
                    logging.debug(f"Defect component {i} failed area check during validation ({area} < {min_validation_area}) or other issue.")

            elif area >= min_area_by_confidence_map_dd.get(defect_mask[labels == i].max() if np.any(labels == i) else default_min_area, default_min_area): # If area is fine but no surrounding for contrast
                 validated_mask[labels == i] = 255 # Keep it if area is okay

        return validated_mask
    
    def validate_defect_by_size(defect_mask: np.ndarray, zone_name: str, um_per_px: Optional[float] = None) -> np.ndarray:
        """Additional size-based validation specific to zones."""
        validated_mask = defect_mask.copy()
        
        # Zone-specific minimum sizes (in pixels if no um_per_px, otherwise in um²)
        min_sizes = {
            "Core": 3 if um_per_px is None else (2.0 / (um_per_px ** 2)),  # 2 µm²
            "Cladding": 5 if um_per_px is None else (5.0 / (um_per_px ** 2)),  # 5 µm²
            "Adhesive": 10 if um_per_px is None else (20.0 / (um_per_px ** 2)),  # 20 µm²
            "Contact": 20 if um_per_px is None else (50.0 / (um_per_px ** 2))  # 50 µm²
        }
        
        min_area = min_sizes.get(zone_name, 5)
        
        # Remove components smaller than zone-specific minimum
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(validated_mask, connectivity=8)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                validated_mask[labels == i] = 0
        
        return validated_mask
    # Add validation before returning
    # The error was here: 'working_image' was not defined.
    # It should be 'working_image_for_processing' which is the most up-to-date image for the current zone.
    final_defect_mask_in_zone = validate_defect_mask(final_defect_mask_in_zone, working_image_for_processing, zone_name)
    final_defect_mask_in_zone = cv2.bitwise_and(final_defect_mask_in_zone, final_defect_mask_in_zone, mask=zone_mask)
    kernel_clean_final_dd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) # Renamed var
    final_defect_mask_in_zone = cv2.morphologyEx(final_defect_mask_in_zone, cv2.MORPH_OPEN, kernel_clean_final_dd)
    final_defect_mask_in_zone = cv2.morphologyEx(final_defect_mask_in_zone, cv2.MORPH_CLOSE, kernel_clean_final_dd)

    logging.debug(f"Defect detection fusion complete for zone '{zone_name}'. Adaptive threshold: {adaptive_threshold_val_dd:.2f}. Fallback config threshold: {confidence_threshold_from_config:.2f}.")
    return final_defect_mask_in_zone, confidence_map

def _lbp_defect_detection(gray_img: np.ndarray) -> np.ndarray:
    """
    Local Binary Pattern detection for texture-based defects
    """
    processed_gray_img = gray_img
    if gray_img.dtype != np.uint8:
        processed_gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #
    
    radius = 1
    n_points = 8 * radius 
    METHOD = 'uniform' 
    lbp = local_binary_pattern(processed_gray_img, n_points, radius, method=METHOD)
    lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #
    thresh = cv2.adaptiveThreshold(lbp_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return thresh

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s') 

    dummy_profile_config_main_test = { # Renamed var
        "preprocessing": {
            "clahe_clip_limit": 2.0, "clahe_tile_grid_size": [8, 8], 
            "gaussian_blur_kernel_size": [5, 5], "enable_illumination_correction": False 
        },
        "localization": {
            "hough_dp": 1.2, "hough_min_dist_factor": 0.15, "hough_param1": 70, 
            "hough_param2": 35, "hough_min_radius_factor": 0.08, 
            "hough_max_radius_factor": 0.45, "use_ellipse_detection": True, "use_circle_fit": True 
        },
        "defect_detection": {
            "region_algorithms": ["do2mr", "morph_gradient", "black_hat", "gabor", "multiscale", "lbp"], 
            "linear_algorithms": ["lei_advanced", "advanced_scratch", "skeletonization"],
            "optional_algorithms": ["wavelet"], 
            "confidence_threshold": 0.8, "min_defect_area_px_high_conf": 3, 
            "min_defect_area_px_med_conf": 6,  
            "algorithm_weights": { 
                "do2mr": 0.7, "morph_gradient": 0.4, "black_hat": 0.6, "gabor": 0.5, 
                "multiscale": 0.6, "lbp": 0.3, "lei_advanced": 0.8, "advanced_scratch": 0.7, 
                "skeletonization": 0.3, "wavelet": 0.4 
            }
        }
    }
    dummy_global_algo_params_main_test = get_config().get("algorithm_parameters", {}) # Renamed var
    dummy_global_algo_params_main_test.update({
        "do2mr_gamma_default": 1.5, "do2mr_gamma_core": 1.2,
        "multiscale_factors": [0.5, 1.0, 1.5], 
    })
    
    dummy_zone_defs_main_test = [ # Renamed var
        {"name": "Core", "r_min_factor": 0.0, "r_max_factor_core_relative": 1.0, "color_bgr": [255,0,0]},
        {"name": "Cladding", "r_min_factor_cladding_relative": 0.0, "r_max_factor_cladding_relative": 1.0, "color_bgr": [0,255,0]},
        {"name": "Adhesive", "r_min_factor_cladding_relative": 1.0, "r_max_factor_cladding_relative": 1.15, "color_bgr": [0,0,255]},
    ]

    test_image_path_str = "sample_fiber_image.png" 
    if not Path(test_image_path_str).exists(): 
        dummy_img_arr_h, dummy_img_arr_w = 600, 800
        dummy_img_arr = np.full((dummy_img_arr_h, dummy_img_arr_w), 128, dtype=np.uint8) 
        # Corrected colors for cv2.circle and cv2.line
        cv2.circle(dummy_img_arr, (dummy_img_arr_w//2, dummy_img_arr_h//2), 150, (200,), -1) 
        cv2.circle(dummy_img_arr, (dummy_img_arr_w//2, dummy_img_arr_h//2), 60, (50,), -1)   
        cv2.line(dummy_img_arr, (dummy_img_arr_w//2 - 100, dummy_img_arr_h//2 - 50), 
                 (dummy_img_arr_w//2 + 100, dummy_img_arr_h//2 + 50), (10,), 3) 
        cv2.circle(dummy_img_arr, (dummy_img_arr_w//2 + 50, dummy_img_arr_h//2 - 30), 15, (20,), -1) 
        noise = np.random.randint(0, 15, (dummy_img_arr_h, dummy_img_arr_w), dtype=np.uint8)
        dummy_img_arr = cv2.add(dummy_img_arr, noise)
        dummy_img_arr_bgr = cv2.cvtColor(dummy_img_arr, cv2.COLOR_GRAY2BGR) 
        cv2.imwrite(test_image_path_str, dummy_img_arr_bgr) 
        logging.info(f"Created a dummy image at {test_image_path_str} for testing.")

    logging.info(f"\n--- Test Case 1: Load and Preprocess Image: {test_image_path_str} ---")
    preprocess_result = load_and_preprocess_image(test_image_path_str, dummy_profile_config_main_test) 
    
    if preprocess_result: 
        original_bgr_test, gray_test, processed_test = preprocess_result 
        logging.info(f"Image preprocessed. Shape of processed image: {processed_test.shape}")
        
        logging.info("\n--- Test Case 2: Locate Fiber Structure ---")
        localization = locate_fiber_structure(processed_test, dummy_profile_config_main_test, original_gray_image=gray_test) 
        
        if localization: 
            logging.info(f"Fiber Localization: {localization}")
            viz_loc_img = original_bgr_test.copy()
            if 'cladding_center_xy' in localization and 'cladding_radius_px' in localization:
                cc_loc = localization['cladding_center_xy'] # Renamed var
                cr_loc = int(localization['cladding_radius_px']) # Renamed var
                cv2.circle(viz_loc_img, cc_loc, cr_loc, (0,255,0), 2) 
                if 'cladding_ellipse_params' in localization:
                     cv2.ellipse(viz_loc_img, localization['cladding_ellipse_params'], (0,255,255), 2) 
            if 'core_center_xy' in localization and 'core_radius_px' in localization:
                coc_loc = localization['core_center_xy'] # Renamed var
                cor_loc = int(localization['core_radius_px']) # Renamed var
                cv2.circle(viz_loc_img, coc_loc, cor_loc, (255,0,0), 2) 
            
            logging.info("\n--- Test Case 3: Generate Zone Masks ---")
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
                
                logging.info("\n--- Test Case 4: Detect Defects (Iterating Zones) ---")
                combined_defects_viz = np.zeros_like(processed_test, dtype=np.uint8)

                for zone_n_test, zone_m_test in zone_masks_generated.items(): # Renamed vars
                    if np.sum(zone_m_test) == 0:
                        logging.info(f"Skipping defect detection for empty zone: {zone_n_test}")
                        continue
                    
                    logging.info(f"--- Detecting defects in Zone: {zone_n_test} ---")
                    defects_mask, conf_map = detect_defects( 
                        processed_test, zone_m_test, zone_n_test, 
                        dummy_profile_config_main_test, dummy_global_algo_params_main_test
                    )
                    logging.info(f"Defect detection in '{zone_n_test}' zone complete. Found {np.sum(defects_mask > 0)} defect pixels.")
                    if np.any(defects_mask):
                        combined_defects_viz = cv2.bitwise_or(combined_defects_viz, defects_mask)
                
                final_viz_img = original_bgr_test.copy()
                final_viz_img[combined_defects_viz > 0] = [0,0,255] 
            else: 
                logging.warning("Zone mask generation failed for defect detection test.")
        else: 
            logging.error("Fiber localization failed. Cannot proceed with further tests.")
    else: 
        logging.error("Image preprocessing failed.")

    if Path(test_image_path_str).exists() and test_image_path_str == "sample_fiber_image.png":
        try:
            Path(test_image_path_str).unlink()
            logging.info(f"Cleaned up dummy image: {test_image_path_str}")
        except OSError as e_os_error: # Renamed e to e_os_error
            logging.error(f"Error removing dummy image {test_image_path_str}: {e_os_error}")
# --- The rest of image_processing.py remains unchanged ---
# The main `detect_defects` function will now automatically call the new, faster
# `_do2mr_detection` function without any other changes needed.
# ...
# ... (all other functions like _lei_scratch_detection, detect_defects, etc. are identical to original)
# ...
