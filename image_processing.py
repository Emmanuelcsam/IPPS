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
from pathlib import Path # Standard library for object-oriented path manipulation.\
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

    def _do2mr_detection(gray_img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Difference of min-max ranking filtering (DO2MR) to detect region defects.
        Returns a binary mask (0/255).
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

    def _gabor_defect_detection(gray_img: np.ndarray) -> np.ndarray:
        """
        Use Gabor filters to highlight region irregularities.
        Returns a binary mask.
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

    def _multiscale_defect_detection(gray_img: np.ndarray, scales: List[float]) -> np.ndarray:
        """
        Run a simple blob detection at multiple scales (Gaussian pyramid) to detect regions.
        Returns a binary mask where any scale detected a candidate.
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

    def _lei_scratch_detection(gray_img: np.ndarray, kernel_lengths: List[int], angle_step: int) -> np.ndarray:
        """
        LEI-inspired linear enhancement scratch detector.
        Returns a float32 response map.
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

    def _advanced_scratch_detection(gray_img: np.ndarray) -> np.ndarray:
        """
        Example: combination of Canny + Hough to detect line segments.
        Returns binary mask of detected lines.
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

    def _wavelet_defect_detection(gray_img: np.ndarray) -> np.ndarray:
        """
        Detect defects using wavelet decomposition (e.g., Haar).  
        Returns a binary mask of potential anomalies.
        """
        coeffs = pywt.dwt2(gray_img.astype(np.float32), 'haar')
        cA, (cH, cV, cD) = coeffs
        # Compute magnitude of detail coefficients
        mag = np.sqrt(cH**2 + cV**2 + cD**2)
        mag_resized = cv2.resize(mag, (gray_img.shape[1], gray_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mag_uint8 = cv2.normalize(mag_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(mag_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask


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
        
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size) # Create CLAHE object. [cite: 162]
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


    # The paper uses Gaussian filtering before DO2MR [cite: 115]
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


# --- Fiber Localization and Zoning ---
def locate_fiber_structure(
    processed_image: np.ndarray,
    profile_config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Locates the fiber cladding and core using HoughCircles or contour fitting.

    Args:
        processed_image: The preprocessed grayscale image.
        profile_config: The specific processing profile sub-dictionary from the main config.

    Returns:
        A dictionary containing localization data:
            'cladding_center_xy': (cx, cy) for cladding.
            'cladding_radius_px': Radius in pixels for cladding.
            'cladding_ellipse_params': Ellipse parameters if contour fitting was used.
            'core_center_xy': (cx, cy) for core.
            'core_radius_px': Radius in pixels for core.
            'localization_method': 'HoughCircles' or 'ContourFit'.
        Returns None if localization fails.
    """
    loc_params = profile_config.get("localization", {}) # Get localization parameters from profile.
    h, w = processed_image.shape[:2] # Get image height and width.
    min_img_dim = min(h, w) # Get the smaller dimension of the image.

    # --- Primary Method: HoughCircles for Cladding Detection ---
    # Parameters for HoughCircles, sourced from config.
    dp = loc_params.get("hough_dp", 1.2)
    min_dist_circles = int(min_img_dim * loc_params.get("hough_min_dist_factor", 0.15))
    param1 = loc_params.get("hough_param1", 70) # Upper Canny threshold for internal edge detection.
    param2 = loc_params.get("hough_param2", 35) # Accumulator threshold for circle centers.
    min_radius = int(min_img_dim * loc_params.get("hough_min_radius_factor", 0.08))
    max_radius = int(min_img_dim * loc_params.get("hough_max_radius_factor", 0.45))

    logging.debug(f"Attempting HoughCircles with dp={dp}, minDist={min_dist_circles}, p1={param1}, p2={param2}, minR={min_radius}, maxR={max_radius}")
    circles = cv2.HoughCircles( # Detect circles in the image. [cite: 89]
        processed_image, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist_circles,
        param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius
    )

    localization_result = {} # Initialize dictionary for localization results.

    if circles is not None: # If HoughCircles found circles.
        logging.info(f"HoughCircles detected {circles.shape[1]} circle(s).")
        # Assume the most prominent (often largest or first detected) is the cladding.
        # A more robust selection might involve scoring circles based on centrality or expected size.
        circles = np.uint16(np.around(circles)) # Convert circle parameters to integers.
        # Select the circle closest to the image center, if multiple, or largest radius.
        # For simplicity, taking the first one, as minDist should handle multiple concentric detections.
        # Or, choose the one with the largest radius within expected range.
        best_circle = None # Initialize variable for the best circle.
        max_r = 0 # Initialize max radius.
        img_center_x, img_center_y = w // 2, h // 2 # Calculate image center.
        min_dist_to_center = float('inf') # Initialize min distance to center.

        for i in circles[0, :]: # Iterate through detected circles.
            cx, cy, r = int(i[0]), int(i[1]), int(i[2]) # Extract circle parameters.
            # Basic scoring: prefer larger circles closer to the image center.
            dist = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2) # Calculate distance to image center.
            if r > max_r - 20 and dist < min_dist_to_center + 20 : # Heuristic for "best" circle
                 if r > max_r or dist < min_dist_to_center: # Prioritize radius then centrality
                    max_r = r # Update max radius.
                    min_dist_to_center = dist # Update min distance.
                    best_circle = i # Update best circle.
        
        if best_circle is None and len(circles[0,:]) > 0: # Fallback if scoring didn't select one.
            best_circle = circles[0,0] # Select the first detected circle.
            logging.warning("Multiple circles from Hough; took the first one as cladding.")


        if best_circle is not None: # If a best circle was selected.
            cladding_cx, cladding_cy, cladding_r = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
            localization_result['cladding_center_xy'] = (cladding_cx, cladding_cy)
            localization_result['cladding_radius_px'] = float(cladding_r)
            localization_result['localization_method'] = 'HoughCircles'
            logging.info(f"Cladding (Hough): Center=({cladding_cx},{cladding_cy}), Radius={cladding_r}px")
        else: # If no best circle could be determined.
            logging.warning("HoughCircles detected circles, but failed to select a 'best' cladding circle.")
            circles = None # Force fallback
    
    if circles is None: # If HoughCircles failed or no best circle was identified.
        logging.warning("HoughCircles failed to detect cladding. Attempting contour fitting fallback.")
        # Fallback Method: Adaptive Thresholding + Contour Fitting + Ellipse Fit
        # This is more robust for angled polishes (APC) which appear elliptical.
        # Adaptive thresholding helps segment the fiber from background.
        adaptive_thresh_block_size = profile_config.get("localization", {}).get("adaptive_thresh_block_size", 31) # Must be odd.
        adaptive_thresh_C = profile_config.get("localization", {}).get("adaptive_thresh_C", 5)
        # Ensure block size is odd
        if adaptive_thresh_block_size % 2 == 0: adaptive_thresh_block_size +=1

        thresh_img = cv2.adaptiveThreshold( # Apply adaptive thresholding.
            processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adaptive_thresh_block_size, adaptive_thresh_C # Invert to get fiber as white.
        )
        # Morphological opening to remove small noise.
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find external contours.
        if contours: # If contours are found.
            # Assume the largest contour is the fiber.
            fiber_contour = max(contours, key=cv2.contourArea)
            if len(fiber_contour) >= 5: # fitEllipse requires at least 5 points.
                ellipse_params = cv2.fitEllipse(fiber_contour) # Fit an ellipse to the contour.
                # ((center_x, center_y), (minor_axis, major_axis), angle)
                cladding_cx, cladding_cy = int(ellipse_params[0][0]), int(ellipse_params[0][1])
                # Use major axis as a proxy for "radius" or diameter.
                # For a near-circle, major and minor axes will be similar.
                cladding_major_axis = ellipse_params[1][1]
                cladding_minor_axis = ellipse_params[1][0]

                localization_result['cladding_center_xy'] = (cladding_cx, cladding_cy)
                # Effective radius for zoning might be average, min, or max axis.
                # Using average for now, but for IEC zones, specific diameter interpretation is key.
                localization_result['cladding_radius_px'] = (cladding_major_axis + cladding_minor_axis) / 4.0 # Average radius
                localization_result['cladding_ellipse_params'] = ellipse_params
                localization_result['localization_method'] = 'ContourFitEllipse'
                logging.info(f"Cladding (ContourFitEllipse): Center=({cladding_cx},{cladding_cy}), Axes=({cladding_minor_axis:.1f},{cladding_major_axis:.1f})px, Angle={ellipse_params[2]:.1f}deg")
            else: # If largest contour is too small for ellipse fitting.
                logging.error("Contour fitting failed: Largest contour has < 5 points.")
                return None # Return None.
        else: # If no contours are found.
            logging.error("Contour fitting failed: No contours found after adaptive thresholding.")
            return None # Return None.
    # --- Method 3: Circle-fit library (High accuracy) ---
    if circles is None:
        try:
            import circle_fit as cf
            logging.info("Attempting circle-fit library method...")
            
            # Edge detection for circle fitting
            edges = cv2.Canny(processed_image, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (likely the cladding)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Convert contour to points for circle-fit
                points = largest_contour.reshape(-1, 2)
                
                # Try multiple fitting methods
                fit_methods = [
                    ('algebraic', cf.least_squares_circle),
                    ('hyper', cf.hyper_fit),
                    ('taubin', cf.taubin_svd)
                ]
                
                best_fit = None
                best_residual = float('inf')
                
                for method_name, method_func in fit_methods:
                    try:
                        xc, yc, r, residual = method_func(points)
                        
                        if min_radius < r < max_radius and residual < best_residual:
                            best_fit = (xc, yc, r)
                            best_residual = residual
                            logging.info(f"Circle-fit {method_name}: Center=({xc:.1f},{yc:.1f}), Radius={r:.1f}px, Residual={residual:.3f}")
                    except Exception as e:
                        logging.debug(f"Circle-fit {method_name} failed: {e}")
                
                if best_fit:
                    xc, yc, r = best_fit
                    localization_result['cladding_center_xy'] = (int(xc), int(yc))
                    localization_result['cladding_radius_px'] = float(r)
                    localization_result['localization_method'] = 'CircleFit'
                    localization_result['fit_residual'] = best_residual
                    logging.info(f"Best circle-fit result: Center=({xc:.1f},{yc:.1f}), Radius={r:.1f}px")
                    circles = np.array([[[xc, yc, r]]], dtype=np.float32)  # Set circles for core detection
        except ImportError:
            logging.warning("circle-fit library not available")
        except Exception as e:
            logging.warning(f"Circle-fit method failed: {e}")    
            
    if circles is None and CIRCLE_FIT_AVAILABLE:
        logging.info("Attempting circle-fit library method...")
        
        # Edge detection for circle fitting
        edges = cv2.Canny(processed_image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (likely the cladding)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Convert contour to points for circle-fit
            points = largest_contour.reshape(-1, 2)
            
            try:
                # Fit circle using algebraic method
                xc, yc, r, _ = cf.least_squares_circle(points)
                
                if r > min_radius and r < max_radius:
                    circles = np.array([[[xc, yc, r]]], dtype=np.float32)
                    logging.info(f"Circle-fit successful: Center=({xc:.1f},{yc:.1f}), Radius={r:.1f}px")
            except Exception as e:
                logging.warning(f"Circle-fit failed: {e}")
                
    if 'cladding_center_xy' not in localization_result: # Final check if cladding was found.
        logging.error("Failed to localize fiber cladding by any method.")
        return None # Return None.

    # --- Core Detection ---
    # The core is a darker region within the identified cladding.
    # Create a mask for the cladding area to search for the core.
    cladding_mask = np.zeros_like(processed_image) # Initialize cladding mask.
    cl_cx, cl_cy = localization_result['cladding_center_xy'] # Get cladding center.
    if localization_result['localization_method'] == 'HoughCircles': # If using HoughCircles result.
        cl_r = int(localization_result['cladding_radius_px'] * 0.95) # Search slightly inside detected cladding.
        cv2.circle(cladding_mask, (cl_cx, cl_cy), cl_r, 255, -1) # Draw cladding circle on mask.
    elif localization_result.get('cladding_ellipse_params'): # If using ellipse result.
        ellipse_p = localization_result['cladding_ellipse_params'] # Get ellipse parameters.
        # Scale down ellipse slightly for core search.
        scaled_axes = (ellipse_p[1][0] * 0.9, ellipse_p[1][1] * 0.9)
        cv2.ellipse(cladding_mask, (ellipse_p[0], scaled_axes, ellipse_p[2]), 255, -1) # Draw ellipse on mask.
    
    # Apply the cladding mask to the original grayscale image (not the preprocessed one for edge detection).
    # Using the original gray for better core/cladding intensity difference.
    # We need the original gray image passed to this function or re-load/re-convert if only processed_image is available.
    # For now, assuming 'processed_image' is suitable if it's not overly blurred for intensity analysis.
    # A better approach would be to use the 'illum_corrected_image' from preprocessing.
    # For simplicity, this example will use processed_image, but note this caveat.
    
    image_for_core_detect = processed_image # Use the preprocessed image for core detection.
    masked_for_core = cv2.bitwise_and(image_for_core_detect, image_for_core_detect, mask=cladding_mask) # Apply mask.

    # The core is darker. Invert for Otsu if expecting core to be bright objects after inversion.
    # Or, directly use THRESH_BINARY with Otsu if core is darker region.
    # The research paper mentions the core is darker [cite: 28]
    
    # Apply Otsu's thresholding within the cladding mask to find the core.
    # The core is darker, so we look for low intensity values.
    # cv2.THRESH_BINARY will make pixels below threshold 0, above 255.
    # We want the core, which is dark.
    _, core_thresh_otsu = cv2.threshold(masked_for_core, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # After Otsu, the core (darker region) should be black (0) and cladding (brighter) white (255).
    # We need to find the black region. Invert if easier.
    # Or, analyze the histogram from Otsu to pick the lower intensity cluster.
    # The paper doesn't detail core detection beyond finding cladding center [cite: 87]
    # My advanced_fiber_inspector.py uses THRESH_BINARY_INV if core is dark.
    # If Otsu separates background and foreground (core), and core is dark, it will be one class.
    # Let's assume Otsu correctly puts core as one region and the rest of cladding as another.
    # We need the dark core.
    
    # In the provided code (advanced_fiber_inspector.py), it applies Otsu then looks for contours.
    # That implies core_thresh_otsu should result in core being white objects.
    # If core is darker, and Otsu sets dark to 0, bright to 255, we need to invert or find dark contours.
    # Let's assume `THRESH_BINARY_INV` is more direct if core is the darkest.
    _, core_thresh_inv_otsu = cv2.threshold(masked_for_core, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    core_thresh_inv_otsu = cv2.bitwise_and(core_thresh_inv_otsu, core_thresh_inv_otsu, mask=cladding_mask) # Re-mask.

    core_contours, _ = cv2.findContours(core_thresh_inv_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find core contours.

    if core_contours: # If core contours are found.
        # Select the largest contour within the cladding, or one closest to cladding center.
        best_core_contour = None # Initialize best core contour.
        min_core_dist_to_cl_center = float('inf') # Initialize min distance.
        # Filter by area and circularity as well.
        for c_contour in core_contours: # Iterate core contours.
            area = cv2.contourArea(c_contour) # Calculate area.
            if area < 10: continue # Skip tiny contours.
            M = cv2.moments(c_contour) # Calculate moments.
            if M["m00"] == 0: continue # Skip if area is zero.
            core_cx_cand, core_cy_cand = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]) # Calculate centroid.
            dist = np.sqrt((core_cx_cand - cl_cx)**2 + (core_cy_cand - cl_cy)**2) # Distance to cladding center.
            
            # Prefer contours closer to the cladding center.
            if dist < min_core_dist_to_cl_center: # If current contour is closer.
                min_core_dist_to_cl_center = dist # Update min distance.
                best_core_contour = c_contour # Update best core contour.
        
        if best_core_contour is not None: # If a best core contour was found.
            (core_cx_fit, core_cy_fit), core_r_fit = cv2.minEnclosingCircle(best_core_contour) # Fit circle to contour.
            localization_result['core_center_xy'] = (int(core_cx_fit), int(core_cy_fit))
            localization_result['core_radius_px'] = float(core_r_fit)
            logging.info(f"Core (ContourFit): Center=({int(core_cx_fit)},{int(core_cy_fit)}), Radius={core_r_fit:.1f}px")
        else: # If no suitable core contour found.
            logging.warning("Could not identify a distinct core contour within the cladding.")
            # Fallback: estimate core based on typical ratio to cladding if available.
            # This is highly dependent on having known fiber type specs.
            # For now, mark as not found.
            localization_result['core_center_xy'] = localization_result['cladding_center_xy'] # Default to cladding center.
            localization_result['core_radius_px'] = localization_result['cladding_radius_px'] * 0.4 # Default typical ratio.
            logging.warning(f"Core detection failed, defaulting to 0.4 * cladding radius.")

    else: # If no core contours found.
        logging.warning("No core contours found using Otsu within cladding mask.")
        localization_result['core_center_xy'] = localization_result['cladding_center_xy'] # Default to cladding center.
        localization_result['core_radius_px'] = localization_result['cladding_radius_px'] * 0.4 # Default typical ratio.
        logging.warning(f"Core detection failed, defaulting to 0.4 * cladding radius.")

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
        current_zone_ellipse_params = None # Store ellipse parameters if needed.

        if um_per_px and reference_cladding_diameter_um: # Micron mode with reference dimensions.
            # Convert um definitions from config (if present) or calculate from factors.
            # This example assumes factors define radii that become absolute after applying to a reference.
            # Example: Core zone's r_max_factor_core_relative=1 means its r_max_um is reference_core_diameter_um / 2.
            if name == "Core" and reference_core_diameter_um: # If zone is Core and core diameter known.
                r_min_um = zone_def.get("r_min_factor", 0.0) * (reference_core_diameter_um / 2.0) # Apply factor to radius.
                r_max_um = zone_def.get("r_max_factor_core_relative", 1.0) * (reference_core_diameter_um / 2.0) # Apply factor.
                current_zone_center = core_center # Use detected core center for core zone.
            elif name == "Cladding" and reference_cladding_diameter_um: # If zone is Cladding and cladding diameter known.
                # Cladding r_min is core's r_max.
                core_def_temp = next((zd for zd in zone_definitions if zd["name"] == "Core"), None) # Get core definition.
                r_min_um_cladding_start = 0.0 # Initialize.
                if core_def_temp and reference_core_diameter_um: # If core definition and diameter exist.
                     r_min_um_cladding_start = core_def_temp.get("r_max_factor_core_relative",1.0) * (reference_core_diameter_um / 2.0)
                
                r_min_um = zone_def.get("r_min_factor_cladding_relative", 0.0) * (reference_cladding_diameter_um / 2.0) # Factor relative to cladding itself (e.g. 0.0 for start of cladding annulus).
                r_min_um = max(r_min_um, r_min_um_cladding_start) # Ensure it starts after core.

                r_max_um = zone_def.get("r_max_factor_cladding_relative", 1.0) * (reference_cladding_diameter_um / 2.0) # Factor relative to cladding.
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
        cx, cy = current_zone_center # Get current zone center.

        if cladding_ellipse_params and (name != "Core" or not core_radius_px): # Use ellipse for non-core or if core not distinctly circular.
            # If cladding was elliptical, subsequent zones (Cladding, Adhesive, Contact) should also be elliptical.
            # The core itself might be more circular and use its own detected center/radius.
            # This logic needs refinement based on how elliptical zones are defined relative to each other.
            # For simplicity, if cladding is elliptical, we scale its axes for other zones.
            base_major_axis = cladding_ellipse_params[1][1] # Cladding major axis.
            base_minor_axis = cladding_ellipse_params[1][0] # Cladding minor axis.
            base_angle = cladding_ellipse_params[2] # Cladding angle.

            # Scale factors for r_min and r_max need to apply to the base ellipse axes.
            # This is complex; for now, using circular approximations even if cladding was elliptical for outer zones,
            # unless the config explicitly supports elliptical factors.
            # The prompt indicates "circular or elliptical", implying fallback to ellipse fit.
            # If an ellipse was fitted for cladding, it's better to use it for all zones.
            # The r_min_px, r_max_px calculated above were effectively radii.
            # For an ellipse, these would correspond to average radii or scaling factors for axes.

            # Create outer ellipse mask
            outer_ellipse_axes_max = (int(r_max_px * 2 * (base_minor_axis / (base_major_axis + base_minor_axis))), int(r_max_px * 2 * (base_major_axis / (base_major_axis + base_minor_axis))))
            # Create inner ellipse mask
            inner_ellipse_axes_min = (int(r_min_px * 2 * (base_minor_axis / (base_major_axis + base_minor_axis))), int(r_min_px * 2 * (base_major_axis + base_minor_axis)))
            
            if r_max_px > 0 : # Only draw if radius is positive.
                 cv2.ellipse(zone_mask_np, (current_zone_center, outer_ellipse_axes_max, base_angle), 255, -1) # Draw outer ellipse.
            if r_min_px > 0 : # Only subtract if inner radius is positive.
                 temp_inner_mask = np.zeros_like(zone_mask_np) # Temporary mask for inner ellipse.
                 cv2.ellipse(temp_inner_mask, (current_zone_center, inner_ellipse_axes_min, base_angle), 255, -1) # Draw inner ellipse.
                 zone_mask_np = cv2.subtract(zone_mask_np, temp_inner_mask) # Subtract inner from outer.
            is_elliptical_zone = True # Mark as elliptical.

        else: # Circular zones.
            dist_sq_map = (X - cx)**2 + (Y - cy)**2 # Squared distance from center map.
            # Mask is 1 where r_min_px^2 <= dist_sq < r_max_px^2.
            zone_mask_np = ((dist_sq_map >= r_min_px**2) & (dist_sq_map < r_max_px**2)).astype(np.uint8) * 255

        masks[name] = zone_mask_np # Store the generated mask.
        logging.debug(f"Generated mask for zone '{name}': Center={current_zone_center}, Rmin_px={r_min_px:.1f}, Rmax_px={r_max_px:.1f}, Elliptical={is_elliptical_zone}")

    return masks # Return dictionary of zone masks.


def _lei_scratch_detection(enhanced_image: np.ndarray, kernel_lengths: List[int], angle_step: int = 15) -> np.ndarray:
    """
    Implements the Linear Enhancement Inspector (LEI) algorithm from the paper.
    Paper: sensors-18-01408-v2.pdf, Section 3.2
    Uses dual-branch approach (red and gray) for scratch detection.
    """
    h, w = enhanced_image.shape[:2]
    max_response_map = np.zeros((h, w), dtype=np.float32)
    
    for length in kernel_lengths:
        # Half-width of the linear detector
        half_width = 2  # As per paper, detector has width of 5 pixels
        
        for angle_deg in range(0, 180, angle_step):
            angle_rad = np.deg2rad(angle_deg)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            # Create response map for this orientation
            response = np.zeros_like(enhanced_image, dtype=np.float32)
            
            # Scan through each pixel
            for y in range(h):
                for x in range(w):
                    # Calculate red branch (center line) average
                    red_sum = 0
                    red_count = 0
                    
                    # Calculate gray branches (surrounding) average
                    gray_sum = 0
                    gray_count = 0
                    
                    # Sample along the line at current orientation
                    for t in range(-length//2, length//2 + 1):
                        # Center line position
                        cx = int(x + t * cos_a)
                        cy = int(y + t * sin_a)
                        
                        if 0 <= cx < w and 0 <= cy < h:
                            red_sum += enhanced_image[cy, cx]
                            red_count += 1
                            
                            # Sample surrounding pixels (gray branches)
                            for offset in [-half_width, half_width]:
                                gx = int(x + t * cos_a - offset * sin_a)
                                gy = int(y + t * sin_a + offset * cos_a)
                                
                                if 0 <= gx < w and 0 <= gy < h:
                                    gray_sum += enhanced_image[gy, gx]
                                    gray_count += 1
                    
                    # Calculate LEI response as per paper equation (7)
                    if red_count > 0 and gray_count > 0:
                        f_r = red_sum / red_count
                        f_g = gray_sum / gray_count
                        response[y, x] = 2 * f_r - f_g
            
            # Update maximum response
            max_response_map = np.maximum(max_response_map, response)
    
    return max_response_map

def _gabor_defect_detection(image: np.ndarray) -> np.ndarray:
    """
    Uses Gabor filters for texture-based defect detection.
    Particularly good for detecting periodic defects and scratches.
    """
    filters = []
    ksize = 31
    sigma = 4.0
    lambd = 10.0
    gamma = 0.5
    
    # Create Gabor filters at different orientations
    for theta in np.arange(0, np.pi, np.pi / 8):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        filters.append(kern)
    
    # Apply filters and combine responses
    responses = []
    for kern in filters:
        filtered = cv2.filter2D(image, cv2.CV_32F, kern)
        responses.append(np.abs(filtered))
    
    # Combine responses - use maximum response across all orientations
    gabor_response = np.max(responses, axis=0)
    
    # Threshold to get defect mask
    _, defect_mask = cv2.threshold(gabor_response.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return defect_mask

def _wavelet_defect_detection(image: np.ndarray) -> np.ndarray:
    """
    Uses wavelet transform for multi-resolution defect detection.
    Effective for detecting defects at different scales.
    """
    # Perform 2D discrete wavelet transform
    coeffs = pywt.dwt2(image, 'db4')
    cA, (cH, cV, cD) = coeffs
    
    # Combine detail coefficients
    details = np.sqrt(cH**2 + cV**2 + cD**2)
    
    # Reconstruct at original size
    details_resized = cv2.resize(details, (image.shape[1], image.shape[0]))
    
    # Threshold to get defects
    _, defect_mask = cv2.threshold(details_resized.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return defect_mask


def _multiscale_defect_detection(image: np.ndarray, scales: List[float] = [0.5, 1.0, 1.5, 2.0]) -> np.ndarray:
    """
    Performs multi-scale defect detection for improved accuracy.
    Based on scale-space theory for robust defect detection.
    """
    h, w = image.shape[:2]
    combined_map = np.zeros((h, w), dtype=np.float32)
    
    for scale in scales:
        # Resize image
        if scale != 1.0:
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_image = image.copy()
        
        # Apply DO2MR at this scale
        do2mr_result = _do2mr_detection(scaled_image, kernel_size=int(5 * scale))
        
        # Resize result back to original size
        if scale != 1.0:
            do2mr_result = cv2.resize(do2mr_result, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Weight by scale (smaller scales for fine details, larger for bigger defects)
        weight = 1.0 / scale if scale > 1 else scale
        combined_map += do2mr_result.astype(np.float32) * weight
    
    # Normalize
    cv2.normalize(combined_map, combined_map, 0, 255, cv2.NORM_MINMAX)
    return combined_map.astype(np.uint8)

def _advanced_scratch_detection(image: np.ndarray) -> np.ndarray:
    """
    Advanced scratch detection using multiple techniques.
    Combines Hough lines, morphological operations, and ridge detection.
    """
    h, w = image.shape[:2]
    scratch_map = np.zeros((h, w), dtype=np.uint8)
    
    # 1. Ridge detection using Hessian
    # Calculate second derivatives
    sobelxx = cv2.Sobel(image, cv2.CV_64F, 2, 0, ksize=5)
    sobelyy = cv2.Sobel(image, cv2.CV_64F, 0, 2, ksize=5)
    sobelxy = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
    
    # Hessian eigenvalues for ridge detection
    for y in range(h):
        for x in range(w):
            hessian = np.array([[sobelxx[y,x], sobelxy[y,x]], 
                               [sobelxy[y,x], sobelyy[y,x]]])
            eigenvalues = np.linalg.eigvals(hessian)
            
            # Ridge response (large negative eigenvalue)
            if eigenvalues.min() < -10:  # Threshold
                scratch_map[y, x] = 255
    
    # 2. Morphological black-hat for dark scratches
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    
    # Rotate kernel for different orientations
    for angle in range(0, 180, 30):
        M = cv2.getRotationMatrix2D((7, 0), angle, 1)
        rotated_kernel = cv2.warpAffine(kernel, M, (15, 15))
        bh_rotated = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, rotated_kernel)
        blackhat = np.maximum(blackhat, bh_rotated)
    
    _, bh_thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    scratch_map = cv2.bitwise_or(scratch_map, bh_thresh)
    
    # 3. Line segment detection
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    
    if lines is not None:
        line_mask = np.zeros_like(image)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
        scratch_map = cv2.bitwise_or(scratch_map, line_mask)
    
    # Clean up noise
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    scratch_map = cv2.morphologyEx(scratch_map, cv2.MORPH_CLOSE, kernel_clean)
    
    return scratch_map

def _do2mr_detection(masked_zone_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Implements the DO2MR (Difference of Min-Max Ranking) algorithm from the paper.
    Paper: sensors-18-01408-v2.pdf, Section 3.1
    """
    # Create structuring element as per paper (square kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Apply min filter (erosion) - finds darkest pixel in neighborhood
    min_filtered = cv2.erode(masked_zone_image, kernel, iterations=1)
    
    # Apply max filter (dilation) - finds brightest pixel in neighborhood  
    max_filtered = cv2.dilate(masked_zone_image, kernel, iterations=1)
    
    # Calculate the residual (difference) - highlights areas of high local contrast
    residual = cv2.subtract(max_filtered, min_filtered)
    
    # Apply sigma-based thresholding as per paper (Section 3.1, Equation 6)
    mask = masked_zone_image > 0
    if np.sum(mask) == 0:
        return np.zeros_like(masked_zone_image, dtype=np.uint8)
        
    mean_res = np.mean(residual[mask])
    std_res = np.std(residual[mask])
    gamma = 1.5  # As per paper
    
    # Create binary mask using threshold
    thresh_value = mean_res + gamma * std_res
    _, defect_binary = cv2.threshold(residual, thresh_value, 255, cv2.THRESH_BINARY)
    
    # Post-processing as per paper (Section 3.1, last paragraph)
    defect_binary = cv2.medianBlur(defect_binary, 3)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    defect_binary = cv2.morphologyEx(defect_binary, cv2.MORPH_OPEN, kernel_open)
    
    return defect_binary




def detect_defects(
    processed_image: np.ndarray,
    zone_mask: np.ndarray,
    profile_config: Dict[str, Any],
    global_algo_params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced defect detection using multi-algorithm fusion approach.
    
    Returns:
        Tuple of (final_defect_mask, confidence_map)
    """
    if np.sum(zone_mask) == 0:
        logging.debug("Defect detection skipped for empty zone mask.")
        return np.zeros_like(processed_image, dtype=np.uint8), np.zeros_like(processed_image, dtype=np.float32)

    masked_zone_image = cv2.bitwise_and(processed_image, processed_image, mask=zone_mask)
    h, w = processed_image.shape[:2]
    confidence_map = np.zeros((h, w), dtype=np.float32)

    detection_cfg = profile_config.get("defect_detection", {})
    region_algos = detection_cfg.get("region_algorithms", [])
    linear_algos = detection_cfg.get("linear_algorithms", [])
    optional_algos = detection_cfg.get("optional_algorithms", [])
    algo_weights = detection_cfg.get("algorithm_weights", {})

    # A. Region Defect Analysis
    if "do2mr" in region_algos:
        do2mr_result = _do2mr_detection(masked_zone_image, kernel_size=5)
        confidence_map[do2mr_result > 0] += algo_weights.get("do2mr", 0.8)
        logging.debug("Applied DO2MR for region defects.")

    if "morph_gradient" in region_algos:
        kernel_size_list = global_algo_params.get("morph_gradient_kernel_size", [5,5])
        kernel_mg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(kernel_size_list))
        morph_gradient_img = cv2.morphologyEx(masked_zone_image, cv2.MORPH_GRADIENT, kernel_mg)
        _, thresh_mg = cv2.threshold(morph_gradient_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        confidence_map[thresh_mg > 0] += algo_weights.get("morph_gradient", 0.4)
        logging.debug("Applied Morphological Gradient for region defects.")

    if "black_hat" in region_algos:
        kernel_size_list = global_algo_params.get("black_hat_kernel_size", [11,11])
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(kernel_size_list))
        black_hat_img = cv2.morphologyEx(masked_zone_image, cv2.MORPH_BLACKHAT, kernel_bh)
        _, thresh_bh = cv2.threshold(black_hat_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        confidence_map[thresh_bh > 0] += algo_weights.get("black_hat", 0.6)
        logging.debug("Applied Black-Hat Transform for region defects.")
    
    if "gabor" in region_algos:
        gabor_result = _gabor_defect_detection(masked_zone_image)
        confidence_map[gabor_result > 0] += algo_weights.get("gabor", 0.4)
        logging.debug("Applied Gabor filters for region defects.")
    
    if "multiscale" in region_algos:
        scales = global_algo_params.get("multiscale_factors", [0.5, 1.0, 1.5, 2.0])
        multiscale_result = _multiscale_defect_detection(masked_zone_image, scales)
        confidence_map[multiscale_result > 0] += algo_weights.get("multiscale", 0.6)
        logging.debug("Applied multi-scale detection for region defects.")

    
    # B. Linear Defect Analysis (Scratches)
    if "lei_advanced" in linear_algos:
        # Apply histogram equalization for LEI
        enhanced_for_lei = cv2.equalizeHist(masked_zone_image)
        lei_kernels = global_algo_params.get("lei_kernel_lengths", [11, 17, 23])
        angle_step = global_algo_params.get("lei_angle_step_deg", 15)
        
        lei_response = _lei_scratch_detection(enhanced_for_lei, lei_kernels, angle_step)
        
        # Normalize and threshold
        cv2.normalize(lei_response, lei_response, 0, 255, cv2.NORM_MINMAX)
        _, thresh_lei = cv2.threshold(lei_response.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up
        kernel_open_lei = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        thresh_lei = cv2.morphologyEx(thresh_lei, cv2.MORPH_OPEN, kernel_open_lei)
        
        confidence_map[thresh_lei > 0] += algo_weights.get("lei_advanced", 0.8)
        logging.debug("Applied LEI-advanced method for linear defects.")
    
    if "advanced_scratch" in linear_algos:
        advanced_scratch_result = _advanced_scratch_detection(masked_zone_image)
        confidence_map[advanced_scratch_result > 0] += algo_weights.get("advanced_scratch", 0.7)
        logging.debug("Applied advanced scratch detection.")

    if "skeletonization" in linear_algos:
        edges = cv2.Canny(masked_zone_image, 50, 150, apertureSize=global_algo_params.get("sobel_scharr_ksize",3))
        try:
            thinned_edges = cv2.ximgproc.thinning(edges, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            dilation_kernel_size_list = global_algo_params.get("skeletonization_dilation_kernel_size",[3,3])
            dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(dilation_kernel_size_list))
            thinned_edges_dilated = cv2.dilate(thinned_edges, dilation_kernel, iterations=1)
            confidence_map[thinned_edges_dilated > 0] += algo_weights.get("skeletonization", 0.3)
            logging.debug("Applied Canny + Skeletonization for linear defects.")
        except AttributeError:
            logging.warning("cv2.ximgproc.thinning not available. Skipping skeletonization.")

    # C. Optional Advanced Methods
    if "wavelet" in optional_algos:
        wavelet_result = _wavelet_defect_detection(masked_zone_image)
        confidence_map[wavelet_result > 0] += algo_weights.get("wavelet", 0.4)
        logging.debug("Applied wavelet transform for defect detection.")
    
    # D. Scratch Dataset Integration
    if global_algo_params.get("scratch_dataset_path") and "dataset_scratch" in optional_algos:
        try:
            from scratch_dataset_handler import ScratchDatasetHandler
            dataset_handler = ScratchDatasetHandler(global_algo_params["scratch_dataset_path"])
            scratch_prob = dataset_handler.augment_scratch_detection(masked_zone_image)
            confidence_map += scratch_prob * algo_weights.get("dataset_scratch", 0.5)
            logging.debug("Applied scratch dataset augmentation.")
        except Exception as e:
            logging.warning(f"Scratch dataset integration failed: {e}")

    # E. Anomaly Detection
    if "anomaly" in optional_algos:
        try:
            from anomaly_detection import AnomalyDetector
            anomaly_detector = AnomalyDetector(global_algo_params.get("anomaly_model_path"))
            anomaly_mask = anomaly_detector.detect_anomalies(masked_zone_image)
            if anomaly_mask is not None:
                confidence_map[anomaly_mask > 0] += algo_weights.get("anomaly", 0.5)
                logging.debug("Applied anomaly detection for defects.")
        except Exception as e:
            logging.warning(f"Anomaly detection failed: {e}")

    # F. Fusion and Final Segmentation
    confidence_threshold = detection_cfg.get("confidence_threshold", 0.9)
    final_defect_mask_in_zone = np.where(confidence_map >= confidence_threshold, 255, 0).astype(np.uint8)
    
    # Ensure the final mask is constrained by the original zone_mask
    final_defect_mask_in_zone = cv2.bitwise_and(final_defect_mask_in_zone, final_defect_mask_in_zone, mask=zone_mask)

    # Final morphological cleaning
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    final_defect_mask_in_zone = cv2.morphologyEx(final_defect_mask_in_zone, cv2.MORPH_OPEN, kernel_clean)
    final_defect_mask_in_zone = cv2.morphologyEx(final_defect_mask_in_zone, cv2.MORPH_CLOSE, kernel_clean)
    
    logging.debug(f"Defect detection fusion complete. Confidence threshold: {confidence_threshold}.")
    return final_defect_mask_in_zone, confidence_map

def _lbp_defect_detection(gray_img: np.ndarray) -> np.ndarray:
    # radius=1, points=8
    lbp = local_binary_pattern(gray_img, P=8, R=1, method="uniform")
    # Regions with high LBP variance might indicate roughness/pits.
    thresh = cv2.threshold(lbp.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

# --- Main function for testing this module (optional) ---
if __name__ == "__main__":
    # This block is for testing the image_processing module independently.
    # It requires a sample image and a dummy config or access to config_loader.
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s') # Basic logging config.

    # --- Dummy Configuration for Testing ---
    # In a real scenario, this would come from config_loader.py
    dummy_profile_config = { # Dummy profile.
        "preprocessing": {"clahe_clip_limit": 2.0, "clahe_tile_grid_size": [8, 8], "gaussian_blur_kernel_size": [5, 5]},
        "localization": {"hough_dp": 1.2, "hough_min_dist_factor": 0.15, "hough_param1": 70, "hough_param2": 35, "hough_min_radius_factor": 0.08, "hough_max_radius_factor": 0.45},
        "defect_detection": {"region_algorithms": ["morph_gradient", "black_hat"], "linear_algorithms": ["lei_advanced", "skeletonization"], "confidence_threshold": 0.8, "algorithm_weights": {"morph_gradient": 0.4, "black_hat": 0.6, "lei_advanced": 0.7, "skeletonization": 0.3}}
    }
    dummy_global_algo_params = get_config().get("algorithm_parameters", {}) # Get global algo params from dummy config.
    
    # --- Dummy Zone Definitions for Testing (replace with actual config loading) ---
    dummy_zone_defs = [ # Dummy zone definitions.
        {"name": "Core", "r_min_factor": 0.0, "r_max_factor_core_relative": 1.0, "color_bgr": [255,0,0]},
        {"name": "Cladding", "r_min_factor_cladding_relative": 0.0, "r_max_factor_cladding_relative": 1.0, "color_bgr": [0,255,0]},
    ]

    # --- Test Case: Load and Preprocess an Image ---
    # Replace "path/to/your/sample_fiber_image.png" with an actual image path for testing.
    # Create a dummy image if you don't have one readily available.
    test_image_path_str = "sample_fiber_image.png" # Placeholder path.
    # Create a dummy image for testing if it doesn't exist
    if not Path(test_image_path_str).exists(): # Check if dummy image exists.
        dummy_img_arr = np.full((600, 800), 128, dtype=np.uint8) # Create dummy array.
        cv2.circle(dummy_img_arr, (400,300), 150, 100, -1) # Draw "cladding".
        cv2.circle(dummy_img_arr, (400,300), 60, 30, -1)   # Draw "core".
        cv2.line(dummy_img_arr, (300,250), (500,350), 0, 2) # Draw a "scratch".
        cv2.circle(dummy_img_arr, (450,280), 10, 10, -1) # Draw a "pit".
        cv2.imwrite(test_image_path_str, cv2.cvtColor(dummy_img_arr, cv2.COLOR_GRAY2BGR)) # Save dummy image.
        logging.info(f"Created a dummy image at {test_image_path_str} for testing.")

    logging.info(f"\n--- Test Case 1: Load and Preprocess Image: {test_image_path_str} ---")
    preprocess_result = load_and_preprocess_image(test_image_path_str, dummy_profile_config) # Load and preprocess.
    if preprocess_result: # If preprocessing successful.
        original_bgr_test, gray_test, processed_test = preprocess_result # Unpack results.
        logging.info(f"Image preprocessed. Shape of processed image: {processed_test.shape}")
        # cv2.imshow("Processed Test Image", processed_test); cv2.waitKey(0); cv2.destroyAllWindows() # Optional: display.

        # --- Test Case 2: Locate Fiber Structure ---
        logging.info("\n--- Test Case 2: Locate Fiber Structure ---")
        localization = locate_fiber_structure(processed_test, dummy_profile_config) # Locate fiber.
        if localization: # If localization successful.
            logging.info(f"Fiber Localization: {localization}")

            # --- Test Case 3: Generate Zone Masks ---
            logging.info("\n--- Test Case 3: Generate Zone Masks ---")
            # For testing, assume pixel mode and provide dummy user diams or None for um_per_px.
            zone_masks_generated = generate_zone_masks( # Generate zone masks.
                processed_test.shape, localization, dummy_zone_defs,
                um_per_px=None, # Test pixel mode or provide a dummy value e.g., 0.5
                user_core_diameter_um=None, # e.g. 9.0
                user_cladding_diameter_um=None # e.g. 125.0
            )
            if zone_masks_generated: # If zone masks generated.
                logging.info(f"Generated masks for zones: {list(zone_masks_generated.keys())}")
                # Example: Display the core mask
                # if "Core" in zone_masks_generated:
                #    cv2.imshow("Core Mask", zone_masks_generated["Core"]); cv2.waitKey(0); cv2.destroyAllWindows()

                # --- Test Case 4: Detect Defects in a Zone (e.g., Cladding) ---
                logging.info("\n--- Test Case 4: Detect Defects (e.g., in Cladding) ---")
                cladding_mask_test = zone_masks_generated.get("Cladding") # Get cladding mask.
                if cladding_mask_test is not None: # If cladding mask exists.
                    defects_in_cladding = detect_defects( # Detect defects in cladding.
                        processed_test, cladding_mask_test, dummy_profile_config, dummy_global_algo_params
                    )
                    logging.info(f"Defect detection in Cladding zone complete. Found {np.sum(defects_in_cladding > 0)} defect pixels.")
                    # cv2.imshow("Defects in Cladding", defects_in_cladding); cv2.waitKey(0); cv2.destroyAllWindows() # Optional: display.
                else: # If cladding mask not found.
                    logging.warning("Cladding mask not found for defect detection test.")
            else: # If zone mask generation failed.
                logging.warning("Zone mask generation failed for defect detection test.")
        else: # If localization failed.
            logging.error("Fiber localization failed. Cannot proceed with further tests.")
    else: # If preprocessing failed.
        logging.error("Image preprocessing failed.")

    # Clean up dummy image
    if Path(test_image_path_str).exists() and test_image_path_str == "sample_fiber_image.png":
        Path(test_image_path_str).unlink()
        logging.info(f"Cleaned up dummy image: {test_image_path_str}")

