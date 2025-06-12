#!/usr/bin/env python3
# image_processing.py

"""
Image Processing Engine
======================================
This module contains the core logic for processing fiber optic end face images.
It includes functions for preprocessing, fiber localization (cladding and core),
zone mask generation, and the multi-algorithm defect detection engine with fusion.
"""
# Missing imports - add these
# Import OpenCV for all core image processing tasks including filtering, morphological operations, and feature detection
import cv2 # OpenCV for all core image processing tasks.
# Import NumPy for efficient numerical computations and array manipulations throughout the image processing pipeline
import numpy as np # NumPy for numerical and array operations.
# Import type hints from typing module to enable static type checking and improve code documentation
from typing import Dict, Any, Optional, List, Tuple # Standard library for type hinting.
# Import logging module to track processing steps, errors, and performance metrics throughout execution
import logging # Standard library for logging events.
# Import Path from pathlib for modern, object-oriented file path handling across different operating systems
from pathlib import Path # Standard library for object-oriented path manipulation.
# Import PyWavelets for wavelet transform-based defect detection algorithm implementation
import pywt  # For wavelet transform
# Import ndimage from SciPy for advanced morphological operations like binary hole filling
from scipy import ndimage # For ndimage.binary_fill_holes
# Import local_binary_pattern from scikit-image for texture-based defect analysis using LBP features
from skimage.feature import local_binary_pattern # For LBP



# These will be fully available when the whole system is assembled.
# Try importing the anomaly detection module which provides ML-based defect detection capabilities
try:
    # Import AnomalyDetector class for machine learning-based anomaly detection in fiber images
    from anomaly_detection import AnomalyDetector
    # Set flag indicating anomaly detection module is available for use in the detection pipeline
    ANOMALY_DETECTION_AVAILABLE = True
# Handle case where anomaly detection module is not installed or accessible
except ImportError:
    # Set flag to False to disable anomaly detection features in the pipeline
    ANOMALY_DETECTION_AVAILABLE = False
    
# Try importing the circle fitting library for precise fiber boundary detection
try:
    # Import circle_fit module which provides least-squares circle fitting algorithms
    import circle_fit as cf #
    # Set flag indicating circle fitting library is available for enhanced localization
    CIRCLE_FIT_AVAILABLE = True
# Handle case where circle_fit library is not installed
except ImportError: #
    # Set flag to False to use fallback localization methods
    CIRCLE_FIT_AVAILABLE = False #
    
# Try importing the configuration loader module for accessing system settings
try:
    # Assuming config_loader.py is in the same directory or Python path.
    # Import get_config function to load global configuration parameters from JSON file
    from config_loader import get_config # Function to access the global configuration.
# Handle case where config_loader is not available (e.g., during standalone testing)
except ImportError:
    # Fallback for standalone testing if config_loader is not directly available.
    # In a full project, this might load a default or raise a more critical error.
    # Log warning that config loader is missing and using fallback configuration
    logging.warning("Could not import get_config from config_loader. Using dummy config for standalone testing.")
    # Define a dummy get_config function that returns minimal configuration for testing
    def get_config() -> Dict[str, Any]: # Define a dummy get_config for standalone testing.
        """Returns a dummy configuration for standalone testing."""
        # This is a simplified dummy config. In reality, it would load from config_loader.
        # For testing image_processing.py, ensure relevant keys are present.
        # Return dictionary with essential algorithm parameters for standalone testing
        return {
            # Algorithm parameters section containing all tunable parameters for defect detection
            "algorithm_parameters": {
                # Path to flat field calibration image for illumination correction (None if not available)
                "flat_field_image_path": None,
                # Kernel size for morphological gradient operation used in region defect detection
                "morph_gradient_kernel_size": [5,5],
                # Kernel size for black-hat transform used to detect dark defects on bright background
                "black_hat_kernel_size": [11,11],
                # List of kernel lengths for LEI scratch detection at multiple scales
                "lei_kernel_lengths": [11,17],
                # Angular step in degrees for LEI scratch detection orientation search
                "lei_angle_step_deg": 15,
                # Sobel/Scharr kernel size for edge detection in skeletonization algorithm
                "sobel_scharr_ksize": 3, # Used if skeletonization relies on Canny with Sobel/Scharr implicitly
                # Dilation kernel size for skeletonization post-processing to thicken detected lines
                "skeletonization_dilation_kernel_size": [3,3]
            },
            # Add other keys as needed by functions in this module for standalone testing
        }
    # --- Helper stubs/implementations for all missing functions ---
    # These stubs are defined if config_loader import fails.
    # Later, more complete local versions of these functions are defined,
    # which will overwrite these stubs if this script is run standalone.

    # Define stub version of DO2MR detection algorithm for standalone testing
    def _do2mr_detection_stub(masked_zone_image: np.ndarray, kernel_size: int = 5, gamma: float = 1.5) -> np.ndarray:
        """
        Difference of min-max ranking filtering (DO2MR) to detect region defects.
        Returns a binary mask (0/255). (STUB VERSION)
        """
        # Use the masked zone image as grayscale input for processing
        gray_img = masked_zone_image # Use the new parameter name
        # Create rectangular structuring element for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        # Apply erosion to find minimum values in local neighborhoods
        min_filt = cv2.erode(gray_img, kernel)
        # Apply dilation to find maximum values in local neighborhoods
        max_filt = cv2.dilate(gray_img, kernel)
        # Calculate residual image showing local contrast variations
        residual = cv2.subtract(max_filt, min_filt)
        # Sigma/mean threshold
        # Extract only non-zero values from residual for statistics calculation
        zone_vals = residual[residual > 0]
        # Check if any values exist in the zone to avoid division by zero
        if zone_vals.size == 0:
            # Return empty mask if no valid pixels in zone
            return np.zeros_like(gray_img, dtype=np.uint8)
        # Cast for safety with np.mean/np.std as per probs.txt suggestion
        # Calculate mean of residual values as float32 for numerical stability
        mean_res = np.mean(zone_vals.astype(np.float32))
        # Calculate standard deviation of residual values for adaptive thresholding
        std_res = np.std(zone_vals.astype(np.float32))
        # Use gamma parameter
        # Initialize binary mask for defect regions
        mask = np.zeros_like(gray_img, dtype=np.uint8)
        # Ensure (residual - mean_res) does not cause issues with uint8 if residual is uint8
        # However, residual from cv2.subtract of uint8s is uint8. Let's assume it handles saturation.
        # For safety, ensure calculation is done with appropriate types if there's risk of underflow/overflow.
        # Given earlier cast of zone_vals, mean_res and std_res are float. residual should be compatible.
        # Apply statistical threshold: pixels with values > mean + gamma*std are defects
        mask[(residual.astype(np.float32) - mean_res) > (gamma * std_res)] = 255 # Cast residual for comparison
        # Apply median blur to remove salt-and-pepper noise from binary mask
        mask = cv2.medianBlur(mask, 3)
        # Apply morphological opening to remove small isolated noise regions
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # Return final binary defect mask
        return mask

    # Define stub version of Gabor filter-based defect detection for texture analysis
    def _gabor_defect_detection_stub(image: np.ndarray) -> np.ndarray:
        """
        Use Gabor filters to highlight region irregularities.
        Returns a binary mask. (STUB VERSION)
        """
        # Use input image as grayscale for Gabor filtering
        gray_img = image # Use the new parameter name
        # Get image dimensions for creating accumulator array
        h, w = gray_img.shape
        # Initialize accumulator for maximum Gabor responses across orientations
        accum = np.zeros((h, w), dtype=np.float32)
        # Apply Gabor filters at multiple orientations (0 to 180 degrees in 45-degree steps)
        for theta in np.arange(0, np.pi, np.pi / 4):
            # Create Gabor kernel with specific parameters for texture detection
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            # Apply Gabor filter to image using 2D convolution
            filtered = cv2.filter2D(gray_img.astype(np.float32), cv2.CV_32F, kernel) # Ensure input is float for filter2D
            # Keep maximum response across all orientations
            accum = np.maximum(accum, filtered)
        # Use Otsu threshold on accumulated response
        # Normalize accumulator to 8-bit range for Otsu thresholding
        accum_uint8 = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #
        # Apply Otsu's automatic threshold to create binary mask
        _, mask = cv2.threshold(accum_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Return binary defect mask
        return mask

    # Define stub version of multi-scale defect detection for scale-invariant detection
    def _multiscale_defect_detection_stub(image: np.ndarray, scales: List[float] = [0.5, 1.0, 1.5, 2.0]) -> np.ndarray:
        """
        Run a simple blob detection at multiple scales (Gaussian pyramid) to detect regions.
        Returns a binary mask where any scale detected a candidate. (STUB VERSION)
        """
        # Use input image as grayscale for multi-scale processing
        gray_img = image # Use the new parameter name
        # Initialize accumulator for combining detections across scales
        accum = np.zeros_like(gray_img, dtype=np.uint8)
        # Process image at each scale in the pyramid
        for s_val in scales: # Renamed s to s_val to avoid conflict
            # Skip invalid scale factors
            if s_val <= 0: continue # Skip invalid scales
            # Calculate scaled dimensions
            scaled_h, scaled_w = int(gray_img.shape[0] * s_val), int(gray_img.shape[1] * s_val)
            # Skip if scaled dimensions are invalid
            if scaled_h <=0 or scaled_w <=0: continue

            # Resize image to current scale
            resized = cv2.resize(gray_img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            # Apply Gaussian blur at current scale for blob detection
            blurred = cv2.GaussianBlur(resized, (5, 5), 0)
            # Use simple threshold in scaled space
            # Apply Otsu threshold to detect blobs at current scale
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Upsample back to original
            # Resize detection mask back to original image size
            up = cv2.resize(thresh, (gray_img.shape[1], gray_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Combine detections from all scales using logical OR
            accum = cv2.bitwise_or(accum, up)
        # Return combined multi-scale detection mask
        return accum

    # Define stub version of LEI (Linear Enhancement Inspector) scratch detection
    def _lei_scratch_detection_stub(enhanced_image: np.ndarray, kernel_lengths: List[int], angle_step: int = 15) -> np.ndarray:
        """
        LEI-inspired linear enhancement scratch detector.
        Returns a float32 response map. (STUB VERSION)
        """
        # Use enhanced image as input for scratch detection
        gray_img = enhanced_image # Use the new parameter name
        # Get image dimensions for response map initialization
        h, w = gray_img.shape
        # Initialize maximum response map across all orientations
        max_resp = np.zeros((h, w), dtype=np.float32)
        # Try different kernel lengths for multi-scale scratch detection
        for length in kernel_lengths:
            # Skip invalid kernel lengths
            if length <= 0: continue # Invalid kernel length
            # Search for scratches at different orientations
            for theta_deg in range(0, 180, angle_step):
                # Create a linear kernel: a rotated line of ones of length 'length'
                # Initialize empty kernel of specified size
                kern = np.zeros((length, length), dtype=np.float32)
                # Corrected color for cv2.line based on Probs.txt and Problems.txt
                # Draw vertical line through center of kernel
                cv2.line(
                    kern,
                    (length // 2, 0),
                    (length // 2, length - 1),
                    (1.0,), thickness=1 # kern is float32, so color is (1.0,)
                )  # vertical line
                # Rotate kernel
                # Ensure center for getRotationMatrix2D is float
                # Calculate precise center point for rotation
                center_rot = (float(length -1) / 2.0, float(length-1) / 2.0) # More precise center for rotation
                # Create rotation matrix for current angle
                M = cv2.getRotationMatrix2D(center_rot, float(theta_deg), 1.0)
                # Apply rotation to linear kernel
                kern_rot = cv2.warpAffine(kern, M, (length, length), flags=cv2.INTER_LINEAR)
                # Ensure gray_img is float32 for filter2D
                # Apply rotated linear filter to detect scratches at current orientation
                resp = cv2.filter2D(gray_img.astype(np.float32), cv2.CV_32F, kern_rot)
                # Keep maximum response across all orientations
                max_resp = np.maximum(max_resp, resp)
        # Return maximum response map showing scratch likelihood
        return max_resp

    # Define stub version of advanced scratch detection using multiple techniques
    def _advanced_scratch_detection_stub(image: np.ndarray) -> np.ndarray:
        """
        Example: combination of Canny + Hough to detect line segments.
        Returns binary mask of detected lines. (STUB VERSION)
        """
        # Use input image as grayscale for edge detection
        gray_img = image # Use the new parameter name
        # Apply Canny edge detection with standard thresholds
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
        # Initialize mask for detected line segments
        mask = np.zeros_like(gray_img, dtype=np.uint8)
        # Detect line segments using probabilistic Hough transform
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=15, minLineLength=10, maxLineGap=5
        )
        # Draw detected lines on mask if any were found
        if lines is not None:
            # Draw each detected line segment on the mask
            for line_seg in lines: # Renamed line to line_seg
                # Extract line endpoints
                x1, y1, x2, y2 = line_seg[0]
                # Corrected color for cv2.line based on Probs.txt
                # Draw line segment on mask
                cv2.line(mask, (x1, y1), (x2, y2), (255,), 1) # mask is uint8
        # Return binary mask with detected line segments
        return mask

    # Define stub version of wavelet-based defect detection for frequency domain analysis
    def _wavelet_defect_detection_stub(image: np.ndarray) -> np.ndarray:
        """
        Detect defects using wavelet decomposition (e.g., Haar).  
        Returns a binary mask of potential anomalies. (STUB VERSION)
        """
        # Use input image as grayscale for wavelet analysis
        gray_img = image # Use the new parameter name
        # Apply 2D discrete wavelet transform using Haar wavelet
        coeffs = pywt.dwt2(gray_img.astype(np.float32), 'haar')
        # Extract approximation and detail coefficients
        cA, (cH, cV, cD) = coeffs
        # Compute magnitude of detail coefficients
        # Calculate magnitude combining horizontal, vertical, and diagonal details
        mag = np.sqrt(cH**2 + cV**2 + cD**2)
        # Resize magnitude map back to original image size
        mag_resized = cv2.resize(mag, (gray_img.shape[1], gray_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Normalize magnitude to 8-bit range for thresholding
        mag_uint8 = cv2.normalize(mag_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #
        # Apply Otsu threshold to create binary defect mask
        _, mask = cv2.threshold(mag_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Return binary defect mask
        return mask

    # Assign stubs to the names that will be later (potentially) overwritten by detailed implementations
    # This ensures that if this module is run standalone and config_loader is missing,
    # the _multiscale_defect_detection (stub or later detailed one) can call _do2mr_detection (stub).
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
# Define main preprocessing function that loads and prepares images for defect detection
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
    # Convert string path to Path object for better path manipulation
    image_path = Path(image_path_str) # Convert string path to Path object.
    # Validate that the path exists and is a file (not a directory)
    if not image_path.exists() or not image_path.is_file(): # Check if the path is a valid file.
        # Log error if file doesn't exist or is not a regular file
        logging.error(f"Image file not found or is not a file: {image_path}")
        # Return None to indicate loading failure
        return None # Return None if image not found or not a file.

    # Load image from disk using OpenCV (loads as BGR by default)
    original_bgr = cv2.imread(str(image_path)) # Read the image using OpenCV.
    # Check if image loading was successful (imread returns None on failure)
    if original_bgr is None: # Check if image loading failed.
        # Log error if image couldn't be loaded (e.g., corrupted file)
        logging.error(f"Failed to load image: {image_path}")
        # Return None to indicate loading failure
        return None # Return None if loading failed.
    # Log successful image loading with filename
    logging.info(f"Image '{image_path.name}' loaded successfully.")

    # Convert BGR color image to 8-bit grayscale for processing
    gray_image = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY) # Convert BGR image to 8-bit grayscale.
    # Log grayscale conversion completion
    logging.debug("Image converted to grayscale.")

    # --- Illumination Correction (CLAHE) ---
    # Get CLAHE parameters from the profile config.
    # Extract CLAHE clip limit parameter (controls contrast enhancement strength)
    clahe_clip_limit = profile_config.get("preprocessing", {}).get("clahe_clip_limit", 2.0)
    # Extract CLAHE tile grid size parameter (controls local region size)
    clahe_tile_size_list = profile_config.get("preprocessing", {}).get("clahe_tile_grid_size", [8, 8])
    # Convert tile size list to tuple, ensuring it's a valid 2-element tuple
    clahe_tile_grid_size = tuple(clahe_tile_size_list) if isinstance(clahe_tile_size_list, list) and len(clahe_tile_size_list) == 2 else (8,8)
        
    # Create CLAHE (Contrast Limited Adaptive Histogram Equalization) object for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size) # Create CLAHE object.
    # The paper mentions histogram equalization for LEI. CLAHE is generally more robust for varying illumination.
    # Apply CLAHE to enhance local contrast in the grayscale image
    illum_corrected_image = clahe.apply(gray_image) # Apply CLAHE to the grayscale image.
    # Log CLAHE application with parameters used
    logging.debug(f"CLAHE applied with clipLimit={clahe_clip_limit}, tileGridSize={clahe_tile_grid_size}.")

    #  --- Advanced Illumination Correction (if enabled) ---
    # Check if advanced illumination correction is enabled in configuration
    if profile_config.get("preprocessing", {}).get("enable_illumination_correction", False):
        # Apply advanced illumination correction using rolling ball algorithm
        illum_corrected_image = _correct_illumination(illum_corrected_image, original_dtype=gray_image.dtype) # Pass original dtype
        # Log application of advanced illumination correction
        logging.debug("Applied advanced illumination correction.")

    # --- Noise Reduction (Gaussian Blur) - Critical for minimizing false detections ---
    # Get Gaussian blur parameters from the profile config.
    # Extract Gaussian blur kernel size from configuration
    blur_kernel_list = profile_config.get("preprocessing", {}).get("gaussian_blur_kernel_size", [5, 5])
    # Convert kernel size list to tuple format required by OpenCV
    gaussian_blur_kernel_size = tuple(blur_kernel_list) if isinstance(blur_kernel_list, list) and len(blur_kernel_list) == 2 else (5,5)
    
    # Ensure kernel dimensions are odd as required by OpenCV
    # OpenCV requires odd kernel dimensions, so increment even values by 1
    gaussian_blur_kernel_size = tuple(
        k if k % 2 == 1 else k + 1 for k in gaussian_blur_kernel_size
    )
    
    # Apply Gaussian blur for denoising - paper specifies this before DO2MR
    # Sigma=0 means OpenCV will calculate it as: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    # Apply Gaussian blur to reduce noise while preserving edges
    denoised_image = cv2.GaussianBlur(illum_corrected_image, gaussian_blur_kernel_size, 0)
    
    # Optional: Apply additional median blur for salt-and-pepper noise if configured
    # Check if additional median blur is enabled for salt-and-pepper noise removal
    if profile_config.get("preprocessing", {}).get("apply_median_blur", False):
        # Get median blur kernel size from configuration
        median_kernel_size = profile_config.get("preprocessing", {}).get("median_blur_kernel_size", 3)
        # Apply median blur to remove impulse noise
        denoised_image = cv2.medianBlur(denoised_image, median_kernel_size)
        # Log median blur application with kernel size
        logging.debug(f"Additional median blur applied with kernel size {median_kernel_size}.")
    
    # Store final processed image after all preprocessing steps
    processed_image = denoised_image
    # Log completion of denoising with Gaussian blur parameters
    logging.debug(f"Denoising completed. Gaussian blur kernel size: {gaussian_blur_kernel_size}.")


    # The paper uses Gaussian filtering before DO2MR
    # Apply final Gaussian blur as specified in the research paper
    processed_image = cv2.GaussianBlur(illum_corrected_image, gaussian_blur_kernel_size, 0) # Apply Gaussian blur.
    # Log Gaussian blur application details
    logging.debug(f"Gaussian blur applied with kernel size {gaussian_blur_kernel_size}.")

    # Return tuple of original BGR, grayscale, and fully processed images
    return original_bgr, gray_image, processed_image # Return original, grayscale, and processed images.


# Define advanced illumination correction function using rolling ball algorithm
def _correct_illumination(gray_image: np.ndarray, original_dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    Performs advanced illumination correction using rolling ball algorithm.
    """
    # Estimate background using morphological closing with large kernel
    # Define large kernel size for background estimation
    kernel_size = 50
    # Create elliptical structuring element for morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Apply morphological closing to estimate background illumination
    background = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    
    # Subtract background
    # Corrected cv2.add with NumPy addition for safety as per Probs.txt
    # Subtract background from original image using 16-bit arithmetic to avoid overflow
    corrected_int16 = cv2.subtract(gray_image.astype(np.int16), background.astype(np.int16))
    # Shift values to mid-gray range to preserve both positive and negative corrections
    corrected_int16 = corrected_int16 + 128  # Shift to mid-gray
    # Clip and convert back to original dtype (passed or default uint8)
    # Clip values to valid range and convert back to original data type
    corrected = np.clip(corrected_int16, 0, 255).astype(original_dtype)
    
    # Return illumination-corrected image
    return corrected

# Define main fiber localization function that finds fiber boundaries and zones
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
    # Extract localization parameters section from configuration
    loc_params = profile_config.get("localization", {})
    # Get image height (h) and width (w).
    # Extract image dimensions for parameter calculations
    h, w = processed_image.shape[:2]
    # Determine the smaller dimension of the image.
    # Calculate minimum dimension for relative parameter sizing
    min_img_dim = min(h, w)

    # --- Initialize Parameters for HoughCircles ---
    # dp: Inverse ratio of accumulator resolution.
    # Extract HoughCircles dp parameter (accumulator resolution ratio)
    dp = loc_params.get("hough_dp", 1.2)
    # minDist: Minimum distance between centers of detected circles (factor of min_img_dim).
    # Calculate minimum distance between circle centers as fraction of image size
    min_dist_circles = int(min_img_dim * loc_params.get("hough_min_dist_factor", 0.15))
    # param1: Upper Canny threshold for internal edge detection in HoughCircles.
    # Extract Canny edge detection threshold for HoughCircles
    param1 = loc_params.get("hough_param1", 70)
    # param2: Accumulator threshold for circle centers at the detection stage.
    # Extract accumulator threshold for circle detection sensitivity
    param2 = loc_params.get("hough_param2", 35)
    # minRadius: Minimum circle radius to detect (factor of min_img_dim).
    # Calculate minimum detectable radius as fraction of image size
    min_radius_hough = int(min_img_dim * loc_params.get("hough_min_radius_factor", 0.08))
    # maxRadius: Maximum circle radius to detect (factor of min_img_dim).
    # Calculate maximum detectable radius as fraction of image size
    max_radius_hough = int(min_img_dim * loc_params.get("hough_max_radius_factor", 0.45))

    # Initialize dictionary to store localization results.
    # Create empty dictionary to store fiber localization results
    localization_result = {}
    # Log the parameters being used for HoughCircles.
    # Log HoughCircles parameters for debugging and optimization
    logging.debug(f"Attempting HoughCircles with dp={dp}, minDist={min_dist_circles}, p1={param1}, p2={param2}, minR={min_radius_hough}, maxR={max_radius_hough}")
    
# --- Primary Method: HoughCircles for Cladding Detection ---
    # Parameters fine-tuned for fiber optic end face detection:
    # - dp: Inverse ratio of accumulator resolution to image resolution (1.0 = same, 2.0 = half)
    # - minDist: Minimum distance between detected circle centers (prevents multiple detections of same fiber)
    # - param1: Upper threshold for Canny edge detector (higher = fewer edges)
    # - param2: Accumulator threshold for circle centers (lower = more circles detected)
    # - minRadius/maxRadius: Expected fiber size range in pixels
    
    # Log detailed HoughCircles parameters for debugging
    logging.debug(f"HoughCircles parameters: dp={dp}, minDist={min_dist_circles}, "
                  f"param1={param1}, param2={param2}, minRadius={min_radius_hough}, maxRadius={max_radius_hough}")
    
    # Apply HoughCircles transform to detect circular fiber boundaries
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
    # Check if HoughCircles failed or cladding center wasn't found
    if circles is None or 'cladding_center_xy' not in localization_result:
        # Log attempt to use enhanced detection methods
        logging.info("Attempting enhanced multi-method circle detection")
        
        # Method 1: Template matching for circular patterns
        # Check if image is large enough for template matching
        if processed_image.shape[0] > 100 and processed_image.shape[1] > 100:
            # Create circular template
            # Calculate template radius as 30% of minimum image dimension
            template_radius = int(min_img_dim * 0.3)
            # Create blank template image
            template = np.zeros((template_radius*2, template_radius*2), dtype=np.uint8)
            # Draw filled circle on template
            cv2.circle(template, (template_radius, template_radius), template_radius, 255, -1)
            
            # Match template at multiple scales
            # Initialize variables to track best match
            best_match_val = 0
            best_match_loc = None
            best_match_scale = 1.0
            
            # Try different scales from 50% to 150% of original template size
            for scale in np.linspace(0.5, 1.5, 11):
                # Resize template to current scale
                scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
                # Skip if template is larger than image
                if scaled_template.shape[0] > processed_image.shape[0] or scaled_template.shape[1] > processed_image.shape[1]:
                    continue
                    
                # Perform normalized cross-correlation template matching
                result = cv2.matchTemplate(processed_image, scaled_template, cv2.TM_CCOEFF_NORMED)
                # Find location of best match
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                # Update best match if current is better
                if max_val > best_match_val:
                    best_match_val = max_val
                    best_match_loc = max_loc
                    best_match_scale = scale
            
            # Check if match quality is sufficient (threshold 0.6)
            if best_match_val > 0.6:  # Threshold for good match
                # Calculate detected circle parameters
                detected_radius = int(template_radius * best_match_scale)
                detected_center = (best_match_loc[0] + detected_radius, best_match_loc[1] + detected_radius)
                
                # Store template matching results
                localization_result['cladding_center_xy'] = detected_center
                localization_result['cladding_radius_px'] = float(detected_radius)
                localization_result['localization_method'] = 'TemplateMatching'
                # Log successful template matching detection
                logging.info(f"Cladding detected via template matching: Center={detected_center}, Radius={detected_radius}px")
    # Check if any circles were found by HoughCircles.
    # Process HoughCircles results if circles were detected
    if circles is not None:
        # Log the number of circles detected.
        # Log number of circles found by HoughCircles
        logging.info(f"HoughCircles detected {circles.shape[1]} circle(s).")
        # Convert circle parameters (x, y, radius) to integers.
        # Round and convert circle parameters to integers for pixel operations
        circles_int = np.uint16(np.around(circles))
        # Initialize variables to select the best circle.
        # Initialize variable to store best candidate circle
        best_circle_hough = None
        # Initialize max radius found so far.
        # Track maximum radius found for best circle selection
        max_r_hough_found = 0
        # Calculate image center coordinates.
        # Calculate image center for evaluating circle positions
        img_center_x, img_center_y = w // 2, h // 2
        # Initialize minimum distance to image center.
        # Track minimum distance to center for best circle selection
        min_dist_to_img_center = float('inf')

        # Iterate through all detected circles to find the best candidate for cladding.
        # Evaluate each detected circle as potential fiber cladding
        for c_hough in circles_int[0, :]:
            # Extract center coordinates (cx, cy) and radius (r).
            # Indexing c_hough[0] etc. is correct, Pylance warning is a false positive
            # Extract circle parameters (center x, center y, radius)
            cx_h, cy_h, r_h = int(c_hough[0]), int(c_hough[1]), int(c_hough[2]) # type: ignore #
            # Calculate distance of circle center from image center.
            # Calculate Euclidean distance from circle center to image center
            dist_h = np.sqrt((cx_h - img_center_x)**2 + (cy_h - img_center_y)**2)
            
            # Heuristic: Prefer larger circles closer to the image center.
            # This condition checks if the current circle is "better" than previously found ones.
            # Allow some tolerance when comparing circles (20 pixels for radius, 20 pixels for distance)
            if r_h > max_r_hough_found - 20 and dist_h < min_dist_to_img_center + 20 : # Allow some tolerance.
                 # Prioritize larger radius, then centrality
                 if r_h > max_r_hough_found or dist_h < min_dist_to_img_center: # Prioritize radius then centrality.
                    # Update best circle parameters
                    max_r_hough_found = r_h # Update max radius.
                    min_dist_to_img_center = dist_h # Update min distance to center.
                    best_circle_hough = c_hough # Update best circle.
        
        # If no specific "best" circle was selected through scoring, and circles were found, pick the first one as a fallback.
        # Fallback to first circle if heuristic didn't select one
        if best_circle_hough is None and len(circles_int[0,:]) > 0:
            # Select first detected circle as cladding
            best_circle_hough = circles_int[0,0] # Select the first detected circle.
            # Log warning about heuristic failure
            logging.warning("Multiple circles from Hough; heuristic didn't pinpoint one, took the first as cladding.")

        # If a best circle was determined by Hough method.
        # Process selected best circle if one was found
        if best_circle_hough is not None:
            # Extract parameters of the best circle.
            # Extract final cladding circle parameters
            cladding_cx, cladding_cy, cladding_r = int(best_circle_hough[0]), int(best_circle_hough[1]), int(best_circle_hough[2]) # type: ignore #
            # Store cladding center coordinates.
            # Save cladding center as tuple
            localization_result['cladding_center_xy'] = (cladding_cx, cladding_cy)
            # Store cladding radius in pixels.
            # Save cladding radius as float
            localization_result['cladding_radius_px'] = float(cladding_r)
            # Store the method used for localization.
            # Record that HoughCircles was successful
            localization_result['localization_method'] = 'HoughCircles'
            # Log the detected cladding parameters.
            # Log successful cladding detection details
            logging.info(f"Cladding (Hough): Center=({cladding_cx},{cladding_cy}), Radius={cladding_r}px")
        else:
            # If HoughCircles detected circles but failed to select a best one (e.g. all too small/off-center).
            # Log failure to select suitable circle from detected ones
            logging.warning("HoughCircles detected circles, but failed to select a suitable cladding circle.")
            # Ensure circles is None to trigger fallback if this path is taken.
            # Set circles to None to trigger fallback methods
            circles = None 
    else:
        # This log occurs if cv2.HoughCircles itself returns None.
        # Only log HoughCircles failure if template matching also failed
        if 'cladding_center_xy' not in localization_result: # Only log if template matching also failed
            # Log complete failure of HoughCircles
            logging.warning("HoughCircles found no circles initially (and template matching did not yield a result).")


    # --- Fallback Method 1: Adaptive Thresholding + Contour Fitting ---
    # This block is executed if the primary HoughCircles method failed to identify a suitable cladding.
    # Check if cladding hasn't been found yet
    if 'cladding_center_xy' not in localization_result:
        # Log that the system is attempting the first fallback method.
        # Log attempt to use adaptive threshold fallback
        logging.warning("Attempting adaptive threshold contour fitting fallback for cladding detection.")
        
        # Get adaptive thresholding parameters from the profile configuration.
        # Extract adaptive threshold block size parameter
        adaptive_thresh_block_size = loc_params.get("adaptive_thresh_block_size", 31) # Block size for adaptive threshold.
        # Extract adaptive threshold constant parameter
        adaptive_thresh_C = loc_params.get("adaptive_thresh_C", 5) # Constant subtracted from the mean.
        # Ensure block size is odd, as required by OpenCV.
        # Ensure block size is odd (required by OpenCV)
        if adaptive_thresh_block_size % 2 == 0: adaptive_thresh_block_size +=1

        # Determine which image to use for thresholding.
        # 'original_gray_image' (if available and less processed) might be better than 'processed_image'.
        # Use original grayscale if available (less processed, better for thresholding)
        image_for_thresh = original_gray_image if original_gray_image is not None else processed_image

        # Apply adaptive thresholding. THRESH_BINARY_INV is used if the fiber is darker than background.
        # If fiber is brighter, THRESH_BINARY should be used.
        # Apply adaptive threshold with Gaussian weighted mean
        thresh_img_adaptive = cv2.adaptiveThreshold(
            image_for_thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adaptive_thresh_block_size, adaptive_thresh_C
        )
        # Log adaptive threshold application
        logging.debug("Adaptive threshold applied for contour fallback.")
        
        # --- Enhanced Morphological Operations for Fallback ---
        # Close small gaps in the fiber structure.
        # Create large elliptical kernel for closing operations
        kernel_close_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)) # Kernel for closing.
        # Apply morphological closing to connect fiber boundary gaps
        closed_adaptive = cv2.morphologyEx(thresh_img_adaptive, cv2.MORPH_CLOSE, kernel_close_large, iterations=2) # Apply closing.
        # Log morphological closing operation
        logging.debug("Applied large closing operation to adaptive threshold result.")
        
        # Fill holes within the identified fiber structure.
        # binary_fill_holes expects a binary image (0 or 1).
        # Convert to binary format (0 or 1) for scipy function
        closed_adaptive_binary = (closed_adaptive // 255).astype(np.uint8) # Convert to 0/1.
        
        # Corrected handling of ndimage.binary_fill_holes
        # Initialize output with pre-fill image as default
        filled_adaptive = closed_adaptive # Default to pre-fill image
        # Attempt to fill holes in binary image
        try:
            # Apply binary hole filling from scipy
            fill_result = ndimage.binary_fill_holes(closed_adaptive_binary)
            # Check if fill operation succeeded
            if fill_result is not None:
                # Convert filled result back to 8-bit format
                filled_adaptive = fill_result.astype(np.uint8) * 255 # Fill holes. # Corrected from dtype=np.uint8
                # Log successful hole filling
                logging.debug("Applied hole filling to adaptive threshold result.")
            else:
                # Log if fill operation returned None
                logging.warning("ndimage.binary_fill_holes returned None. Using pre-fill image.")
                # filled_adaptive remains closed_adaptive (already set as default)
        # Handle potential errors in binary_fill_holes.
        except Exception as e_fill: # Handle potential errors in binary_fill_holes.
            # Log hole filling failure and continue with unfilled image
            logging.warning(f"Hole filling failed: {e_fill}. Proceeding with un-filled image.")
            # filled_adaptive remains closed_adaptive

        # Open to remove small noise or protrusions after filling.
        # Create smaller elliptical kernel for opening operations
        kernel_open_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)) # Smaller kernel for opening.
        # Apply morphological opening to remove small noise
        opened_adaptive = cv2.morphologyEx(filled_adaptive, cv2.MORPH_OPEN, kernel_open_small, iterations=1) # Apply opening.
        # Log morphological opening operation
        logging.debug("Applied small opening operation to adaptive threshold result.")
        
        # Find contours on the cleaned binary image.
        # Find external contours in the processed binary image
        contours_adaptive, _ = cv2.findContours(opened_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # List to store valid fiber contours found by this fallback method.
        # Initialize list for storing valid fiber candidate contours
        valid_fiber_contours = []
        # Process contours if any were found
        if contours_adaptive: # If any contours were found.
            # Log number of contours found
            logging.debug(f"Found {len(contours_adaptive)} contours after adaptive thresholding and morphology.")
            # Evaluate each contour as potential fiber boundary
            for c_adap in contours_adaptive: # Iterate through each contour.
                # Calculate contour area in pixels
                area = cv2.contourArea(c_adap) # Calculate contour area.
                
                # Filter by area: contour must be reasonably large.
                # These are relative to min/max radius expected by Hough, providing some bounds.
                # Calculate minimum expected area based on Hough parameters
                min_area_expected = (np.pi * (min_radius_hough**2)) * 0.3 # Heuristic: e.g. 30% of min Hough area.
                # Calculate maximum expected area based on Hough parameters
                max_area_expected = (np.pi * (max_radius_hough**2)) * 2.0 # Heuristic: e.g. 200% of max Hough area.
                # Check if contour area is within expected range
                if not (min_area_expected < area < max_area_expected): # If area is outside expected range.
                    # Log why contour was rejected
                    logging.debug(f"Contour skipped: Area {area:.1f}px outside range ({min_area_expected:.1f}-{max_area_expected:.1f})px.")
                    # Skip to next contour
                    continue # Skip this contour.

                # Calculate contour perimeter for circularity check
                perimeter = cv2.arcLength(c_adap, True) # Calculate contour perimeter.
                # Check for zero perimeter to avoid division by zero
                if perimeter == 0: continue # Avoid division by zero if perimeter is zero.
                # Calculate circularity metric (1.0 = perfect circle)
                circularity = 4 * np.pi * area / (perimeter**2) # Calculate circularity.
                
                # Filter by circularity: fiber end face should be somewhat circular.
                # A perfect circle has circularity 1.0.
                # Check if contour is circular enough to be a fiber
                if circularity < 0.5: # Adjust this threshold based on expected fiber shape.
                    # Log why contour was rejected
                    logging.debug(f"Contour skipped: Circularity {circularity:.2f} < 0.5.")
                    # Skip to next contour
                    continue # Skip this contour.
                
                # Add contour to valid candidates list
                valid_fiber_contours.append(c_adap) # Add valid contour to list.

            # If valid fiber contours were found by adaptive thresholding.
            # Process valid contours if any passed the filters
            if valid_fiber_contours:
                # Select the largest valid contour as the best candidate for the fiber.
                # Select largest contour as most likely fiber boundary
                fiber_contour_adaptive = max(valid_fiber_contours, key=cv2.contourArea)
                # Log selected contour properties
                logging.info(f"Selected largest valid contour (Area: {cv2.contourArea(fiber_contour_adaptive):.1f}px, Circularity: {4 * np.pi * cv2.contourArea(fiber_contour_adaptive) / (cv2.arcLength(fiber_contour_adaptive, True)**2):.2f}) for fitting.")
                
                # Check if the contour has enough points for ellipse fitting.
                # Verify contour has enough points for ellipse fitting (minimum 5)
                if len(fiber_contour_adaptive) >= 5:
                    # Check config if ellipse fitting is preferred for this profile.
                    # Check if ellipse detection is enabled in configuration
                    if loc_params.get("use_ellipse_detection", True): # Default to True if not specified
                        # Fit an ellipse to the contour.
                        # Fit ellipse to contour points
                        ellipse_params = cv2.fitEllipse(fiber_contour_adaptive)
                        # Extract ellipse parameters: center (cx, cy), axes (minor, major), angle.
                        # Indexing ellipse_params is correct, Pylance warning is likely false positive
                        # Extract ellipse center coordinates
                        cladding_cx, cladding_cy = int(ellipse_params[0][0]), int(ellipse_params[0][1])
                        # Extract ellipse minor axis length
                        cladding_minor_axis = ellipse_params[1][0] # Minor axis.
                        # Extract ellipse major axis length
                        cladding_major_axis = ellipse_params[1][1] # Major axis.
                        # Store ellipse parameters in the localization result.
                        # Save ellipse center as cladding center
                        localization_result['cladding_center_xy'] = (cladding_cx, cladding_cy)
                        # Calculate average radius from major and minor axes.
                        # Calculate average radius from ellipse axes
                        localization_result['cladding_radius_px'] = (cladding_major_axis + cladding_minor_axis) / 4.0 # Using /4 for radius from two axes
                        # Save complete ellipse parameters for zone generation
                        localization_result['cladding_ellipse_params'] = ellipse_params # Store full ellipse parameters.
                        # Record ellipse fitting as localization method
                        localization_result['localization_method'] = 'ContourFitEllipse' # Mark method.
                        # Log successful ellipse fitting details
                        logging.info(f"Cladding (ContourFitEllipse): Center=({cladding_cx},{cladding_cy}), Axes=({cladding_minor_axis:.1f},{cladding_major_axis:.1f})px, Angle={ellipse_params[2]:.1f}deg")
                    else: # If ellipse detection is disabled, fit a minimum enclosing circle.
                        # Fit minimum enclosing circle to contour
                        (cx_circ, cy_circ), r_circ = cv2.minEnclosingCircle(fiber_contour_adaptive) # Fit circle.
                        # Save circle center as cladding center
                        localization_result['cladding_center_xy'] = (int(cx_circ), int(cy_circ)) # Store center.
                        # Save circle radius as cladding radius
                        localization_result['cladding_radius_px'] = float(r_circ) # Store radius.
                        # Record circle fitting as localization method
                        localization_result['localization_method'] = 'ContourFitCircle' # Mark method.
                        # Log successful circle fitting details
                        logging.info(f"Cladding (ContourFitCircle): Center=({int(cx_circ)},{int(cy_circ)}), Radius={r_circ:.1f}px")
                else:
                    # Log if the largest contour is too small for fitting.
                    # Log insufficient points for contour fitting
                    logging.warning("Adaptive contour found, but too small for robust ellipse/circle fitting (less than 5 points).")
            else:
                # Log if no suitable contours were found after filtering.
                # Log failure to find suitable contours
                logging.warning("Adaptive thresholding did not yield any suitable fiber contours after filtering.")
        else:
            # Log if no contours were found at all by adaptive thresholding.
            # Log complete absence of contours
            logging.warning("No contours found after adaptive thresholding and initial morphological operations.")


    # --- Fallback Method 2: Circle-Fit library (if enabled and previous methods failed) ---
    # This block is executed if cladding_center_xy is still not found and circle-fit is enabled and available.
    # Check if circle-fit library should be used as final fallback
    if 'cladding_center_xy' not in localization_result and loc_params.get("use_circle_fit", True) and CIRCLE_FIT_AVAILABLE:
        # Log attempt to use circle-fit library.
        # Log circle-fit library usage attempt
        logging.info("Attempting circle-fit library method as a further fallback for cladding detection.")
        try:
            # Using 'processed_image' for Canny to get edge points for circle_fit.
            # Alternative: use 'original_gray_image' if it provides cleaner edges for this specific method.
            # Apply Canny edge detection to find fiber boundaries
            edges_for_circle_fit = cv2.Canny(processed_image, 50, 150) # Standard Canny parameters.
            
            # Find contours on these Canny edges.
            # Extract contours from edge image
            contours_cf, _ = cv2.findContours(edges_for_circle_fit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours if any were found
            if contours_cf: # If contours are found.
                # Concatenate points from all reasonably large contours for a more robust fit.
                # This helps if the fiber edge is broken into multiple segments by Canny.
                # Initialize list to collect all edge points
                all_points_for_cf = []
                # Collect points from all significant contours
                for c_cf in contours_cf:
                    # Filter out very small noisy contours
                    if cv2.contourArea(c_cf) > 50: # Heuristic: filter out very small noisy contours.
                        # Add contour points to collection
                        all_points_for_cf.extend(c_cf.reshape(-1,2))
                
                # Check if enough points for reliable circle fitting
                if len(all_points_for_cf) > 20 : # Ensure enough points for a reliable fit.
                    # Convert point list to NumPy array for circle fitting
                    points_for_cf_np = np.array(all_points_for_cf) # Convert list of points to NumPy array.
                    
                    # Define fitting methods from circle_fit library to try.
                    # List of circle fitting algorithms to try
                    fit_methods_cf = [
                        ('algebraic', cf.least_squares_circle), # Fast, but can be sensitive to outliers.
                        ('hyper', cf.hyper_fit),             # Generally more robust.
                        # ('taubin', cf.taubin_svd) # Commented out due to Pylance error: "taubin_svd" is not a known attribute
                    ]
                    
                    # Initialize variables for best fit tracking
                    best_fit_circle_cf = None # Initialize best fit circle.
                    best_residual_cf = float('inf') # Initialize best residual.
                    
                    # Iterate through defined fitting methods.
                    # Try each fitting method and keep best result
                    for method_name_cf, fit_func_cf in fit_methods_cf:
                        try:
                            # Perform circle fitting.
                            # Apply circle fitting algorithm
                            xc_cf, yc_cf, r_cf, residual_cf = fit_func_cf(points_for_cf_np)
                            # Sanity checks for the fitted circle:
                            # - Radius within expected Hough bounds (loosened slightly).
                            # - Center within image boundaries.
                            # Validate fitted circle parameters
                            if min_radius_hough * 0.7 < r_cf < max_radius_hough * 1.3 and \
                               0 < xc_cf < w and 0 < yc_cf < h:
                                # Update best fit if current has lower residual
                                if residual_cf < best_residual_cf: # If current fit is better.
                                    # Save current fit as best
                                    best_fit_circle_cf = (xc_cf, yc_cf, r_cf) # Update best fit.
                                    best_residual_cf = residual_cf # Update best residual.
                                    # Log fit quality metrics
                                    logging.debug(f"Circle-fit ({method_name_cf}): Center=({xc_cf:.1f},{yc_cf:.1f}), R={r_cf:.1f}px, Residual={residual_cf:.3f}")
                        # Handle errors during fitting.
                        except Exception as e_cf_fit: # Handle errors during fitting.
                            # Log fitting method failure
                            logging.debug(f"Circle-fit method {method_name_cf} failed: {e_cf_fit}")
                    
                    # Process best fit if one was found
                    if best_fit_circle_cf: # If a best fit was found.
                        # Extract best fit parameters
                        xc_final_cf, yc_final_cf, r_final_cf = best_fit_circle_cf # Unpack best fit parameters.
                        # Store results.
                        # Save circle-fit results as cladding parameters
                        localization_result['cladding_center_xy'] = (int(xc_final_cf), int(yc_final_cf))
                        localization_result['cladding_radius_px'] = float(r_final_cf)
                        localization_result['localization_method'] = 'CircleFitLib' # Mark method.
                        localization_result['fit_residual'] = best_residual_cf # Store fit residual.
                        # Log successful circle-fit detection
                        logging.info(f"Cladding (CircleFitLib best): Center=({int(xc_final_cf)},{int(yc_final_cf)}), Radius={r_final_cf:.1f}px, Residual={best_residual_cf:.3f}")
                    else: # If no suitable circle found by circle_fit.
                        # Log failure to fit suitable circle
                        logging.warning("Circle-fit library methods did not yield a suitable circle.")
                else: # If not enough points for fitting.
                    # Log insufficient points for circle fitting
                    logging.warning(f"Not enough contour points ({len(all_points_for_cf)}) for robust circle-fit library method.")
            else: # If no contours found for circle-fit.
                # Log absence of contours from Canny edges
                logging.warning("No contours found from Canny edges for circle-fit library method.")
        # Handle if circle_fit library is not actually available.
        except ImportError: # Handle if circle_fit library is not actually available.
            # Log import error for circle_fit
            logging.error("circle_fit library was marked as available in config but failed to import.")
            # Ensure CIRCLE_FIT_AVAILABLE is False if it fails here to prevent repeated attempts.
            # global CIRCLE_FIT_AVAILABLE # (if it's a global flag, this would be needed)
            # CIRCLE_FIT_AVAILABLE = False # This should ideally modify the global flag if this function can be re-entered.
        # Handle other errors during circle-fit process.
        except Exception as e_circle_fit_main: # Handle other errors during circle-fit process.
            # Log general circle-fit error
            logging.error(f"An error occurred during the circle-fit library attempt: {e_circle_fit_main}")

    # --- After all attempts, check if cladding was found ---
    # Check if cladding localization succeeded
    if 'cladding_center_xy' not in localization_result: # If cladding center is still not found.
        # Log critical failure to locate fiber
        logging.error("Failed to localize fiber cladding by any method.")
        # Return None to indicate complete localization failure
        return None # Critical failure, return None.

    # --- Core Detection (Proceeds if cladding was successfully found) ---
    # Ensure original_gray_image is used for better intensity distinction if available.
    # Use original grayscale for core detection (better intensity contrast)
    image_for_core_detect = original_gray_image if original_gray_image is not None else processed_image
    # After core detection, add adhesive layer detection
    
    # Check if both core and cladding have been successfully detected
    if 'core_center_xy' in localization_result and 'cladding_center_xy' in localization_result:
        # Detect adhesive layer between core and cladding
        # Extract cladding and core radii for adhesive detection
        cladding_radius = localization_result['cladding_radius_px']
        core_radius = localization_result['core_radius_px']
        
        # Create mask for the region between core and cladding
        # Initialize mask for adhesive search region
        adhesive_search_mask = np.zeros_like(image_for_core_detect, dtype=np.uint8)
        # Get cladding center coordinates, which is also used for the adhesive layer search mask
        cl_cx_core, cl_cy_core = localization_result['cladding_center_xy']
        # Draw filled circle at 95% of cladding radius
        cv2.circle(adhesive_search_mask, (cl_cx_core, cl_cy_core), int(cladding_radius * 0.95), 255, -1)
        # Subtract filled circle at 105% of core radius to create an annular mask
        cv2.circle(adhesive_search_mask, (cl_cx_core, cl_cy_core), int(core_radius * 1.05), 0, -1)
        
        # Look for adhesive layer characteristics
        # Apply mask to isolate adhesive region
        masked_adhesive_region = cv2.bitwise_and(image_for_core_detect, image_for_core_detect, mask=adhesive_search_mask)
        
        # Adhesive often appears as a ring with different intensity
        # Calculate histogram of adhesive region for intensity analysis
        hist = cv2.calcHist([masked_adhesive_region], [0], adhesive_search_mask, [256], [0, 256])
        
        # Find peaks in histogram (adhesive layer often has distinct intensity)
        # Check if histogram was calculated successfully
        if hist is not None and len(hist) > 0:
            # Simple peak detection for adhesive layer
            # Initialize list for intensity peaks
            adhesive_intensity_peaks = []
            # Search for peaks in histogram (avoiding edge bins)
            for i in range(10, 246):  # Avoid edges
                # Detect local maxima above a threshold relative to the mean histogram value
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist) * 0.5:
                    # Add peak position to list
                    adhesive_intensity_peaks.append(i)
            
            # Process detected adhesive layer peaks
            if adhesive_intensity_peaks:
                # Store adhesive layer information
                # Mark adhesive layer as detected
                localization_result['adhesive_detected'] = True
                # Store intensity peak positions
                localization_result['adhesive_intensity_range'] = adhesive_intensity_peaks
                # Log adhesive layer detection
                logging.info(f"Adhesive layer detected with intensity peaks at: {adhesive_intensity_peaks}")
                
            
    # Create a mask for the cladding area to search for the core.
    # Initialize mask for core detection within cladding
    cladding_mask_for_core_det = np.zeros_like(image_for_core_detect, dtype=np.uint8)
    # Re-extract cladding center for core search (already defined as cl_cx_core, cl_cy_core)
    cl_cx_core, cl_cy_core = localization_result['cladding_center_xy'] # Get cladding center.

    # Use the determined localization method to create the search mask for the core.
    # Reduce search radius slightly (e.g., 90-95% of cladding) to avoid cladding edge effects.
    # Define search radius factor to avoid edge effects
    search_radius_factor = 0.90 
    # Check if cladding was detected as circular
    if localization_result.get('localization_method') in ['HoughCircles', 'CircleFitLib', 'ContourFitCircle', 'TemplateMatching']:
        # Calculate core search radius as fraction of cladding radius
        cl_r_core_search = int(localization_result['cladding_radius_px'] * search_radius_factor)
        # Corrected color for cv2.circle
        # Draw filled circle mask for core search area
        cv2.circle(cladding_mask_for_core_det, (cl_cx_core, cl_cy_core), cl_r_core_search, (255,), -1)
    # Check if cladding was detected as elliptical
    elif localization_result.get('cladding_ellipse_params'): # If cladding was an ellipse.
        # Extract ellipse parameters for core search
        ellipse_p_core = localization_result['cladding_ellipse_params']
        # Scale down ellipse axes for core search.
        # Scale ellipse axes by search radius factor
        scaled_axes_core = (ellipse_p_core[1][0] * search_radius_factor, ellipse_p_core[1][1] * search_radius_factor)
        # Corrected color for cv2.ellipse
        # Draw filled ellipse mask for core search area
        cv2.ellipse(cladding_mask_for_core_det, (ellipse_p_core[0], scaled_axes_core, ellipse_p_core[2]), (255,), -1)
    else: # Should not happen if cladding_center_xy is present, but as a safeguard.
        # Log error if localization method is unknown
        logging.error("Cladding localization method unknown for core detection masking. Cannot proceed with core detection.")
        # Return with at least cladding info, core will be marked as not found or estimated.
        # Default core to cladding center if detection fails
        localization_result['core_center_xy'] = localization_result['cladding_center_xy'] # Default to cladding center.
        # Estimate core radius as typical 40% of cladding radius
        localization_result['core_radius_px'] = localization_result['cladding_radius_px'] * 0.4 # Default typical ratio.
        # Log core estimation due to masking failure
        logging.warning(f"Core detection failed due to masking issue, defaulting to 0.4 * cladding radius.")
        # Return partial results with estimated core
        return localization_result

    # Apply the cladding mask to the image chosen for core detection.
    # Apply mask to isolate cladding region for core search
    masked_for_core = cv2.bitwise_and(image_for_core_detect, image_for_core_detect, mask=cladding_mask_for_core_det)


    # Enhanced core detection with multiple methods
    
    # Method 1: Adaptive threshold for better local contrast handling
    # Apply adaptive threshold for core detection with local contrast adaptation
    adaptive_core = cv2.adaptiveThreshold(masked_for_core, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 31, 5)
    
    # Method 2: Otsu's thresholding
    # Apply Otsu's automatic thresholding for core detection
    _, otsu_core = cv2.threshold(masked_for_core, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Method 3: Gradient-based detection
    # Calculate horizontal gradient using Sobel operator
    gradient_x = cv2.Sobel(masked_for_core, cv2.CV_64F, 1, 0, ksize=3)
    # Calculate vertical gradient using Sobel operator
    gradient_y = cv2.Sobel(masked_for_core, cv2.CV_64F, 0, 1, ksize=3)
    # Calculate gradient magnitude from x and y components
    gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
    # Normalize gradient magnitude to 8-bit range
    gradient_mag_norm = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Threshold gradient magnitude to detect core edges
    _, gradient_core = cv2.threshold(gradient_mag_norm, 30, 255, cv2.THRESH_BINARY)
    
    # Combine methods using voting
    # Initialize combined result mask
    combined_core = np.zeros_like(masked_for_core, dtype=np.uint8)
    # Set pixels where adaptive and Otsu agree
    combined_core[(adaptive_core > 0) & (otsu_core > 0)] = 255
    # Set pixels where adaptive and gradient agree
    combined_core[(adaptive_core > 0) & (gradient_core > 0)] = 255
    # Set pixels where Otsu and gradient agree
    combined_core[(otsu_core > 0) & (gradient_core > 0)] = 255
    
    # Re-mask to ensure it's strictly within the search area
    # Apply mask to ensure core is within cladding bounds
    core_thresh_inv_otsu = cv2.bitwise_and(combined_core, combined_core, mask=cladding_mask_for_core_det)

    # Otsu's thresholding: Core is darker, so THRESH_BINARY_INV makes core white.
    # Apply Otsu thresholding with inversion (core is darker than cladding)
    _, core_thresh_inv_otsu = cv2.threshold(masked_for_core, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Re-mask to ensure it's strictly within the search area.
    # Ensure core detection stays within cladding mask
    core_thresh_inv_otsu = cv2.bitwise_and(core_thresh_inv_otsu, core_thresh_inv_otsu, mask=cladding_mask_for_core_det)
    
    # Morphological opening to remove small noise from core thresholding.
    # Create small elliptical kernel for noise removal
    kernel_core_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # Apply morphological opening to remove noise
    core_thresh_inv_otsu_opened = cv2.morphologyEx(core_thresh_inv_otsu, cv2.MORPH_OPEN, kernel_core_open, iterations=1)

    # Find contours of potential core regions.
    # Extract contours from thresholded core image
    core_contours, _ = cv2.findContours(core_thresh_inv_otsu_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process core contours if any were found
    if core_contours: # If core contours are found.
        # Initialize variables for best core selection
        best_core_contour = None # Initialize best core contour.
        min_core_dist_to_cl_center = float('inf') # Initialize min distance to cladding center.
        max_core_area = 0 # Initialize max core area (alternative selection criteria).

        # Iterate through found core contours.
        # Evaluate each contour as potential core
        for c_core_contour in core_contours:
            # Calculate contour area
            area_core = cv2.contourArea(c_core_contour) # Calculate area.
            # Min area for core (e.g., related to min_radius_hough, but for core which is smaller).
            # Skip very small contours (noise)
            if area_core < np.pi * (min_radius_hough * 0.1)**2 : continue 
            
            # Calculate contour moments for centroid
            M_core = cv2.moments(c_core_contour) # Calculate moments.
            # Skip if area is zero (degenerate contour)
            if M_core["m00"] == 0: continue # Skip if area is zero.
            # Calculate centroid coordinates from moments
            core_cx_cand, core_cy_cand = int(M_core["m10"] / M_core["m00"]), int(M_core["m01"] / M_core["m00"]) # Centroid.
            
            # Distance from this candidate core center to the established cladding center.
            # Calculate distance from candidate core to cladding center
            dist_to_cladding_center = np.sqrt((core_cx_cand - cl_cx_core)**2 + (core_cy_cand - cl_cy_core)**2)
            
            # Core should be very close to the cladding center.
            # Max allowed offset could be a small fraction of cladding radius.
            # Define maximum allowed offset from cladding center
            max_offset_allowed = localization_result['cladding_radius_px'] * 0.2 # e.g., 20% of cladding radius.

            # Check if core is reasonably centered
            if dist_to_cladding_center < max_offset_allowed: # If core is reasonably centered.
                # Prefer the largest valid contour that is well-centered.
                # Update best core if current is larger
                if area_core > max_core_area:
                    # Save current contour as best core candidate
                    max_core_area = area_core
                    best_core_contour = c_core_contour
                    min_core_dist_to_cl_center = dist_to_cladding_center # Also track its distance
        
        # Process best core contour if one was selected
        if best_core_contour is not None: # If a best core contour was selected.
            # Fit minimum enclosing circle to core contour
            (core_cx_fit, core_cy_fit), core_r_fit = cv2.minEnclosingCircle(best_core_contour) # Fit circle to contour.
            # Store core parameters.
            # Save core center coordinates
            localization_result['core_center_xy'] = (int(core_cx_fit), int(core_cy_fit))
            # Save core radius
            localization_result['core_radius_px'] = float(core_r_fit)
            # Corrected typo cy_fit to core_cy_fit
            # Log successful core detection
            logging.info(f"Core (ContourFit): Center=({int(core_cx_fit)},{int(core_cy_fit)}), Radius={core_r_fit:.1f}px")
        else: # If no suitable core contour found.
            # Log failure to find distinct core
            logging.warning("Could not identify a distinct core contour within the cladding using current criteria.")
            # Fallback: estimate core based on typical ratio to cladding.
            # Default core center to cladding center
            localization_result['core_center_xy'] = localization_result['cladding_center_xy'] # Default to cladding center.
            # Estimate core radius as 40% of cladding (typical for single-mode fiber)
            localization_result['core_radius_px'] = localization_result['cladding_radius_px'] * 0.4 # Default typical ratio.
            # Log core estimation
            logging.warning(f"Core detection failed, defaulting to 0.4 * cladding radius.")
    else: # If no core contours found by Otsu.
        # Log absence of core contours
        logging.warning("No core contours found using Otsu within cladding mask.")
        # Default core center to cladding center
        localization_result['core_center_xy'] = localization_result['cladding_center_xy'] # Default to cladding center.
        # Estimate core radius as 40% of cladding
        localization_result['core_radius_px'] = localization_result['cladding_radius_px'] * 0.4 # Default typical ratio.
        # Log core estimation
        logging.warning(f"Core detection defaulting to 0.4 * cladding radius.")

    # Return complete localization results
    return localization_result # Return all localization data.


# Define zone mask generation function based on IEC standards
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
    # Initialize dictionary to store zone masks
    masks: Dict[str, np.ndarray] = {} # Initialize dictionary for masks.
    # Extract image dimensions
    h, w = image_shape[:2] # Get image height and width.
    # Create coordinate grids for distance calculations
    Y, X = np.ogrid[:h, :w] # Create Y and X coordinate grids.

    # Get detected fiber parameters
    # Extract cladding center from localization data
    cladding_center = localization_data.get('cladding_center_xy') # Get cladding center.
    # Extract core center, defaulting to cladding center if not found
    core_center = localization_data.get('core_center_xy', cladding_center) # Default core center to cladding center.
    # Extract detected core radius (may be None or 0)
    core_radius_px_detected = localization_data.get('core_radius_px') # Get detected core radius (can be None or 0)
    
    # Extract detected cladding radius in pixels
    detected_cladding_radius_px = localization_data.get('cladding_radius_px') # Get cladding radius.
    # Extract ellipse parameters if cladding is elliptical
    cladding_ellipse_params = localization_data.get('cladding_ellipse_params') # Get cladding ellipse parameters.


    # Check if cladding center was successfully localized
    if cladding_center is None: # If cladding center not found.
        # Log error - cannot generate zones without cladding center
        logging.error("Cannot generate zone masks: Cladding center not localized.")
        # Return empty masks dictionary
        return masks # Return empty masks.

    # Store user-provided reference diameters
    reference_cladding_diameter_um = user_cladding_diameter_um 
    reference_core_diameter_um = user_core_diameter_um

    # Process each zone definition from configuration
    for zone_def in zone_definitions: # Iterate through each zone definition.
        # Extract zone name from definition
        name = zone_def["name"]
        # Initialize zone radius limits
        r_min_px: float = 0.0 
        r_max_px: float = 0.0 
        
        # Default zone center to cladding center
        current_zone_center = cladding_center 
        # Flag for elliptical zone shape
        is_elliptical_zone = False

        # Mode determination: Micron-based or Pixel-based
        # Prefer micron mode if all necessary info is present
        # Check if micron-based calculations are possible
        micron_mode_possible = um_per_px is not None and um_per_px > 0 and reference_cladding_diameter_um is not None
        
        # Use micron-based zone definitions if possible
        if micron_mode_possible:
            # Log use of micron mode for this zone
            logging.debug(f"Zone '{name}': Using micron mode for definitions.")
            # All calculations in microns first, then convert to pixels
            # Initialize micron radius limits
            r_min_um: Optional[float] = None
            r_max_um: Optional[float] = None

            # Handle Core zone definition
            if name == "Core" and reference_core_diameter_um is not None:
                # Calculate core radius from diameter
                core_radius_ref_um = reference_core_diameter_um / 2.0
                # Apply zone definition factors to core radius
                r_min_um = zone_def.get("r_min_factor", 0.0) * core_radius_ref_um
                r_max_um = zone_def.get("r_max_factor_core_relative", 1.0) * core_radius_ref_um
                # Use core center if detected, otherwise cladding center
                current_zone_center = core_center if core_center is not None else cladding_center
            # Handle Cladding zone definition
            elif name == "Cladding":
                # Calculate cladding radius from diameter
                cladding_radius_ref_um = reference_cladding_diameter_um / 2.0
                # Apply zone definition factor for outer boundary
                r_max_um = zone_def.get("r_max_factor_cladding_relative", 1.0) * cladding_radius_ref_um
                # r_min for cladding is core's r_max
                # Calculate inner boundary (should start after core)
                if reference_core_diameter_um is not None:
                    # Find core zone definition to get its outer boundary
                    core_def_temp = next((zd for zd in zone_definitions if zd["name"] == "Core"), None)
                    core_radius_ref_um_for_cladding_min = reference_core_diameter_um / 2.0
                    r_min_um_cladding_start = 0.0
                    # Get core's outer boundary
                    if core_def_temp:
                         r_min_um_cladding_start = core_def_temp.get("r_max_factor_core_relative",1.0) * core_radius_ref_um_for_cladding_min
                    
                    # Apply cladding's minimum factor
                    r_min_um_from_factor = zone_def.get("r_min_factor_cladding_relative", 0.0) * cladding_radius_ref_um
                    # Use maximum of factor-based and core-based minimum
                    r_min_um = max(r_min_um_from_factor, r_min_um_cladding_start)
                else: # No core reference, cladding r_min relative to its own diameter
                    # Log missing core reference
                    logging.warning(f"Zone '{name}': Missing reference core diameter for precise r_min_um. r_min relative to cladding.")
                    # Calculate minimum based on cladding only
                    r_min_um = zone_def.get("r_min_factor_cladding_relative", 0.0) * cladding_radius_ref_um
            else: # Other zones (Adhesive, Contact) relative to cladding outer diameter
                # Calculate boundaries relative to cladding
                cladding_outer_r_um = reference_cladding_diameter_um / 2.0
                r_min_um = zone_def.get("r_min_factor_cladding_relative", 1.0) * cladding_outer_r_um
                r_max_um = zone_def.get("r_max_factor_cladding_relative", 1.15) * cladding_outer_r_um
            
            # Convert micron values to pixels
            if r_min_um is not None and r_max_um is not None and um_per_px is not None and um_per_px > 0: # Ensure um_per_px is valid
                # Convert micron radii to pixel radii
                r_min_px = r_min_um / um_per_px
                r_max_px = r_max_um / um_per_px
            else:
                # Log conversion failure
                logging.error(f"Cannot define zone '{name}' in micron mode due to missing data (r_min/max_um or um_per_px). Falling back.")
                # Force fallback to pixel mode
                micron_mode_possible = False # Force fallback to pixel mode if critical data missing

        # Pixel mode (either primary choice or fallback from micron mode)
        # Use pixel-based zone definitions if micron mode not possible
        if not micron_mode_possible:
            # Log use of pixel mode for this zone
            logging.debug(f"Zone '{name}': Using pixel mode for definitions.")
            # Check if detected cladding radius is valid
            if detected_cladding_radius_px is not None and detected_cladding_radius_px > 0:
                # Handle Core zone in pixel mode
                if name == "Core":
                    # Core starts at center
                    r_min_px = 0.0
                    # Use detected core radius if valid, else estimate from cladding
                    # Determine core radius to use
                    core_r_px_to_use = core_radius_px_detected if core_radius_px_detected is not None and core_radius_px_detected > 0 else (detected_cladding_radius_px * 0.4)
                    # Apply zone definition factor
                    r_max_px = zone_def.get("r_max_factor_core_relative", 1.0) * core_r_px_to_use # Assume factor applies to actual/estimated core radius
                    # Use core center if available
                    current_zone_center = core_center if core_center is not None else cladding_center
                # Handle Cladding zone in pixel mode
                elif name == "Cladding":
                    # Determine core radius for cladding inner boundary
                    core_r_px_for_cladding_min = core_radius_px_detected if core_radius_px_detected is not None and core_radius_px_detected > 0 else (detected_cladding_radius_px * 0.4)
                    # Calculate minimum radius from factor
                    r_min_px = zone_def.get("r_min_factor_cladding_relative", 0.0) * detected_cladding_radius_px # Factor relative to cladding
                    # Ensure r_min for cladding starts after core
                    # Ensure cladding starts after core boundary
                    r_min_px = max(r_min_px, core_r_px_for_cladding_min) 
                    # Calculate maximum radius for cladding
                    r_max_px = zone_def.get("r_max_factor_cladding_relative", 1.0) * detected_cladding_radius_px
                else: # Adhesive, Contact relative to detected cladding radius
                    # Calculate boundaries relative to detected cladding
                    r_min_px = zone_def.get("r_min_factor_cladding_relative", 1.0) * detected_cladding_radius_px
                    r_max_px = zone_def.get("r_max_factor_cladding_relative", 1.15) * detected_cladding_radius_px
            else:
                # Log error if cladding radius is invalid
                logging.error(f"Cannot define zone '{name}' in pixel mode: detected_cladding_radius_px is missing or invalid.")
                # Skip this zone if no valid radius
                continue # Skip this zone

        # Create mask for the current zone
        # Initialize blank mask for current zone
        zone_mask_np = np.zeros((h, w), dtype=np.uint8)
        
        # Check if zone center is valid
        if current_zone_center is None: # Should be caught by cladding_center check, but safeguard
            # Log critical error for missing zone center
            logging.error(f"Critical: current_zone_center is None for zone '{name}'. Skipping.")
            # Skip to next zone
            continue
        # Extract zone center coordinates
        cx_zone, cy_zone = current_zone_center

        # Determine if ellipse should be used for this zone
        # Use ellipse if cladding was found as elliptical AND (it's not Core OR Core wasn't distinctly found/matches cladding shape)
        # Check if zone should be elliptical based on cladding shape
        use_ellipse_for_zone = (cladding_ellipse_params is not None) and \
                               (name != "Core" or \
                                (name == "Core" and (core_radius_px_detected is None or core_radius_px_detected <= 0)) or \
                                (name == "Core" and localization_data.get('core_center_xy') == localization_data.get('cladding_center_xy')))


        # Create elliptical zone mask if appropriate
        if use_ellipse_for_zone and cladding_ellipse_params is not None: # Ensure cladding_ellipse_params is not None
            # Indexing cladding_ellipse_params is correct, Pylance error is likely false positive
            # Extract ellipse center
            base_center_ell = (int(cladding_ellipse_params[0][0]), int(cladding_ellipse_params[0][1]))
            # Ensure axes are tuple of floats for cv2.ellipse
            # Extract and convert ellipse axes to float
            base_minor_axis = float(cladding_ellipse_params[1][0]) 
            base_major_axis = float(cladding_ellipse_params[1][1])
            base_angle = float(cladding_ellipse_params[2])

            # Calculate average radius from ellipse axes
            avg_cladding_ellipse_radius = (base_major_axis + base_minor_axis) / 4.0 # Using /4 for radius from two axes

            # Calculate scale factors for zone boundaries
            if avg_cladding_ellipse_radius > 1e-6: # Avoid division by zero or very small numbers
                # Ensure r_max_px is float for division
                assert isinstance(r_max_px, float), "r_max_px should be a float for division" #
                # Ensure avg_cladding_ellipse_radius is valid
                assert isinstance(avg_cladding_ellipse_radius, float) and avg_cladding_ellipse_radius != 0, \
                    "avg_cladding_ellipse_radius must be a non-zero float for division" #
                # Calculate scale factor for outer boundary
                scale_factor_max = r_max_px / avg_cladding_ellipse_radius #
                # Ensure r_min_px is float for division
                assert isinstance(r_min_px, float), "r_min_px should be a float for division" #
                # Calculate scale factor for inner boundary
                scale_factor_min = r_min_px / avg_cladding_ellipse_radius #
            else:
                # Default scale factors if radius too small
                scale_factor_max = 1.0
                scale_factor_min = 0.0
            
            # Axes for cv2.ellipse should be (major_axis/2, minor_axis/2) or (width, height)
            # The cv2.fitEllipse returns (minorAxis, majorAxis) which are full lengths.
            # cv2.ellipse expects (majorAxisRadius, minorAxisRadius) or rather (axes_width/2, axes_height/2)
            # Here, base_minor_axis and base_major_axis are full lengths.
            # For cv2.ellipse, the axes tuple is (width, height) which are full lengths.
            # Calculate scaled ellipse axes for outer boundary
            outer_ellipse_axes_tuple = (int(base_minor_axis * scale_factor_max), int(base_major_axis * scale_factor_max))
            # Calculate scaled ellipse axes for inner boundary
            inner_ellipse_axes_tuple = (int(base_minor_axis * scale_factor_min), int(base_major_axis * scale_factor_min))
            
            # Draw outer ellipse boundary if valid
            if r_max_px > 0 and outer_ellipse_axes_tuple[0] > 0 and outer_ellipse_axes_tuple[1] > 0:
                 # Color for cv2.ellipse
                 # Draw filled outer ellipse
                 cv2.ellipse(zone_mask_np, (base_center_ell, outer_ellipse_axes_tuple, base_angle), (255,), -1)
            
            # Subtract inner ellipse to create annular region if needed
            if r_min_px > 0 and inner_ellipse_axes_tuple[0] > 0 and inner_ellipse_axes_tuple[1] > 0:
                 # Create temporary mask for inner ellipse
                 temp_inner_mask = np.zeros_like(zone_mask_np)
                 # Color for cv2.ellipse
                 # Draw filled inner ellipse on temporary mask
                 cv2.ellipse(temp_inner_mask, (base_center_ell, inner_ellipse_axes_tuple, base_angle), (255,), -1)
                 # Subtract inner ellipse from outer to create annulus
                 zone_mask_np = cv2.subtract(zone_mask_np, temp_inner_mask)
            # Mark zone as elliptical
            is_elliptical_zone = True

        else: # Circular zones.
            # Calculate squared distance map from zone center
            dist_sq_map = (X - cx_zone)**2 + (Y - cy_zone)**2 
            # Create annular mask using distance map and radius limits
            zone_mask_np = ((dist_sq_map >= r_min_px**2) & (dist_sq_map < r_max_px**2)).astype(np.uint8) * 255

        # Store generated mask in dictionary with zone name as key
        masks[name] = zone_mask_np 
        # Log details of generated zone mask
        logging.debug(f"Generated mask for zone '{name}': Center={current_zone_center}, Rmin_px={r_min_px:.1f}, Rmax_px={r_max_px:.1f}, Elliptical={is_elliptical_zone}")

    # Return dictionary of all generated zone masks
    return masks

# --- DETAILED ALGORITHM IMPLEMENTATIONS ---
# These will overwrite stubs if stubs were defined (i.e., if config_loader import failed)



# Define complete LEI scratch detection algorithm based on research paper
def _lei_scratch_detection(enhanced_image: np.ndarray, kernel_lengths: List[int], angle_step: int = 15) -> np.ndarray:
    """
    Complete LEI implementation following paper Section 3.2 exactly.
    Uses dual-branch linear detector with proper response calculation.
    Optimized using vectorized operations for better performance.
    """
    # Get image dimensions
    h, w = enhanced_image.shape[:2]
    # Initialize maximum response map
    max_response_map = np.zeros((h, w), dtype=np.float32)
    
    # Paper specifies 12 orientations (0° to 165° in 15° steps)
    # Define angles for scratch search
    angles_deg = np.arange(0, 180, angle_step)
    
    # Pre-compute coordinate grids for vectorization
    Y, X = np.ogrid[:h, :w]
    
    # Iterate through different kernel lengths for multi-scale detection
    for length in kernel_lengths:
        # Paper specifies branch offset of 2 pixels
        # Define offset between detector branches
        branch_offset = 2
        # Calculate half length of kernel
        half_length = length // 2
        
        # Iterate through all search angles
        for angle_deg in angles_deg:
            # Convert angle to radians for trigonometric calculations
            angle_rad = np.deg2rad(angle_deg)
            # Pre-calculate sine and cosine
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            # Create kernels for red and gray branches
            # Initialize kernels for center (red) and parallel (gray) branches
            red_kernel = np.zeros((length, length), dtype=np.float32)
            gray_kernel = np.zeros((length, length), dtype=np.float32)
            
            # Fill kernels based on paper's dual-branch design
            # Populate kernel elements along line for each branch
            for t in range(-half_length, half_length + 1):
                # Red branch (center line)
                # Calculate coordinates for center branch
                cx = half_length + int(round(t * cos_a))
                cy = half_length + int(round(t * sin_a))
                # Set kernel value if within bounds
                if 0 <= cx < length and 0 <= cy < length:
                    red_kernel[cy, cx] = 1.0
                
                # Gray branches (parallel lines)
                # Iterate for both parallel branches
                for side in [-1, 1]:
                    # Calculate coordinates for parallel branches
                    gx = half_length + int(round(t * cos_a + side * branch_offset * (-sin_a)))
                    gy = half_length + int(round(t * sin_a + side * branch_offset * cos_a))
                    # Set kernel value if within bounds
                    if 0 <= gx < length and 0 <= gy < length:
                        gray_kernel[gy, gx] = 1.0
            
            # Normalize kernels
            # Calculate sum of kernel elements for normalization
            red_sum = np.sum(red_kernel)
            gray_sum = np.sum(gray_kernel)
            
            # Normalize center branch kernel
            if red_sum > 0:
                red_kernel /= red_sum
            # Normalize parallel branch kernel
            if gray_sum > 0:
                gray_kernel /= gray_sum
            
            # Apply filters using convolution
            # Apply filters only if both kernels are valid
            if red_sum > 0 and gray_sum > 0:
                # Convolve image with center branch kernel
                red_response = cv2.filter2D(enhanced_image.astype(np.float32), cv2.CV_32F, red_kernel)
                # Convolve image with parallel branch kernel
                gray_response = cv2.filter2D(enhanced_image.astype(np.float32), cv2.CV_32F, gray_kernel)
                
                # Paper's formula: s_θ(x,y) = 2*f_r - f_g
                # Calculate scratch strength according to paper's formula
                response = np.maximum(0, 2 * red_response - gray_response)
                # Update maximum response map
                max_response_map = np.maximum(max_response_map, response)
    
    # Apply Gaussian smoothing as per paper
    # Apply Gaussian blur to smooth response map
    max_response_map = cv2.GaussianBlur(max_response_map, (3, 3), 0.5)
    
    # Normalize to 0-1 range
    # Normalize final response map to range [0, 1]
    if np.max(max_response_map) > 0:
        max_response_map = max_response_map / np.max(max_response_map)
    
    # Return normalized maximum response map
    return max_response_map




# Define Gabor filter-based defect detection function
def _gabor_defect_detection(image: np.ndarray) -> np.ndarray:
    """
    Uses Gabor filters for texture-based defect detection.
    """
    # Initialize list to store Gabor filters
    gabor_filters_list = [] # Renamed from filters to gabor_filters_list
    # Gabor filter kernel size
    ksize = 31 
    # Standard deviation of Gaussian envelope
    sigma = 4.0 
    # Wavelength of sinusoidal factor
    lambd = 10.0 
    # Spatial aspect ratio (gamma)
    gamma_gabor = 0.5 # Renamed gamma to gamma_gabor to avoid conflict with other gamma variables
    # Phase offset
    psi = 0 
    
    # Create Gabor filters for multiple orientations
    for theta in np.arange(0, np.pi, np.pi / 8): 
        # Generate Gabor kernel for current orientation
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma_gabor, psi, ktype=cv2.CV_32F)
        # Add kernel to list
        gabor_filters_list.append(kern)
    
    # Initialize list to store filter responses
    responses = []
    # Apply each Gabor filter to the image
    for kern in gabor_filters_list:
        # Apply filter and get response
        filtered = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kern) 
        # Store absolute value of response
        responses.append(np.abs(filtered)) 
    
    # Take maximum response across all orientations
    gabor_response = np.max(np.array(responses), axis=0) 
    # Normalize response to 8-bit range
    gabor_response_norm = cv2.normalize(gabor_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #
    # Apply Otsu's threshold to create binary mask
    _, defect_mask = cv2.threshold(gabor_response_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Return binary defect mask
    return defect_mask

# Define wavelet transform-based defect detection function
def _wavelet_defect_detection(image: np.ndarray) -> np.ndarray:
    """
    Uses wavelet transform for multi-resolution defect detection.
    """
    # Apply 2D Discrete Wavelet Transform with Daubechies 4 wavelet
    coeffs = pywt.dwt2(image.astype(np.float32), 'db4') 
    # Extract approximation and detail coefficients
    cA, (cH, cV, cD) = coeffs 
    # Calculate magnitude of detail coefficients
    details_magnitude = np.sqrt(cH**2 + cV**2 + cD**2)
    # Resize detail magnitude map to original image size
    details_resized = cv2.resize(details_magnitude, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Normalize detail magnitude to 8-bit range
    details_norm = cv2.normalize(details_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #
    # Apply Otsu's threshold to create binary mask
    _, defect_mask = cv2.threshold(details_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Return binary defect mask
    return defect_mask


# Define detailed DO2MR region defect detection algorithm based on research paper
def _do2mr_detection(masked_zone_image: np.ndarray, kernel_size: int = 5, gamma: float = 1.5) -> np.ndarray:
    """
    Enhanced DO2MR implementation following the research paper exactly.
    Paper Section 3.1 specifies the exact methodology:
    1. Maximum filtering (dilation)
    2. Minimum filtering (erosion)  
    3. Residual generation
    4. Sigma-based thresholding
    5. Morphological opening for noise removal
    """
    # Ensure input is uint8 for morphological operations
    # Ensure input image is 8-bit unsigned integer type
    if masked_zone_image.dtype != np.uint8:
        # Normalize and convert image if not already uint8
        normalized = cv2.normalize(masked_zone_image, None, 0, 255, cv2.NORM_MINMAX)
        masked_zone_image = normalized.astype(np.uint8)
    
    # Step 1 & 2: Min-Max Filtering as specified in paper
    # Paper uses square structuring element
    # Create rectangular structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Maximum filtering: Find brightest pixel in neighborhood (morphological dilation)
    # Apply dilation (maximum filter)
    I_max = cv2.dilate(masked_zone_image, kernel, iterations=1)
    
    # Minimum filtering: Find darkest pixel in neighborhood (morphological erosion)
    # Apply erosion (minimum filter)
    I_min = cv2.erode(masked_zone_image, kernel, iterations=1)
    
    # Step 3: Generate residual map - highlights areas with high local contrast
    # Paper equation: I_r(x,y) = I_max(x,y) - I_min(x,y)
    # Calculate residual image by subtracting minimum from maximum filtered images
    I_residual = cv2.subtract(I_max, I_min)
    
    # Apply median filter as specified in paper (3x3)
    # Apply median blur to reduce noise in residual image
    I_residual_filtered = cv2.medianBlur(I_residual, 3)
    
    # Step 4: Sigma-based thresholding
    # Calculate statistics only on non-zero pixels (active zone area)
    # Create mask of active (non-black) pixels in the zone
    active_pixels_mask = masked_zone_image > 0
    # Return empty mask if zone is empty
    if np.sum(active_pixels_mask) == 0:
        return np.zeros_like(masked_zone_image, dtype=np.uint8)
    
    # Paper specifies using mean (μ) and standard deviation (σ) of residual in the zone
    # Extract residual values from active zone for statistical analysis
    zone_residual_values = I_residual_filtered[active_pixels_mask].astype(np.float32)
    # Calculate mean of residual values
    mu = np.mean(zone_residual_values)
    # Calculate standard deviation of residual values
    sigma = np.std(zone_residual_values)
    
    # Paper's threshold equation: T = μ + γ*σ 
    # Calculate adaptive threshold based on statistics and sensitivity parameter gamma
    threshold_value = mu + gamma * sigma
    
    # Apply threshold to create binary defect mask
    # Apply calculated threshold to create binary defect mask
    _, defect_binary = cv2.threshold(I_residual_filtered, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Step 5: Morphological opening to remove small noise (paper specifies 3x3 kernel)
    # Create elliptical kernel for noise removal
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPse, (3, 3))
    # Apply morphological opening to remove small noise regions
    defect_binary_cleaned = cv2.morphologyEx(defect_binary, cv2.MORPH_OPEN, kernel_open)
    
    # Filter by minimum area as per paper (5 pixels minimum)
    # Find connected components in the cleaned binary mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_binary_cleaned, connectivity=8)
    # Initialize final mask
    final_mask = np.zeros_like(defect_binary_cleaned)
    
    # Define minimum defect area in pixels from paper
    min_defect_area_px = 5  # Paper's minimum defect area
    # Filter components by area
    for i in range(1, num_labels):
        # Get area of current component
        area = stats[i, cv2.CC_STAT_AREA]
        # Keep component if its area is above the minimum threshold
        if area >= min_defect_area_px:
            final_mask[labels == i] = 255
    
    # Return final, cleaned defect mask
    return final_mask

# Define multi-scale DO2MR detection function for improved accuracy
def _multiscale_do2mr_detection(image: np.ndarray, scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5]) -> np.ndarray:
    """
    Multi-scale DO2MR detection as suggested in the paper for improved accuracy.
    Combines results from multiple scales to reduce false positives.
    """
    # Get image dimensions
    h, w = image.shape[:2]
    # Initialize combined result map
    combined_result = np.zeros((h, w), dtype=np.float32)
    
    # Iterate through different scales
    for scale in scales:
        # Resize image
        # If scale is not 1, resize the image
        if scale != 1.0:
            # Calculate scaled dimensions
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            # Resize image to current scale
            scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        else:
            # Use original image if scale is 1
            scaled_image = image.copy()
        
        # Apply DO2MR at this scale
        # Adjust kernel size based on scale
        # Adjust kernel size based on scale factor
        kernel_size = max(3, int(5 * scale))
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply DO2MR with adjusted kernel size
        result = _do2mr_detection(scaled_image, kernel_size=kernel_size)
        
        # Resize result back to original size
        # Resize detection mask back to original size
        if scale != 1.0:
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Weight by scale (smaller scales get higher weight for small defects)
        # Weight results by scale factor (smaller scales are more sensitive)
        weight = 1.0 / scale if scale > 0 else 1.0
        # Add weighted result to combined map
        combined_result += result.astype(np.float32) * weight
    
    # Normalize and threshold
    # Normalize combined result map
    combined_result = combined_result / len(scales)
    # Threshold combined map to get final binary mask
    _, final_result = cv2.threshold(combined_result.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    
    # Return final multi-scale detection mask
    return final_result


# Define detailed multi-scale defect detection function
def _multiscale_defect_detection(image: np.ndarray, scales: List[float] = [0.5, 1.0, 1.5, 2.0]) -> np.ndarray:
    """
    Performs multi-scale defect detection using detailed _do2mr_detection.
    """
    # Get image dimensions
    h, w = image.shape[:2]
    # Initialize combined map for accumulating results
    combined_map_float = np.zeros((h, w), dtype=np.float32) 
    
    # Iterate through specified scales
    for scale_ms in scales: # Renamed scale to scale_ms
        # Start with a copy of the original image
        scaled_image = image.copy() 
        # Resize image if scale is not 1.0
        if scale_ms != 1.0:
            # Skip invalid scale values
            if scale_ms <= 0: continue 
            # Calculate scaled dimensions
            scaled_h, scaled_w = int(h * scale_ms), int(w * scale_ms)
            # Skip if scaled dimensions are invalid
            if scaled_h <=0 or scaled_w <=0: continue 
            # Resize image to current scale
            scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        
        # Adjust DO2MR kernel size based on scale
        do2mr_kernel_size_at_scale = max(3, int(5 * scale_ms)) 
        # Ensure kernel size is odd
        if do2mr_kernel_size_at_scale % 2 == 0: do2mr_kernel_size_at_scale +=1
        
        # Apply DO2MR detection at the current scale
        do2mr_result_at_scale = _do2mr_detection(scaled_image, kernel_size=do2mr_kernel_size_at_scale) 
        
        # Resize detection mask back to original size
        do2mr_result_resized = do2mr_result_at_scale 
        if scale_ms != 1.0:
            do2mr_result_resized = cv2.resize(do2mr_result_at_scale, (w, h), interpolation=cv2.INTER_NEAREST) 
        
        # Weight results based on scale (smaller scales are more sensitive)
        weight = 1.0 / scale_ms if scale_ms > 1 else scale_ms if scale_ms > 0 else 1.0
        # Add weighted result to combined map
        combined_map_float += do2mr_result_resized.astype(np.float32) * weight
    
    # Initialize 8-bit map for final result
    combined_map_uint8 = np.zeros((h,w), dtype=np.uint8)
    # Normalize combined map if it contains any detections
    if np.any(combined_map_float): 
        combined_map_uint8 = cv2.normalize(combined_map_float, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply Otsu's threshold to create final binary mask
    _, final_binary_mask = cv2.threshold(combined_map_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Return final multi-scale detection mask
    return final_binary_mask


# Define advanced scratch detection function combining multiple techniques
def _advanced_scratch_detection(image: np.ndarray) -> np.ndarray:
    """
    Advanced scratch detection using multiple techniques.
    """
    # Use input image for processing
    processed_image = image
    # Normalize to 8-bit if not already
    if image.dtype != np.uint8: 
        processed_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #

    # Get image dimensions
    h, w = processed_image.shape[:2]
    # Initialize combined scratch map
    scratch_map_combined = np.zeros((h, w), dtype=np.uint8) 
    
    # Calculate first-order Sobel gradients
    sobelx = cv2.Sobel(processed_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(processed_image, cv2.CV_64F, 0, 1, ksize=5)
    # Calculate second-order derivatives for Hessian matrix
    sobelxx = cv2.Sobel(sobelx, cv2.CV_64F, 1, 0, ksize=5) 
    sobelyy = cv2.Sobel(sobely, cv2.CV_64F, 0, 1, ksize=5) 
    sobelxy = cv2.Sobel(sobelx, cv2.CV_64F, 0, 1, ksize=5) 

    # Initialize ridge response map
    ridge_response = np.zeros_like(processed_image, dtype=np.float64)
    # Iterate through each pixel to analyze Hessian eigenvalues
    for r_idx in range(h): # Renamed y/r to r_idx
        for c_idx in range(w): # Renamed x/c to c_idx
            # Construct Hessian matrix at current pixel
            hessian_matrix = np.array([[sobelxx[r_idx,c_idx], sobelxy[r_idx,c_idx]], 
                                       [sobelxy[r_idx,c_idx], sobelyy[r_idx,c_idx]]])
            try:
                # Calculate eigenvalues of Hessian
                eigenvalues, _ = np.linalg.eig(hessian_matrix) 
                # Check for large negative eigenvalue (indicates a ridge)
                if eigenvalues.min() < -50: 
                    # Store magnitude of ridge response
                    ridge_response[r_idx, c_idx] = np.abs(eigenvalues.min())
            except np.linalg.LinAlgError:
                # Ignore linear algebra errors (e.g., singular matrix)
                pass 

    # Process ridge response if any was detected
    if np.any(ridge_response):
        # Normalize ridge response to 8-bit range
        ridge_response_norm = cv2.normalize(ridge_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #
        # Threshold ridge response to get binary mask
        _, ridge_mask = cv2.threshold(ridge_response_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Combine with main scratch map
        scratch_map_combined = cv2.bitwise_or(scratch_map_combined, ridge_mask)
    
    # Apply directional black-hat transforms for scratch detection
    kernel_bh_rect_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)) 
    kernel_bh_rect_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)) 
    # Apply vertical black-hat transform
    blackhat_v = cv2.morphologyEx(processed_image, cv2.MORPH_BLACKHAT, kernel_bh_rect_vertical)
    # Apply horizontal black-hat transform
    blackhat_h = cv2.morphologyEx(processed_image, cv2.MORPH_BLACKHAT, kernel_bh_rect_horizontal)
    # Combine vertical and horizontal responses
    blackhat_combined = np.maximum(blackhat_v, blackhat_h)

    # Process black-hat result if any was detected
    if np.any(blackhat_combined):
        # Threshold combined black-hat response
        _, bh_thresh = cv2.threshold(blackhat_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Combine with main scratch map
        scratch_map_combined = cv2.bitwise_or(scratch_map_combined, bh_thresh)
    
    # Use Canny edge detection followed by Hough transform for line detection
    edges = cv2.Canny(processed_image, 50, 150, apertureSize=3) 
    # Detect line segments using probabilistic Hough transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=7) 
    
    # Draw detected lines if any were found
    if lines is not None:
        # Initialize mask for Hough lines
        line_mask = np.zeros_like(processed_image, dtype=np.uint8)
        # Draw each detected line segment
        for line_segment in lines: # Renamed line to line_segment
            # Extract line endpoints
            x1, y1, x2, y2 = line_segment[0]
            # Corrected color for cv2.line
            # Draw line on mask
            cv2.line(line_mask, (x1, y1), (x2, y2), (255,), 1) 
        # Combine with main scratch map
        scratch_map_combined = cv2.bitwise_or(scratch_map_combined, line_mask)
    
    # Clean up final scratch map using morphological operations
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Apply opening to remove small noise
    scratch_map_combined = cv2.morphologyEx(scratch_map_combined, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    # Apply closing to connect broken line segments
    scratch_map_combined = cv2.morphologyEx(scratch_map_combined, cv2.MORPH_CLOSE, kernel_clean, iterations=1)
    
    # Return final combined scratch mask
    return scratch_map_combined


# Define main defect detection engine with multi-algorithm fusion
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
    # Skip detection if zone mask is empty
    if np.sum(zone_mask) == 0:
        # Log skipping empty zone
        logging.debug(f"Defect detection skipped for empty zone mask in zone '{zone_name}'.")
        # Return empty defect mask and confidence map
        return np.zeros_like(processed_image, dtype=np.uint8), np.zeros_like(processed_image, dtype=np.float32)

    # Get image dimensions
    h, w = processed_image.shape[:2]
    # Initialize confidence map for accumulating algorithm votes
    confidence_map = np.zeros((h, w), dtype=np.float32)
    # Isolate current zone from the processed image
    working_image_input = cv2.bitwise_and(processed_image, processed_image, mask=zone_mask)

    # Ensure working_image is uint8 for certain operations
    # Check if image is already 8-bit unsigned integer
    if working_image_input.dtype != np.uint8:
        # Log conversion to uint8 if necessary
        logging.debug(f"Original working_image for zone '{zone_name}' is {working_image_input.dtype}, will normalize to uint8 for some steps.")
        # Keep a float version if needed for some algos, but ensure uint8 for others
        # Normalize and convert to uint8
        working_image_uint8 = cv2.normalize(working_image_input, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        # Use a copy to avoid modifying the original processed image slice
        working_image_uint8 = working_image_input.copy() # Use copy to avoid modifying original processed_image slice

    # Set default processing image to the uint8 version
    working_image_for_processing = working_image_uint8 # Default to uint8 version

    # Apply zone-specific preprocessing for Core
    if zone_name == "Core":
        # Log application of Core-specific preprocessing
        logging.debug(f"Applying {zone_name}-specific preprocessing.")
        # Median blur expects uint8
        # Apply median blur for noise reduction
        blurred_core = cv2.medianBlur(working_image_uint8, 3)
        # Apply CLAHE if there are non-zero pixels in the zone
        if np.any(blurred_core[zone_mask > 0]): 
            # Create CLAHE object for Core zone
            clahe_core = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            # CLAHE expects uint8
            # Apply CLAHE for local contrast enhancement
            enhanced_region = clahe_core.apply(blurred_core) 
            # Re-mask the enhanced region
            working_image_for_processing = cv2.bitwise_and(enhanced_region, enhanced_region, mask=zone_mask)
        else:
            # Use blurred image if CLAHE had no effect
            working_image_for_processing = blurred_core # Use blurred if CLAHE had no effect or region was blank
    # Apply zone-specific preprocessing for Cladding
    elif zone_name == "Cladding":
        # Log application of Cladding-specific preprocessing
        logging.debug(f"Applying {zone_name}-specific preprocessing.")
        # Bilateral filter works on uint8 or float32. If input was float, could use working_image_input
        # For consistency with uint8 path:
        # Apply bilateral filter for edge-preserving smoothing
        working_image_for_processing = cv2.bilateralFilter(working_image_uint8, d=5, sigmaColor=50, sigmaSpace=50)
        # Re-mask the filtered region
        working_image_for_processing = cv2.bitwise_and(working_image_for_processing, working_image_for_processing, mask=zone_mask)
    
    # Log details of the image being used for defect detection
    logging.debug(f"Proceeding with defect detection for zone: '{zone_name}' using specifically preprocessed image of type {working_image_for_processing.dtype}.")

    # Get defect detection configuration for current profile
    detection_cfg = profile_config.get("defect_detection", {})
    # Get list of region-based detection algorithms to run
    region_algos = detection_cfg.get("region_algorithms", [])
    # Get list of linear (scratch) detection algorithms to run
    linear_algos = detection_cfg.get("linear_algorithms", [])
    # Get list of optional advanced detection algorithms
    optional_algos = detection_cfg.get("optional_algorithms", [])
    # Get weights for each algorithm for confidence map fusion
    algo_weights = detection_cfg.get("algorithm_weights", {})

    # Ensure working_image_for_processing is used by subsequent algorithms
    # Some algorithms might prefer float input, others uint8. Adjust as necessary.
    # For now, most stubs and detailed implementations expect uint8 or handle conversion.

    # Apply DO2MR algorithm if enabled for region defects
    if "do2mr" in region_algos:
        # Use zone-specific gamma values as per paper
        # Get default DO2MR gamma value
        current_do2mr_gamma = global_algo_params.get("do2mr_gamma_default", 1.5)
        # Use Core-specific gamma if applicable
        if zone_name == "Core":
            current_do2mr_gamma = global_algo_params.get("do2mr_gamma_core", 1.2)
        # Use Cladding-specific gamma if applicable
        elif zone_name == "Cladding":
            current_do2mr_gamma = global_algo_params.get("do2mr_gamma_cladding", 1.5)
        # Use Adhesive-specific gamma if applicable
        elif zone_name == "Adhesive":
            current_do2mr_gamma = global_algo_params.get("do2mr_gamma_adhesive", 2.0)
            
        # Use multi-scale DO2MR for better accuracy
        # Apply multi-scale DO2MR if enabled
        if "multiscale" in region_algos:
            # Apply multi-scale DO2MR detection
            multiscale_result = _multiscale_do2mr_detection(working_image_for_processing)
            # Add weighted result to confidence map
            confidence_map[multiscale_result > 0] += algo_weights.get("multiscale_do2mr", 0.9)
        else:
            # Apply single-scale DO2MR detection
            do2mr_result = _do2mr_detection(working_image_for_processing, kernel_size=5, gamma=current_do2mr_gamma)
            # Add weighted result to confidence map
            confidence_map[do2mr_result > 0] += algo_weights.get("do2mr", 0.8)
        # Log DO2MR application with gamma value
        logging.debug(f"Applied DO2MR with gamma={current_do2mr_gamma} for zone '{zone_name}'")

    # Apply Morphological Gradient algorithm if enabled
    if "morph_gradient" in region_algos:
        # Get kernel size from global parameters
        kernel_size_list_mg_dd = global_algo_params.get("morph_gradient_kernel_size", [5,5]) # Renamed var
        # Create elliptical structuring element
        kernel_mg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(kernel_size_list_mg_dd))
        # MORPH_GRADIENT expects single channel uint8, float32, or float64
        # Calculate morphological gradient to highlight edges
        morph_gradient_img = cv2.morphologyEx(working_image_for_processing, cv2.MORPH_GRADIENT, kernel_mg)
        # Apply Otsu's threshold to gradient image
        _, thresh_mg = cv2.threshold(morph_gradient_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Add weighted result to confidence map
        confidence_map[thresh_mg > 0] += algo_weights.get("morph_gradient", 0.4)
        # Log application of Morphological Gradient
        logging.debug("Applied Morphological Gradient for region defects.")

    # Apply Black-Hat Transform if enabled
    if "black_hat" in region_algos:
        # Get kernel size from global parameters
        kernel_size_list_bh_dd = global_algo_params.get("black_hat_kernel_size", [11,11]) # Renamed var
        # Create elliptical structuring element
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(kernel_size_list_bh_dd))
        # Apply black-hat transform to find dark defects
        black_hat_img = cv2.morphologyEx(working_image_for_processing, cv2.MORPH_BLACKHAT, kernel_bh)
        # Apply Otsu's threshold to black-hat image
        _, thresh_bh = cv2.threshold(black_hat_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Add weighted result to confidence map
        confidence_map[thresh_bh > 0] += algo_weights.get("black_hat", 0.6)
        # Log application of Black-Hat Transform
        logging.debug("Applied Black-Hat Transform for region defects.")
    
    # Apply Gabor filters if enabled
    if "gabor" in region_algos:
        # Apply Gabor filter-based detection
        gabor_result = _gabor_defect_detection(working_image_for_processing) # Expects uint8 or float, handles astype(np.float32)
        # Add weighted result to confidence map
        confidence_map[gabor_result > 0] += algo_weights.get("gabor", 0.4)
        # Log application of Gabor filters
        logging.debug("Applied Gabor filters for region defects.")
    
    # Apply multi-scale detection if enabled
    if "multiscale" in region_algos:
        # Get scale factors from global parameters
        scales_ms_dd = global_algo_params.get("multiscale_factors", [0.5, 1.0, 1.5, 2.0]) # Renamed var
        # Apply multi-scale detection
        multiscale_result = _multiscale_defect_detection(working_image_for_processing, scales_ms_dd) # _multiscale_defect_detection calls _do2mr_detection
        # Add weighted result to confidence map
        confidence_map[multiscale_result > 0] += algo_weights.get("multiscale", 0.6)
        # Log application of multi-scale detection
        logging.debug("Applied multi-scale detection for region defects.")

    # Apply Local Binary Pattern (LBP) analysis if enabled
    if "lbp" in region_algos:
            # Import LBP function from scikit-image
            from skimage.feature import local_binary_pattern
            # Define LBP parameters
            radius = 1
            n_points = 8 * radius
            # Apply LBP with uniform patterns
            lbp = local_binary_pattern(working_image_for_processing, n_points, radius, method='uniform')
            # Normalize LBP image to 8-bit range
            lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Threshold LBP image to get defect mask
            _, lbp_mask = cv2.threshold(lbp_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Add weighted result to confidence map
            confidence_map[lbp_mask > 0] += algo_weights.get("lbp", 0.3)
            # Log application of LBP analysis
            logging.debug("Applied LBP texture analysis for defects.")
    
    # Apply advanced LEI scratch detection if enabled
    if "lei_advanced" in linear_algos:
            # Step 1: Image Enhancement using histogram equalization (Paper Section 3.2)
            # Apply histogram equalization for scratch enhancement
            enhanced_for_lei = cv2.equalizeHist(working_image_for_processing)
            # Log LEI enhancement step
            logging.debug("Applied histogram equalization for LEI scratch detection")
            
            # Step 2: Scratch Searching with linear detectors at multiple orientations
            # Get LEI kernel lengths from global parameters
            lei_kernel_lengths = global_algo_params.get("lei_kernel_lengths", [11, 17, 23])
            # Get LEI angle step from global parameters
            angle_step_deg = global_algo_params.get("lei_angle_step_deg", 15)
            
            # Create response maps for each orientation
            # Initialize maximum response map
            max_response_map = np.zeros_like(enhanced_for_lei, dtype=np.float32)
            
            # Iterate through kernel lengths for multi-scale scratch detection
            for kernel_length in lei_kernel_lengths:
                # Iterate through search angles
                for angle_deg in range(0, 180, angle_step_deg):
                    # Create linear kernel for current orientation
                    # Convert angle to radians
                    angle_rad = np.radians(angle_deg)
                    
                    # Create a line kernel
                    # Initialize empty kernel
                    kernel = np.zeros((kernel_length, kernel_length), dtype=np.float32)
                    # Get kernel center
                    center = kernel_length // 2
                    
                    # Draw line through center at specified angle
                    # Populate kernel elements along line
                    for i in range(kernel_length):
                        # Calculate coordinates on line
                        x = int(center + (i - center) * np.cos(angle_rad))
                        y = int(center + (i - center) * np.sin(angle_rad))
                        # Set kernel value if within bounds
                        if 0 <= x < kernel_length and 0 <= y < kernel_length:
                            kernel[y, x] = 1.0
                    
                    # Normalize kernel
                    # Calculate sum for normalization
                    kernel_sum = np.sum(kernel)
                    # Normalize kernel if sum is positive
                    if kernel_sum > 0:
                        kernel /= kernel_sum
                    
                    # Apply filter to get response for this orientation
                    # Apply linear filter to get scratch response
                    response = cv2.filter2D(enhanced_for_lei.astype(np.float32), cv2.CV_32F, kernel)
                    # Update maximum response map
                    max_response_map = np.maximum(max_response_map, response)
            
            # Step 3: Scratch Segmentation - threshold the response map
            # Normalize response map to 0-255 range for thresholding
            # Normalize response map for thresholding
            max_response_uint8 = cv2.normalize(max_response_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Apply Otsu's threshold to get binary scratch mask
            _, scratch_binary = cv2.threshold(max_response_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Step 4: Result Synthesis is handled by the confidence map
            # Clean up the result with morphological operations
            # Create kernel for cleaning up scratch mask
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # Horizontal kernel for scratches
            # Apply morphological closing to connect scratch segments
            scratch_binary_cleaned = cv2.morphologyEx(scratch_binary, cv2.MORPH_CLOSE, kernel_clean)
            
            # Add weighted LEI result to confidence map
            confidence_map[scratch_binary_cleaned > 0] += algo_weights.get("lei_advanced", 0.8)
            # Log LEI completion
            logging.debug("Completed LEI scratch detection")
    
    # Apply advanced scratch detection if enabled
    if "advanced_scratch" in linear_algos:
        # _advanced_scratch_detection handles internal normalization if not uint8
        # Apply advanced scratch detection algorithm
        advanced_scratch_result = _advanced_scratch_detection(working_image_for_processing)
        # Add weighted result to confidence map
        confidence_map[advanced_scratch_result > 0] += algo_weights.get("advanced_scratch", 0.7)
        # Log application of advanced scratch detection
        logging.debug("Applied advanced scratch detection.")

    # Apply skeletonization for linear defect detection if enabled
    if "skeletonization" in linear_algos:
        # Use preprocessed zone image for Canny edge detection
        img_for_canny_skel = working_image_for_processing # Use the preprocessed image for the zone
        # Ensure image is uint8 for Canny
        if img_for_canny_skel.dtype != np.uint8: # Canny prefers uint8
            img_for_canny_skel = cv2.normalize(img_for_canny_skel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #

        # Apply Canny edge detection
        edges_skel_dd = cv2.Canny(img_for_canny_skel, 50, 150, apertureSize=global_algo_params.get("sobel_scharr_ksize",3)) # Renamed var
        try:
            # Apply thinning to skeletonize edges
            thinned_edges = cv2.ximgproc.thinning(edges_skel_dd, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            # Get dilation kernel size from parameters
            dilation_kernel_size_list_skel_dd = global_algo_params.get("skeletonization_dilation_kernel_size",[3,3]) # Renamed var
            # Create dilation kernel
            dilation_kernel_skel_dd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(dilation_kernel_size_list_skel_dd)) # Renamed var
            # Dilate thinned edges to make them more visible
            thinned_edges_dilated = cv2.dilate(thinned_edges, dilation_kernel_skel_dd, iterations=1)
            # Add weighted result to confidence map
            confidence_map[thinned_edges_dilated > 0] += algo_weights.get("skeletonization", 0.3)
            # Log application of skeletonization
            logging.debug("Applied Canny + Skeletonization for linear defects.")
        except AttributeError:
            # Log warning if skeletonization is not available (requires opencv-contrib-python)
            logging.warning("cv2.ximgproc.thinning not available (opencv-contrib-python needed). Skipping skeletonization.")
        except cv2.error as e_cv_error: # Renamed e to e_cv_error
            # Log OpenCV error during skeletonization
            logging.warning(f"OpenCV error during skeletonization (thinning): {e_cv_error}. Skipping.")

    # Apply wavelet transform for defect detection if enabled
    if "wavelet" in optional_algos:
        # Import pywt library for wavelet transform
        import pywt
        # Apply 2D DWT with Daubechies 4 wavelet
        coeffs = pywt.dwt2(working_image_for_processing.astype(np.float32), 'db4')
        # Extract approximation and detail coefficients
        cA, (cH, cV, cD) = coeffs
        # Calculate magnitude of detail coefficients
        details_magnitude = np.sqrt(cH**2 + cV**2 + cD**2)
        # Resize detail magnitude map to original size
        details_resized = cv2.resize(details_magnitude, working_image_for_processing.shape[::-1])
        # Normalize detail magnitude to 8-bit range
        details_norm = cv2.normalize(details_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Threshold detail magnitude to get defect mask
        _, wavelet_mask = cv2.threshold(details_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Add weighted result to confidence map
        confidence_map[wavelet_mask > 0] += algo_weights.get("wavelet", 0.4)
        # Log application of wavelet transform
        logging.debug("Applied wavelet transform for defect detection.")
    
    # Apply scratch dataset augmentation if enabled and path is provided
    if global_algo_params.get("scratch_dataset_path") and "dataset_scratch" in optional_algos:
        try:
            # Import scratch dataset handler module
            from scratch_dataset_handler import ScratchDatasetHandler 
            # Initialize scratch dataset handler with dataset path
            dataset_handler = ScratchDatasetHandler(global_algo_params["scratch_dataset_path"])
            # Ensure working_image_for_processing is suitable for augment_scratch_detection
            # Get scratch probability map from dataset handler
            scratch_prob = dataset_handler.augment_scratch_detection(working_image_for_processing) 
            # Add weighted result to confidence map
            confidence_map += scratch_prob * algo_weights.get("dataset_scratch", 0.5)
            # Log application of scratch dataset augmentation
            logging.debug("Applied scratch dataset augmentation.")
        except ImportError:
            # Log warning if module is not found
            logging.warning("ScratchDatasetHandler module not found. Skipping scratch dataset integration.")
        except Exception as e_sds: # Renamed e to e_sds
            # Log other errors during dataset integration
            logging.warning(f"Scratch dataset integration failed: {e_sds}")

    # Apply machine learning-based anomaly detection if enabled and available
    if "anomaly" in optional_algos and ANOMALY_DETECTION_AVAILABLE: 
        try:
            # Initialize anomaly detector with model path
            anomaly_detector = AnomalyDetector(global_algo_params.get("anomaly_model_path"))
            # Ensure working_image_for_processing is suitable for detect_anomalies (e.g. BGR or Grayscale as expected by model)
            # AnomalyDetector's detect_anomalies handles BGR/Grayscale conversion to RGB if needed
            # Get anomaly mask from detector
            anomaly_mask = anomaly_detector.detect_anomalies(working_image_for_processing) 
            # Add weighted result to confidence map if detection was successful
            if anomaly_mask is not None:
                confidence_map[anomaly_mask > 0] += algo_weights.get("anomaly", 0.5)
                # Log application of anomaly detection
                logging.debug("Applied anomaly detection for defects.")
        except Exception as e_anomaly: # Renamed e to e_anomaly
            # Log anomaly detection failure
            logging.warning(f"Anomaly detection failed: {e_anomaly}")
    # Log warning if anomaly detection is specified but module is unavailable
    elif "anomaly" in optional_algos and not ANOMALY_DETECTION_AVAILABLE:
        logging.warning("Anomaly detection algorithm specified, but AnomalyDetector module is not available.")

    # Get confidence threshold from configuration
    confidence_threshold_from_config = detection_cfg.get("confidence_threshold", 0.9) 
    # Define zone-specific adaptive thresholds
    zone_adaptive_threshold_map_dd = { # Renamed var
        "Core": 0.7,      
        "Cladding": 0.9,  
        "Adhesive": 1.1,  
        "Contact": 1.2
    }
    # Get adaptive threshold for the current zone
    adaptive_threshold_val_dd = zone_adaptive_threshold_map_dd.get(zone_name, confidence_threshold_from_config) # Renamed var
    # Ensure threshold value is numeric
    assert isinstance(adaptive_threshold_val_dd, (float, int)), \
        f"adaptive_threshold_val_dd is expected to be a number, got {type(adaptive_threshold_val_dd)}" #

    # Operator issues with adaptive_threshold_val_dd being None are unlikely here due to defaults
    # Create high-confidence defect mask
    high_confidence_mask = (confidence_map >= adaptive_threshold_val_dd).astype(np.uint8) * 255 #
    # Create medium-confidence defect mask
    medium_confidence_mask = ((confidence_map >= adaptive_threshold_val_dd * 0.7) & #
                              (confidence_map < adaptive_threshold_val_dd)).astype(np.uint8) * 128 #

    # Combine high and medium confidence masks
    combined_defect_mask_dd = cv2.bitwise_or(high_confidence_mask, medium_confidence_mask) # Renamed var
    # Find connected components in the combined mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_defect_mask_dd, connectivity=8) 
    # Initialize final defect mask
    final_defect_mask_in_zone = np.zeros_like(combined_defect_mask_dd, dtype=np.uint8) 

    # Define minimum area thresholds by confidence level
    min_area_by_confidence_map_dd = { # Renamed var
        255: detection_cfg.get("min_defect_area_px_high_conf", 5), 
        128: detection_cfg.get("min_defect_area_px_med_conf", 10)  
    }
    # Get default minimum area
    default_min_area = detection_cfg.get("min_defect_area_px", 5)

    # Filter defects by size based on confidence level
    for i in range(1, num_labels): 
        # Get area of current component
        area = stats[i, cv2.CC_STAT_AREA]
        # Create mask for current component
        component_mask = (labels == i)
        # Check if component mask is valid
        if np.any(component_mask):
            # Get confidence level of the component
            mask_val = combined_defect_mask_dd[component_mask].max() 
            # Get minimum area for this confidence level
            min_area = min_area_by_confidence_map_dd.get(mask_val, default_min_area)
            # Keep component if area is above threshold
            if area >= min_area:
                final_defect_mask_in_zone[component_mask] = 255 
        else:
            # Log skipping of empty labeled region
            logging.debug(f"Skipping empty labeled region {i} during size-based filtering.")

    # Define defect validation function to reduce false positives
    def validate_defect_mask(defect_mask, original_image_for_validation, zone_name_for_validation): # Renamed args for clarity
        """Validate defects using additional criteria to reduce false positives."""
        # Initialize mask for validated defects
        validated_mask = np.zeros_like(defect_mask)
        
        # Find all potential defects
        # Find connected components in the defect mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
        
        # Iterate through each potential defect
        for i in range(1, num_labels):
            # Extract bounding box and area of the component
            x, y, w, h, area = stats[i]
            
            # Extract defect region from original image
            # Ensure ROI coordinates are within image bounds
            # Define region of interest (ROI) for the defect
            y_end, x_end = min(y + h, original_image_for_validation.shape[0]), min(x + w, original_image_for_validation.shape[1])
            # Extract defect ROI from the image
            defect_roi = original_image_for_validation[y:y_end, x:x_end]
            
            # Ensure labels ROI matches defect_roi shape
            # Create a mask for the current defect component within the ROI
            defect_mask_roi = (labels[y:y_end, x:x_end] == i).astype(np.uint8)


            # Calculate contrast with surrounding area
            # Check for shape mismatch between ROI and mask
            if defect_roi.shape[0] != defect_mask_roi.shape[0] or defect_roi.shape[1] != defect_mask_roi.shape[1]:
                # This case should ideally not happen if stats and labels are consistent.
                # If it does, skip this defect or log an error.
                # Log warning if shape mismatch occurs
                logging.warning(f"Shape mismatch between defect_roi {defect_roi.shape} and defect_mask_roi {defect_mask_roi.shape} for defect component {i}. Skipping contrast check.")
                # Keep defect if it passes area check, even if contrast check fails
                if area >= min_area_by_confidence_map_dd.get(defect_mask[labels == i].max() if np.any(labels==i) else default_min_area, default_min_area): # Check area again before adding
                     validated_mask[labels == i] = 255 # Add defect if area is fine but contrast check failed due to shape
                # Continue to next defect
                continue

            # Define kernel size for creating surrounding area
            surrounding_kernel_size = 5 # Define kernel size for dilation
            # Dilate defect mask to get surrounding area
            dilated_defect_mask_roi = cv2.dilate(defect_mask_roi, np.ones((surrounding_kernel_size,surrounding_kernel_size), np.uint8))
            # Create mask for surrounding area by subtracting original defect
            surrounding_mask = dilated_defect_mask_roi - defect_mask_roi
            
            # Check if defect and surrounding masks are valid
            if np.sum(defect_mask_roi) > 0 and np.sum(surrounding_mask) > 0:
                # Ensure defect_roi is not empty before trying to access pixels.
                # Also ensure defect_mask_roi has some True values to avoid errors with empty slices.
                # Extract pixel values of defect and surrounding regions
                defect_pixels = defect_roi[defect_mask_roi > 0]
                surrounding_pixels = defect_roi[surrounding_mask > 0]

                # Calculate contrast if both regions have pixels
                if defect_pixels.size > 0 and surrounding_pixels.size > 0:
                    # Calculate mean intensity of defect
                    defect_mean = np.mean(defect_pixels)
                    # Calculate mean intensity of surrounding area
                    surrounding_mean = np.mean(surrounding_pixels)
                    # Calculate contrast as absolute difference of means
                    contrast = abs(defect_mean - surrounding_mean)
                else:
                    # Set contrast to 0 if not enough pixels
                    contrast = 0 # Not enough pixels for contrast calculation
                
                # Zone-specific validation thresholds
                # Define minimum contrast thresholds for each zone
                min_contrast = {
                    "Core": 15,      # Core requires higher contrast
                    "Cladding": 10,  # Cladding moderate contrast
                    "Adhesive": 8,   # Adhesive lower contrast
                    "Contact": 5     # Contact lowest contrast
                }.get(zone_name_for_validation, 10) # Use renamed arg
                
                # Default min_area for validation, can be different from initial filtering
                # Define minimum area for validation
                min_validation_area = 5 
                
                # Validate based on contrast and size
                # Keep defect if it meets both contrast and area criteria
                if contrast >= min_contrast and area >= min_validation_area :
                    validated_mask[labels == i] = 255
                # Log failure if contrast is too low
                elif area >= min_validation_area : # If contrast is low, but area is okay, still consider it if other filters passed
                    # This is a policy decision: what to do if contrast is low.
                    # For now, let's assume if it passed other filters, it's a candidate,
                    # but ideally, contrast should be a strong factor.
                    # If strict contrast is required, remove this elif.
                    # Let's keep it for now to be less aggressive in filtering here.
                    # validated_mask[labels == i] = 255 # (Original logic was to add it)
                    # Let's be stricter: if contrast fails, it fails validation here.
                    # Log contrast check failure
                    logging.debug(f"Defect component {i} failed contrast check ({contrast:.1f} < {min_contrast}). Area: {area}")
                else:
                    # Log area check failure during validation
                    logging.debug(f"Defect component {i} failed area check during validation ({area} < {min_validation_area}) or other issue.")

            # Keep defect if it passes area check but has no surrounding for contrast
            elif area >= min_area_by_confidence_map_dd.get(defect_mask[labels == i].max() if np.any(labels == i) else default_min_area, default_min_area): # If area is fine but no surrounding for contrast
                 validated_mask[labels == i] = 255 # Keep it if area is okay

        # Return the mask of validated defects
        return validated_mask
    
    # Define size-based validation function specific to zones
    def validate_defect_by_size(defect_mask: np.ndarray, zone_name: str, um_per_px: Optional[float] = None) -> np.ndarray:
        """Additional size-based validation specific to zones."""
        # Create a copy of the input mask to modify
        validated_mask = defect_mask.copy()
        
        # Zone-specific minimum sizes (in pixels if no um_per_px, otherwise in um²)
        # Define minimum defect sizes for each zone
        min_sizes = {
            "Core": 3 if um_per_px is None else (2.0 / (um_per_px ** 2)),  # 2 µm²
            "Cladding": 5 if um_per_px is None else (5.0 / (um_per_px ** 2)),  # 5 µm²
            "Adhesive": 10 if um_per_px is None else (20.0 / (um_per_px ** 2)),  # 20 µm²
            "Contact": 20 if um_per_px is None else (50.0 / (um_per_px ** 2))  # 50 µm²
        }
        
        # Get minimum area for the current zone
        min_area = min_sizes.get(zone_name, 5)
        
        # Remove components smaller than zone-specific minimum
        # Find connected components in the mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(validated_mask, connectivity=8)
        
        # Iterate through components and remove small ones
        for i in range(1, num_labels):
            # Remove component if its area is below the minimum
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                validated_mask[labels == i] = 0
        
        # Return the size-validated mask
        return validated_mask
    # Add validation before returning
    # The error was here: 'working_image' was not defined.
    # It should be 'working_image_for_processing' which is the most up-to-date image for the current zone.
    # Apply final validation step to the defect mask
    final_defect_mask_in_zone = validate_defect_mask(final_defect_mask_in_zone, working_image_for_processing, zone_name)
    # Ensure final mask is within the current zone
    final_defect_mask_in_zone = cv2.bitwise_and(final_defect_mask_in_zone, final_defect_mask_in_zone, mask=zone_mask)
    # Create final cleaning kernel
    kernel_clean_final_dd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) # Renamed var
    # Apply morphological opening for final noise removal
    final_defect_mask_in_zone = cv2.morphologyEx(final_defect_mask_in_zone, cv2.MORPH_OPEN, kernel_clean_final_dd)
    # Apply morphological closing to connect small gaps
    final_defect_mask_in_zone = cv2.morphologyEx(final_defect_mask_in_zone, cv2.MORPH_CLOSE, kernel_clean_final_dd)

    # Log completion of defect detection for the zone
    logging.debug(f"Defect detection fusion complete for zone '{zone_name}'. Adaptive threshold: {adaptive_threshold_val_dd:.2f}. Fallback config threshold: {confidence_threshold_from_config:.2f}.")
    # Return the final defect mask and the confidence map
    return final_defect_mask_in_zone, confidence_map

# Define Local Binary Pattern (LBP) based defect detection function
def _lbp_defect_detection(gray_img: np.ndarray) -> np.ndarray:
    """
    Local Binary Pattern detection for texture-based defects
    """
    # Use input image as grayscale
    processed_gray_img = gray_img
    # Normalize image to 8-bit if not already
    if gray_img.dtype != np.uint8:
        processed_gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #
    
    # Define LBP parameters: radius and number of points
    radius = 1
    n_points = 8 * radius 
    # Use 'uniform' LBP method for rotation invariance
    METHOD = 'uniform' 
    # Apply LBP operator
    lbp = local_binary_pattern(processed_gray_img, n_points, radius, method=METHOD)
    # Normalize LBP result to 8-bit range for visualization and thresholding
    lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #
    # Apply adaptive threshold to LBP image to find texture anomalies
    thresh = cv2.adaptiveThreshold(lbp_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2) 
    # Create elliptical kernel for noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Apply morphological opening to remove noise from thresholded image
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Return binary mask of texture defects
    return thresh

# Main block for standalone testing of the module
if __name__ == "__main__":
    # Configure basic logging for console output
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s') 

    # Define a dummy processing profile for testing purposes
    dummy_profile_config_main_test = { # Renamed var
        # Preprocessing parameters section
        "preprocessing": {
            "clahe_clip_limit": 2.0, "clahe_tile_grid_size": [8, 8], 
            "gaussian_blur_kernel_size": [5, 5], "enable_illumination_correction": False 
        },
        # Localization parameters section
        "localization": {
            "hough_dp": 1.2, "hough_min_dist_factor": 0.15, "hough_param1": 70, 
            "hough_param2": 35, "hough_min_radius_factor": 0.08, 
            "hough_max_radius_factor": 0.45, "use_ellipse_detection": True, "use_circle_fit": True 
        },
        # Defect detection parameters section
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
    # Load global algorithm parameters from dummy config
    dummy_global_algo_params_main_test = get_config().get("algorithm_parameters", {}) # Renamed var
    # Update global parameters with test-specific values
    dummy_global_algo_params_main_test.update({
        "do2mr_gamma_default": 1.5, "do2mr_gamma_core": 1.2,
        "multiscale_factors": [0.5, 1.0, 1.5], 
    })
    
    # Define dummy zone definitions for testing
    dummy_zone_defs_main_test = [ # Renamed var
        {"name": "Core", "r_min_factor": 0.0, "r_max_factor_core_relative": 1.0, "color_bgr": [255,0,0]},
        {"name": "Cladding", "r_min_factor_cladding_relative": 0.0, "r_max_factor_cladding_relative": 1.0, "color_bgr": [0,255,0]},
        {"name": "Adhesive", "r_min_factor_cladding_relative": 1.0, "r_max_factor_cladding_relative": 1.15, "color_bgr": [0,0,255]},
    ]

    # Define path for the test image
    test_image_path_str = "sample_fiber_image.png" 
    # Create a dummy image if it doesn't exist
    if not Path(test_image_path_str).exists(): 
        # Define dummy image dimensions
        dummy_img_arr_h, dummy_img_arr_w = 600, 800
        # Create a gray background
        dummy_img_arr = np.full((dummy_img_arr_h, dummy_img_arr_w), 128, dtype=np.uint8) 
        # Corrected colors for cv2.circle and cv2.line
        # Draw a brighter circle to simulate the cladding
        cv2.circle(dummy_img_arr, (dummy_img_arr_w//2, dummy_img_arr_h//2), 150, (200,), -1) 
        # Draw a darker circle to simulate the core
        cv2.circle(dummy_img_arr, (dummy_img_arr_w//2, dummy_img_arr_h//2), 60, (50,), -1)   
        # Draw a dark line to simulate a scratch
        cv2.line(dummy_img_arr, (dummy_img_arr_w//2 - 100, dummy_img_arr_h//2 - 50), 
                 (dummy_img_arr_w//2 + 100, dummy_img_arr_h//2 + 50), (10,), 3) 
        # Draw a dark spot to simulate a pit
        cv2.circle(dummy_img_arr, (dummy_img_arr_w//2 + 50, dummy_img_arr_h//2 - 30), 15, (20,), -1) 
        # Add random noise to the image
        noise = np.random.randint(0, 15, (dummy_img_arr_h, dummy_img_arr_w), dtype=np.uint8)
        # Add noise to the image
        dummy_img_arr = cv2.add(dummy_img_arr, noise)
        # Convert the grayscale dummy image to BGR format for saving
        dummy_img_arr_bgr = cv2.cvtColor(dummy_img_arr, cv2.COLOR_GRAY2BGR) 
        # Save the dummy image to disk
        cv2.imwrite(test_image_path_str, dummy_img_arr_bgr) 
        # Log the creation of the dummy image
        logging.info(f"Created a dummy image at {test_image_path_str} for testing.")

    # Start Test Case 1: Image Loading and Preprocessing
    logging.info(f"\n--- Test Case 1: Load and Preprocess Image: {test_image_path_str} ---")
    # Call the preprocessing function
    preprocess_result = load_and_preprocess_image(test_image_path_str, dummy_profile_config_main_test) 
    
    # Check if preprocessing was successful
    if preprocess_result: 
        # Unpack the results from preprocessing
        original_bgr_test, gray_test, processed_test = preprocess_result 
        # Log the shape of the processed image
        logging.info(f"Image preprocessed. Shape of processed image: {processed_test.shape}")
        
        # Start Test Case 2: Fiber Localization
        logging.info("\n--- Test Case 2: Locate Fiber Structure ---")
        # Call the fiber localization function
        localization = locate_fiber_structure(processed_test, dummy_profile_config_main_test, original_gray_image=gray_test) 
        
        # Check if localization was successful
        if localization: 
            # Log the localization results
            logging.info(f"Fiber Localization: {localization}")
            # Create a copy of the original image for visualization
            viz_loc_img = original_bgr_test.copy()
            # Draw the detected cladding circle if found
            if 'cladding_center_xy' in localization and 'cladding_radius_px' in localization:
                # Get cladding center coordinates
                cc_loc = localization['cladding_center_xy'] # Renamed var
                # Get cladding radius
                cr_loc = int(localization['cladding_radius_px']) # Renamed var
                # Draw cladding circle in green
                cv2.circle(viz_loc_img, cc_loc, cr_loc, (0,255,0), 2) 
                # Draw the detected ellipse if found
                if 'cladding_ellipse_params' in localization:
                     # Draw ellipse in cyan
                     cv2.ellipse(viz_loc_img, localization['cladding_ellipse_params'], (0,255,255), 2) 
            # Draw the detected core circle if found
            if 'core_center_xy' in localization and 'core_radius_px' in localization:
                # Get core center coordinates
                coc_loc = localization['core_center_xy'] # Renamed var
                # Get core radius
                cor_loc = int(localization['core_radius_px']) # Renamed var
                # Draw core circle in blue
                cv2.circle(viz_loc_img, coc_loc, cor_loc, (255,0,0), 2) 
            
            # Start Test Case 3: Zone Mask Generation
            logging.info("\n--- Test Case 3: Generate Zone Masks ---")
            # Define dummy calibration and user input values for testing
            um_per_px_test = 0.5 
            user_core_diam_test = 9.0 
            user_cladding_diam_test = 125.0 
            
            # Call the zone mask generation function
            zone_masks_generated = generate_zone_masks( 
                processed_test.shape, localization, dummy_zone_defs_main_test,
                um_per_px=um_per_px_test, 
                user_core_diameter_um=user_core_diam_test, 
                user_cladding_diameter_um=user_cladding_diam_test
            )
            # Check if zone masks were generated successfully
            if zone_masks_generated: 
                # Log the names of the generated zones
                logging.info(f"Generated masks for zones: {list(zone_masks_generated.keys())}")
                
                # Start Test Case 4: Defect Detection
                logging.info("\n--- Test Case 4: Detect Defects (Iterating Zones) ---")
                # Initialize a blank image to visualize combined defects
                combined_defects_viz = np.zeros_like(processed_test, dtype=np.uint8)

                # Iterate through each generated zone to detect defects
                for zone_n_test, zone_m_test in zone_masks_generated.items(): # Renamed vars
                    # Skip empty zones
                    if np.sum(zone_m_test) == 0:
                        # Log skipping of empty zone
                        logging.info(f"Skipping defect detection for empty zone: {zone_n_test}")
                        # Continue to the next zone
                        continue
                    
                    # Log the start of defect detection for the current zone
                    logging.info(f"--- Detecting defects in Zone: {zone_n_test} ---")
                    # Call the main defect detection function
                    defects_mask, conf_map = detect_defects( 
                        processed_test, zone_m_test, zone_n_test, 
                        dummy_profile_config_main_test, dummy_global_algo_params_main_test
                    )
                    # Log the result of defect detection for the zone
                    logging.info(f"Defect detection in '{zone_n_test}' zone complete. Found {np.sum(defects_mask > 0)} defect pixels.")
                    # If defects are found, add them to the combined visualization image
                    if np.any(defects_mask):
                        combined_defects_viz = cv2.bitwise_or(combined_defects_viz, defects_mask)
                
                # Create a final visualization image with defects overlaid
                final_viz_img = original_bgr_test.copy()
                # Overlay detected defects in red on the final visualization
                final_viz_img[combined_defects_viz > 0] = [0,0,255] 
            # Log failure if zone mask generation failed
            else: 
                logging.warning("Zone mask generation failed for defect detection test.")
        # Log failure if fiber localization failed
        else: 
            logging.error("Fiber localization failed. Cannot proceed with further tests.")
    # Log failure if image preprocessing failed
    else: 
        logging.error("Image preprocessing failed.")

    # Clean up the dummy image file after testing
    if Path(test_image_path_str).exists() and test_image_path_str == "sample_fiber_image.png":
        try:
            # Delete the dummy image file
            Path(test_image_path_str).unlink()
            # Log successful cleanup
            logging.info(f"Cleaned up dummy image: {test_image_path_str}")
        # Handle potential OS errors during file deletion
        except OSError as e_os_error: # Renamed e to e_os_error
            # Log error if cleanup fails
            logging.error(f"Error removing dummy image {test_image_path_str}: {e_os_error}")