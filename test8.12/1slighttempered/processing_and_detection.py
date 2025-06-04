#!/usr/bin/env python3
"""
Part 2: Image Processing and Defect Detection Logic
Advanced Fiber Optic End Face Defect Detection System
=====================================================
This file contains the methods for image preprocessing, fiber zone identification,
and defect detection algorithms (DO2MR, LEI, and others) to be integrated
into the FiberInspector class from part 1.

Author: Gemini AI
Date: June 4, 2025
Version: 1.0
"""

# Imports from Part 1 (assuming config_and_setup.py is in the same directory or accessible)
# from config_and_setup import (
#     FiberInspector, InspectorConfig, FiberSpecifications, ZoneDefinition,
#     DetectedZoneInfo, DefectInfo, DefectMeasurement, ImageResult, ImageAnalysisStats,
#     _log_message, _start_timer, _log_duration
# )
# For standalone execution of this part for development, you might need to
# uncomment the above or ensure the classes are defined.
# For the final combined script, these imports won't be needed here if methods are added to the class.

import cv2 # OpenCV for image processing operations
import numpy as np # NumPy for efficient numerical computations
from typing import Dict, List, Tuple, Optional, Any # Type hints for better code clarity
from pathlib import Path # Path handling
from datetime import datetime # Datetime for timestamping
import time # For timing operations

# --- Helper Functions (if any specific to this part) ---
# (None for now, assuming utility functions are in config_and_setup.py)

# --- Methods to be added to the FiberInspector class ---

class FiberInspector: # Extended from Part 1
    # --- Preprocessing Methods ---
    def _preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Applies various preprocessing techniques to the input image.
        Args:
            image: The input BGR image.
        Returns:
            A dictionary of preprocessed images (grayscale), including 'original_gray',
            'gaussian_blurred', 'bilateral_filtered', 'clahe_enhanced', 'hist_equalized'.
        """
        # Start timer for preprocessing.
        preprocess_start_time = _start_timer()
        # Log the start of preprocessing.
        _log_message("Starting image preprocessing...")

        # Ensure the input image is not None.
        if image is None:
            _log_message("Input image for preprocessing is None.", level="ERROR")
            # Return an empty dictionary or raise an error.
            return {}

        # Convert to grayscale if it's a color image.
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR image to grayscale.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            # Image is already grayscale.
            gray = image.copy() # Work on a copy.
        else:
            # Log an error for unsupported image format.
            _log_message(f"Unsupported image format for preprocessing: shape {image.shape}", level="ERROR")
            # Return an empty dictionary.
            return {}

        # Initialize a dictionary to store processed images.
        processed_images: Dict[str, np.ndarray] = {}
        # Store the original grayscale image.
        processed_images['original_gray'] = gray.copy()

        # 1. Gaussian Blur
        try:
            # Apply Gaussian blur using parameters from config.
            blurred = cv2.GaussianBlur(gray, self.config.GAUSSIAN_BLUR_KERNEL_SIZE, self.config.GAUSSIAN_BLUR_SIGMA)
            # Store the blurred image.
            processed_images['gaussian_blurred'] = blurred
        except Exception as e:
            # Log error if Gaussian blur fails.
            _log_message(f"Error during Gaussian Blur: {e}", level="WARNING")
            # Store original gray if blur fails.
            processed_images['gaussian_blurred'] = gray.copy()

        # 2. Bilateral Filter (Edge-preserving smoothing)
        try:
            # Apply Bilateral filter using parameters from config.
            bilateral = cv2.bilateralFilter(gray, self.config.BILATERAL_FILTER_D,
                                            self.config.BILATERAL_FILTER_SIGMA_COLOR,
                                            self.config.BILATERAL_FILTER_SIGMA_SPACE)
            # Store the bilaterally filtered image.
            processed_images['bilateral_filtered'] = bilateral
        except Exception as e:
            # Log error if Bilateral filter fails.
            _log_message(f"Error during Bilateral Filter: {e}", level="WARNING")
            # Store original gray if filter fails.
            processed_images['bilateral_filtered'] = gray.copy()

        # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        try:
            # Create a CLAHE object with parameters from config.
            clahe = cv2.createCLAHE(clipLimit=self.config.CLAHE_CLIP_LIMIT,
                                    tileGridSize=self.config.CLAHE_TILE_GRID_SIZE)
            # Apply CLAHE to the bilaterally filtered image for better results.
            clahe_enhanced = clahe.apply(processed_images.get('bilateral_filtered', gray))
            # Store the CLAHE enhanced image.
            processed_images['clahe_enhanced'] = clahe_enhanced
        except Exception as e:
            # Log error if CLAHE fails.
            _log_message(f"Error during CLAHE: {e}", level="WARNING")
            # Store original gray if CLAHE fails.
            processed_images['clahe_enhanced'] = gray.copy()

        # 4. Standard Histogram Equalization
        try:
            # Apply standard histogram equalization.
            hist_equalized = cv2.equalizeHist(gray)
            # Store the histogram equalized image.
            processed_images['hist_equalized'] = hist_equalized
        except Exception as e:
            # Log error if histogram equalization fails.
            _log_message(f"Error during Histogram Equalization: {e}", level="WARNING")
            # Store original gray if equalization fails.
            processed_images['hist_equalized'] = gray.copy()

        # Log the duration of preprocessing.
        _log_duration("Image Preprocessing", preprocess_start_time, self.current_image_result)
        # Return the dictionary of processed images.
        return processed_images

    # --- Fiber Center and Zone Detection Methods ---
    def _find_fiber_center_and_radius(self, processed_images: Dict[str, np.ndarray]) -> Optional[Tuple[Tuple[int, int], float]]:
        """
        Robustly finds the primary circular feature (assumed cladding) center and radius.
        Uses Hough Circle Transform on multiple preprocessed images and selects the best candidate.
        Args:
            processed_images: Dictionary of preprocessed grayscale images.
        Returns:
            A tuple (center_xy, radius_px) or None if no reliable circle is found.
        """
        # Start timer for this operation.
        detection_start_time = _start_timer()
        # Log the start of fiber center detection.
        _log_message("Starting fiber center and radius detection...")

        # List to store all detected circle candidates (cx, cy, r, confidence, source_image_key).
        all_detected_circles: List[Tuple[int, int, int, float, str]] = []
        # Get the original grayscale image dimensions for reference.
        h, w = processed_images['original_gray'].shape[:2]
        # Minimum distance between circle centers, scaled by image size.
        min_dist_circles = int(min(h, w) * self.config.HOUGH_MIN_DIST_FACTOR)
        # Minimum and maximum radius for Hough circles, scaled by image size.
        min_radius_hough = int(min(h, w) * self.config.HOUGH_MIN_RADIUS_FACTOR)
        max_radius_hough = int(min(h, w) * self.config.HOUGH_MAX_RADIUS_FACTOR)

        # Iterate over different preprocessed images suitable for Hough transform.
        # Prefer images with good contrast and smoothed edges.
        for image_key in ['gaussian_blurred', 'bilateral_filtered', 'clahe_enhanced']:
            # Get the specific preprocessed image.
            img_to_process = processed_images.get(image_key)
            # Continue if the image is not available.
            if img_to_process is None: continue

            # Iterate over different Hough Circle parameters for robustness.
            for dp in self.config.HOUGH_DP_VALUES:
                for param1 in self.config.HOUGH_PARAM1_VALUES:
                    for param2 in self.config.HOUGH_PARAM2_VALUES:
                        try:
                            # Detect circles using HoughCircles.
                            circles = cv2.HoughCircles(
                                img_to_process, # Input image.
                                cv2.HOUGH_GRADIENT, # Detection method.
                                dp=dp, # Inverse ratio of accumulator resolution.
                                minDist=min_dist_circles, # Min distance between centers.
                                param1=param1, # Upper Canny threshold.
                                param2=param2, # Accumulator threshold.
                                minRadius=min_radius_hough, # Min radius.
                                maxRadius=max_radius_hough  # Max radius.
                            )
                            # If circles are found.
                            if circles is not None:
                                # Convert circle parameters to integers.
                                circles = np.uint16(np.around(circles))
                                # Process each detected circle.
                                for i in circles[0, :]:
                                    cx, cy, r = int(i[0]), int(i[1]), int(i[2])
                                    # Basic confidence: circles closer to center with reasonable radius are better.
                                    # This is a simple heuristic; more advanced confidence scoring can be added.
                                    dist_to_img_center = np.sqrt((cx - w//2)**2 + (cy - h//2)**2)
                                    # Confidence score (higher is better). Penalize distance from center.
                                    # Normalize radius to avoid bias for very large/small images.
                                    normalized_r = r / max_radius_hough if max_radius_hough > 0 else 0
                                    # Confidence based on param2 (accumulator threshold) and normalized radius.
                                    confidence = (param2 / max(self.config.HOUGH_PARAM2_VALUES)) * 0.5 + \
                                                 (normalized_r) * 0.5 - \
                                                 (dist_to_img_center / (min(h,w)/2)) * 0.2 # Penalize off-center
                                    confidence = max(0, min(1, confidence)) # Clamp between 0 and 1.

                                    # Add detected circle with its confidence and source.
                                    all_detected_circles.append((cx, cy, r, confidence, image_key))
                        except Exception as e:
                            # Log any error during HoughCircles.
                            _log_message(f"Error in HoughCircles on {image_key} with params ({dp},{param1},{param2}): {e}", level="WARNING")

        # Check if any circles were detected.
        if not all_detected_circles:
            # Log warning if no circles found.
            _log_message("No circles detected by Hough Transform.", level="WARNING")
            # Log duration and return None.
            _log_duration("Fiber Center Detection (No Circles)", detection_start_time, self.current_image_result)
            return None

        # Select the best circle based on confidence score.
        # Sort circles by confidence in descending order.
        all_detected_circles.sort(key=lambda x: x[3], reverse=True)
        # Get the best circle (highest confidence).
        best_circle_cx, best_circle_cy, best_circle_r, best_confidence, source = all_detected_circles[0]

        # Check if the best confidence meets a threshold.
        if best_confidence < self.config.CIRCLE_CONFIDENCE_THRESHOLD:
            _log_message(f"Best detected circle confidence ({best_confidence:.2f} from {source}) is below threshold ({self.config.CIRCLE_CONFIDENCE_THRESHOLD}).", level="WARNING")
            _log_duration("Fiber Center Detection (Low Confidence)", detection_start_time, self.current_image_result)
            return None

        # Log the details of the best detected circle.
        _log_message(f"Best fiber center detected at ({best_circle_cx}, {best_circle_cy}) with radius {best_circle_r}px. Confidence: {best_confidence:.2f} (from {source}).")
        # Log the duration of the operation.
        _log_duration("Fiber Center Detection", detection_start_time, self.current_image_result)
        # Return the center and radius of the best circle.
        return (best_circle_cx, best_circle_cy), float(best_circle_r)


    def _calculate_pixels_per_micron(self, detected_cladding_radius_px: float) -> Optional[float]:
        """
        Calculates the pixels_per_micron ratio.
        This is called if operating_mode is MICRON_CALCULATED or MICRON_INFERRED.
        Args:
            detected_cladding_radius_px: The detected radius of the cladding in pixels.
        Returns:
            The calculated pixels_per_micron, or None if it cannot be determined.
        """
        # Start timer for this calculation.
        calc_start_time = _start_timer()
        # Log the start of pixel per micron calculation.
        _log_message("Calculating pixels per micron...")

        # Initialize pixels_per_micron to None.
        calculated_ppm: Optional[float] = None

        # Check if operating in a mode that requires micron conversion and if specs are available.
        if self.operating_mode in ["MICRON_CALCULATED", "MICRON_INFERRED"]:
            # Ensure cladding diameter in microns is available from fiber_specs.
            if self.fiber_specs.cladding_diameter_um is not None and self.fiber_specs.cladding_diameter_um > 0:
                # Ensure detected cladding radius is valid.
                if detected_cladding_radius_px > 0:
                    # Calculate pixels per micron: (2 * radius_px) / diameter_um.
                    calculated_ppm = (2 * detected_cladding_radius_px) / self.fiber_specs.cladding_diameter_um
                    # Store this in the inspector instance and current image result.
                    self.pixels_per_micron = calculated_ppm
                    if self.current_image_result:
                         self.current_image_result.stats.microns_per_pixel = 1.0 / calculated_ppm if calculated_ppm > 0 else None
                    # Log the calculated ratio.
                    _log_message(f"Calculated pixels_per_micron: {calculated_ppm:.4f} px/µm (µm/px: {1/calculated_ppm:.4f}).")
                else:
                    # Log warning if detected radius is invalid.
                    _log_message("Detected cladding radius is zero or negative, cannot calculate px/µm.", level="WARNING")
            else:
                # Log warning if cladding diameter spec is missing.
                _log_message("Cladding diameter in microns not specified, cannot calculate px/µm.", level="WARNING")
        else:
            # Log info if not in a micron conversion mode.
            _log_message("Not in MICRON_CALCULATED or MICRON_INFERRED mode, skipping px/µm calculation.", level="DEBUG")

        # Log the duration of the calculation.
        _log_duration("Pixels per Micron Calculation", calc_start_time, self.current_image_result)
        # Return the calculated ratio.
        return calculated_ppm

    def _create_zone_masks(self, image_shape: Tuple[int, int],
                           fiber_center_px: Tuple[int, int],
                           detected_cladding_radius_px: float) -> Dict[str, DetectedZoneInfo]:
        """
        Creates binary masks for each defined fiber zone (core, cladding, ferrule, etc.).
        Args:
            image_shape: Tuple (height, width) of the image.
            fiber_center_px: Tuple (cx, cy) of the detected fiber center in pixels.
            detected_cladding_radius_px: The radius of the detected cladding in pixels.
        Returns:
            A dictionary where keys are zone names and values are DetectedZoneInfo objects.
        """
        # Start timer for mask creation.
        mask_start_time = _start_timer()
        # Log the start of zone mask creation.
        _log_message("Creating zone masks...")

        # Initialize dictionary to store zone information.
        detected_zones_info: Dict[str, DetectedZoneInfo] = {}
        # Get image height and width.
        h, w = image_shape[:2]
        # Get fiber center coordinates.
        cx, cy = fiber_center_px

        # Create a radial distance map from the center.
        # Create Y and X coordinate grids.
        y_coords, x_coords = np.ogrid[:h, :w]
        # Calculate squared distance from center for each pixel.
        dist_from_center_sq = (x_coords - cx)**2 + (y_coords - cy)**2

        # Iterate through the active zone definitions.
        for zone_def in self.active_zone_definitions:
            # Initialize radii in pixels.
            r_min_px: float = 0
            r_max_px: float = 0
            # Initialize radius in microns.
            r_min_um: Optional[float] = None
            r_max_um: Optional[float] = None

            # Determine radii based on operating mode.
            if self.operating_mode == "PIXEL_ONLY" or self.operating_mode == "MICRON_INFERRED":
                # Radii are factors of the detected cladding radius.
                r_min_px = zone_def.r_min_factor_or_um * detected_cladding_radius_px
                r_max_px = zone_def.r_max_factor_or_um * detected_cladding_radius_px
                # If in MICRON_INFERRED mode and conversion is available, calculate micron radii.
                if self.operating_mode == "MICRON_INFERRED" and self.pixels_per_micron and self.pixels_per_micron > 0:
                    r_min_um = r_min_px / self.pixels_per_micron
                    r_max_um = r_max_px / self.pixels_per_micron
            elif self.operating_mode == "MICRON_CALCULATED":
                # Radii are given directly in microns, convert to pixels.
                if self.pixels_per_micron and self.pixels_per_micron > 0:
                    r_min_px = zone_def.r_min_factor_or_um * self.pixels_per_micron
                    r_max_px = zone_def.r_max_factor_or_um * self.pixels_per_micron
                    r_min_um = zone_def.r_min_factor_or_um
                    r_max_um = zone_def.r_max_factor_or_um
                else:
                    # Log warning if conversion ratio is missing in MICRON_CALCULATED mode.
                    _log_message(f"Pixels_per_micron not available in MICRON_CALCULATED mode for zone '{zone_def.name}'. Mask creation might be inaccurate.", level="WARNING")
                    # Fallback to treating factors as pixel values (less accurate).
                    r_min_px = zone_def.r_min_factor_or_um
                    r_max_px = zone_def.r_max_factor_or_um


            # Create the binary mask for the current zone (annulus).
            # Mask is 1 where r_min_px^2 <= dist_sq < r_max_px^2.
            zone_mask_np = ((dist_from_center_sq >= r_min_px**2) & (dist_from_center_sq < r_max_px**2)).astype(np.uint8) * 255
            # Store the zone information.
            detected_zones_info[zone_def.name] = DetectedZoneInfo(
                name=zone_def.name,
                center_px=fiber_center_px,
                radius_px=r_max_px, # Using r_max_px as the representative radius of the zone.
                radius_um=r_max_um,
                mask=zone_mask_np
            )
            # Log the created zone mask details.
            _log_message(f"Created mask for zone '{zone_def.name}': r_min={r_min_px:.2f}px, r_max={r_max_px:.2f}px.")

        # Log the duration of mask creation.
        _log_duration("Zone Mask Creation", mask_start_time, self.current_image_result)
        # Return the dictionary of zone information.
        return detected_zones_info

    # --- Defect Detection Algorithm Implementations ---
    def _detect_region_defects_do2mr(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """
        Detects region-based defects (dirt, pits, contamination) using a DO2MR-inspired method.
        Args:
            image_gray: Grayscale image of the zone to inspect.
            zone_mask: Binary mask for the current zone.
            zone_name: Name of the current zone (for logging).
        Returns:
            A binary mask of detected region defects, or None if an error occurs.
        """
        # Start timer for DO2MR detection.
        do2mr_start_time = _start_timer()
        # Log the start of DO2MR defect detection.
        _log_message(f"Starting DO2MR region defect detection for zone '{zone_name}'...")

        # Ensure image and mask are valid.
        if image_gray is None or zone_mask is None:
            _log_message("Input image or mask is None for DO2MR.", level="ERROR")
            return None
        # Apply the zone mask to the image.
        masked_image = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask)

        # Initialize a combined defect map.
        combined_defect_map = np.zeros_like(image_gray, dtype=np.uint8)
        # Count how many parameter sets detect each pixel as defect.
        vote_map = np.zeros_like(image_gray, dtype=np.float32)

        # Iterate over configured kernel sizes for DO2MR.
        for kernel_size in self.config.DO2MR_KERNEL_SIZES:
            # Create structuring element.
            struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            # Apply minimum filter (erosion).
            min_filtered = cv2.erode(masked_image, struct_element)
            # Apply maximum filter (dilation).
            max_filtered = cv2.dilate(masked_image, struct_element)
            # Calculate residual image (difference between max and min).
            residual = cv2.subtract(max_filtered, min_filtered)

            # Apply median blur to the residual image to reduce noise.
            if self.config.DO2MR_MEDIAN_BLUR_KERNEL_SIZE > 0:
                residual_blurred = cv2.medianBlur(residual, self.config.DO2MR_MEDIAN_BLUR_KERNEL_SIZE)
            else:
                residual_blurred = residual # Skip blur if kernel size is 0.

            # Iterate over configured gamma values for thresholding.
            for gamma in self.config.DO2MR_GAMMA_VALUES:
                # Calculate threshold using mean and standard deviation of the residual within the mask.
                # Consider only non-zero pixels in the masked residual for stats to avoid bias from masked-out areas.
                masked_residual_values = residual_blurred[zone_mask > 0]
                if masked_residual_values.size == 0: # Avoid error if mask is empty.
                    _log_message(f"Zone mask for '{zone_name}' is empty or residual is all zero. Skipping DO2MR for gamma={gamma}, kernel={kernel_size}.", level="WARNING")
                    continue

                mean_val = np.mean(masked_residual_values) # Mean of residual values.
                std_val = np.std(masked_residual_values) # Standard deviation of residual values.
                # Calculate dynamic threshold.
                threshold_val = mean_val + gamma * std_val
                # Ensure threshold is within valid 0-255 range.
                threshold_val = np.clip(threshold_val, 0, 255)

                # Apply threshold to create a binary defect mask.
                _, defect_mask_single_pass = cv2.threshold(residual_blurred, threshold_val, 255, cv2.THRESH_BINARY)
                # Ensure defects are only within the current zone.
                defect_mask_single_pass = cv2.bitwise_and(defect_mask_single_pass, defect_mask_single_pass, mask=zone_mask)

                # Apply morphological opening to remove small noise artifacts.
                if self.config.DO2MR_MORPH_OPEN_KERNEL_SIZE[0] > 0 and self.config.DO2MR_MORPH_OPEN_KERNEL_SIZE[1] > 0:
                    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.config.DO2MR_MORPH_OPEN_KERNEL_SIZE)
                    defect_mask_single_pass = cv2.morphologyEx(defect_mask_single_pass, cv2.MORPH_OPEN, open_kernel)

                # Add to the vote map.
                vote_map += (defect_mask_single_pass / 255.0) # Increment vote for detected pixels.

        # Final defect map based on votes (e.g., if detected by at least N parameter sets).
        # Number of parameter sets = len(kernels) * len(gammas).
        num_param_sets = len(self.config.DO2MR_KERNEL_SIZES) * len(self.config.DO2MR_GAMMA_VALUES)
        # Require a certain proportion of votes, e.g., 30% of sets.
        min_votes_required = max(1, int(num_param_sets * 0.3))
        # Create final defect map where vote count meets threshold.
        combined_defect_map = np.where(vote_map >= min_votes_required, 255, 0).astype(np.uint8)

        # Log the duration of DO2MR detection.
        _log_duration(f"DO2MR Detection for {zone_name}", do2mr_start_time, self.current_image_result)
        # Return the combined defect map.
        return combined_defect_map

    def _detect_scratches_lei(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """
        Detects linear scratches using an LEI-inspired method.
        Args:
            image_gray: Grayscale image of the zone to inspect.
            zone_mask: Binary mask for the current zone.
            zone_name: Name of the current zone (for logging).
        Returns:
            A binary mask of detected scratches, or None if an error occurs.
        """
        # Start timer for LEI detection.
        lei_start_time = _start_timer()
        # Log the start of LEI scratch detection.
        _log_message(f"Starting LEI scratch detection for zone '{zone_name}'...")

        # Ensure image and mask are valid.
        if image_gray is None or zone_mask is None:
            _log_message("Input image or mask is None for LEI.", level="ERROR")
            return None

        # Enhance image contrast, e.g., using CLAHE or simple histogram equalization.
        # Using CLAHE from preprocessed images if available, else equalizeHist.
        # For LEI, often direct equalization on the masked region is preferred.
        masked_image = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask)
        # Apply histogram equalization to the masked region to enhance scratches.
        enhanced_image = cv2.equalizeHist(masked_image)
        # Re-apply mask as equalizeHist might affect pixels outside the original mask.
        enhanced_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=zone_mask)

        # Initialize a map to store the maximum response from all orientations.
        max_response_map = np.zeros_like(enhanced_image, dtype=np.float32)

        # Iterate over different kernel lengths for multi-scale scratch detection.
        for kernel_length in self.config.LEI_KERNEL_LENGTHS:
            # Iterate through angles from 0 to 180 degrees.
            for angle_deg in range(0, 180, self.config.LEI_ANGLE_STEP):
                # Create a linear structuring element (kernel) for the current angle.
                # Kernel is a 1D line of `kernel_length`.
                line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
                # Get rotation matrix to rotate the kernel.
                # Center of rotation is at (kernel_length // 2, 0) assuming kernel is initially horizontal.
                rot_matrix = cv2.getRotationMatrix2D((kernel_length // 2, 0), angle_deg, 1.0)
                # Warp the kernel to rotate it. Output size needs to accommodate rotated kernel.
                # A square bounding box for the rotated kernel.
                rotated_kernel_bbox_size = int(np.ceil(kernel_length * 1.5)) # Ensure enough space.
                rotated_kernel = cv2.warpAffine(line_kernel, rot_matrix, (rotated_kernel_bbox_size, rotated_kernel_bbox_size))
                # Normalize the kernel (sum of elements = 1).
                # Ensure kernel sum is not zero to avoid division by zero.
                if np.sum(rotated_kernel) > 0:
                    rotated_kernel = rotated_kernel.astype(np.float32) / np.sum(rotated_kernel)
                else: # If kernel sum is zero (e.g., very small kernel_length or issue with warpAffine), skip.
                    continue

                # Apply the filter (convolution) to the enhanced image.
                # Using float32 for response to maintain precision.
                response = cv2.filter2D(enhanced_image.astype(np.float32), -1, rotated_kernel)
                # Update the max_response_map with the maximum response at each pixel.
                max_response_map = np.maximum(max_response_map, response)

        # Normalize the max_response_map to 0-255 range for thresholding.
        if np.max(max_response_map) > 0: # Avoid division by zero if map is all zeros.
            cv2.normalize(max_response_map, max_response_map, 0, 255, cv2.NORM_MINMAX)
        # Convert to 8-bit unsigned integer.
        response_8u = max_response_map.astype(np.uint8)

        # Threshold the response map to get binary scratch mask.
        # Using Otsu's thresholding, potentially with a factor if needed.
        # The LEI_THRESHOLD_FACTOR could be used to adjust Otsu's result or set a manual threshold.
        _, scratch_mask = cv2.threshold(response_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Alternative: threshold_val = np.mean(response_8u[zone_mask > 0]) + self.config.LEI_THRESHOLD_FACTOR * np.std(response_8u[zone_mask > 0])
        # _, scratch_mask = cv2.threshold(response_8u, threshold_val, 255, cv2.THRESH_BINARY)


        # Apply morphological closing to connect broken scratch segments.
        # Use an elongated kernel appropriate for linear features.
        if self.config.LEI_MORPH_CLOSE_KERNEL_SIZE[0] > 0 and self.config.LEI_MORPH_CLOSE_KERNEL_SIZE[1] > 0:
            # Kernel for closing, typically elongated in one direction.
            # We might want to try closing with kernels oriented along detected scratch directions,
            # but a general elongated kernel is simpler here.
            # Example: (length, thickness)
            close_kernel_shape = self.config.LEI_MORPH_CLOSE_KERNEL_SIZE
            # For general scratches, a small rectangular or elliptical kernel is often used.
            # If LEI_MORPH_CLOSE_KERNEL_SIZE is (5,1), it implies closing horizontal features.
            # A more general approach might be a small square or elliptical kernel.
            # Let's use a small elliptical kernel for general closing.
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) # Example
            scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, close_kernel)

        # Ensure scratches are only within the current zone.
        scratch_mask = cv2.bitwise_and(scratch_mask, scratch_mask, mask=zone_mask)

        # Filter out very small artifacts that might remain.
        # This is handled by MIN_DEFECT_AREA_PX or LEI_MIN_SCRATCH_AREA_PX later during contour analysis.

        # Log the duration of LEI detection.
        _log_duration(f"LEI Scratch Detection for {zone_name}", lei_start_time, self.current_image_result)
        # Return the binary scratch mask.
        return scratch_mask

    # --- Placeholder for other detection methods ---
    def _detect_defects_canny(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """Detects defects using Canny edge detection followed by morphological operations."""
        # Log start.
        _log_message(f"Starting Canny defect detection for zone '{zone_name}'...")
        # Apply Canny edge detection.
        edges = cv2.Canny(image_gray, self.config.CANNY_LOW_THRESHOLD, self.config.CANNY_HIGH_THRESHOLD)
        # Apply zone mask.
        edges_masked = cv2.bitwise_and(edges, edges, mask=zone_mask)
        # Use morphological closing to connect edges and form defect regions.
        # Kernel size can be configured.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(edges_masked, cv2.MORPH_CLOSE, kernel)
        _log_message(f"Canny defect detection for zone '{zone_name}' complete.")
        return closed_edges

    def _detect_defects_adaptive_thresh(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """Detects defects using adaptive thresholding."""
        # Log start.
        _log_message(f"Starting Adaptive Threshold defect detection for zone '{zone_name}'...")
        # Apply adaptive thresholding.
        # THRESH_BINARY_INV is used because defects are often darker or lighter than a varying background.
        adaptive_thresh_mask = cv2.adaptiveThreshold(image_gray, 255,
                                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY_INV, # Invert to get defects as white.
                                                     self.config.ADAPTIVE_THRESH_BLOCK_SIZE,
                                                     self.config.ADAPTIVE_THRESH_C)
        # Apply zone mask.
        defects_masked = cv2.bitwise_and(adaptive_thresh_mask, adaptive_thresh_mask, mask=zone_mask)
        _log_message(f"Adaptive Threshold defect detection for zone '{zone_name}' complete.")
        return defects_masked

    # --- Defect Combination and Analysis ---
    def _combine_defect_masks(self, defect_maps: Dict[str, Optional[np.ndarray]], image_shape: Tuple[int,int]) -> np.ndarray:
        """
        Combines defect masks from multiple methods using a voting or weighted scheme.
        Args:
            defect_maps: Dictionary where keys are method names and values are binary defect masks.
            image_shape: Tuple (height, width) for creating the combined map.
        Returns:
            A single binary mask representing confirmed defects.
        """
        # Start timer for combining defect masks.
        combine_start_time = _start_timer()
        # Log the start of combining defect masks.
        _log_message("Combining defect masks from multiple methods...")

        # Initialize a vote map with zeros, same shape as the input images.
        h, w = image_shape
        vote_map = np.zeros((h, w), dtype=np.float32)
        # Initialize total weight for normalization.
        total_weight_applied = 0.0

        # Iterate through each defect map from different methods.
        for method_name, mask in defect_maps.items():
            # Check if the mask is valid (not None).
            if mask is not None:
                # Get the confidence weight for the current method from config.
                weight = self.config.CONFIDENCE_WEIGHTS.get(method_name.split('_')[0], 0.5) # Default weight if method not in config.
                # Add weighted votes to the vote map.
                # Pixels in the mask (value 255) contribute 'weight' to the vote.
                vote_map[mask == 255] += weight
                # Add to total weight applied.
                total_weight_applied += weight


        # Determine the threshold for confirming a defect.
        # A defect is confirmed if its weighted vote exceeds a certain threshold.
        # This threshold can be a fixed number of methods or a percentage of total possible weighted votes.
        # Here, we use MIN_METHODS_FOR_CONFIRMED_DEFECT from config, but interpret it as a minimum weighted sum.
        # For simplicity, let's use a threshold relative to the max possible vote if all methods agreed.
        # Or, more simply, a fixed number of methods (e.g., if MIN_METHODS_FOR_CONFIRMED_DEFECT is 2,
        # a pixel needs to be detected by at least 2 (weighted) methods).
        # Let's use a threshold based on MIN_METHODS_FOR_CONFIRMED_DEFECT as a simple sum of unweighted positive detections.
        # For weighted sum, a threshold like 1.5 might mean "at least one strong method and one weaker, or two medium".
        # A more robust approach: normalize vote_map by max possible weighted sum if all methods detected a pixel.
        # For now, threshold is based on MIN_METHODS_FOR_CONFIRMED_DEFECT.
        # This implies that if a method has weight 1.0, it counts as 1 "vote".
        # If a method has weight 0.5, it counts as 0.5 "votes".
        confirmation_threshold = float(self.config.MIN_METHODS_FOR_CONFIRMED_DEFECT)

        # Create the final combined binary mask.
        # A pixel is part of a confirmed defect if its total weighted vote meets the threshold.
        combined_mask = np.where(vote_map >= confirmation_threshold, 255, 0).astype(np.uint8)

        # Log the duration of combining masks.
        _log_duration("Combine Defect Masks", combine_start_time, self.current_image_result)
        # Return the final combined mask.
        return combined_mask


    def _analyze_defect_contours(self, combined_defect_mask: np.ndarray,
                                 original_image_filename: str,
                                 all_defect_maps_by_method: Dict[str, Optional[np.ndarray]]) -> List[DefectInfo]:
        """
        Analyzes contours from the combined defect mask to extract defect properties.
        Args:
            combined_defect_mask: The final binary mask of confirmed defects.
            original_image_filename: Filename of the original image, for defect ID generation.
            all_defect_maps_by_method: Dictionary of defect maps from each individual method.
        Returns:
            A list of DefectInfo objects.
        """
        # Start timer for defect analysis.
        analysis_start_time = _start_timer()
        # Log the start of defect contour analysis.
        _log_message("Analyzing defect contours...")

        # List to store information about each detected defect.
        detected_defects: List[DefectInfo] = []
        # Find contours in the combined defect mask.
        # RETR_EXTERNAL retrieves only the outer contours.
        # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments.
        contours, hierarchy = cv2.findContours(combined_defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize defect ID counter for the current image.
        defect_counter = 0
        # Iterate through each detected contour.
        for contour in contours:
            # Calculate the area of the contour.
            area_px = cv2.contourArea(contour)
            # Filter out contours smaller than the minimum defect area specified in config.
            if area_px < self.config.MIN_DEFECT_AREA_PX:
                continue # Skip small contours.

            # Increment defect counter.
            defect_counter += 1
            # Create a unique defect ID.
            defect_id = f"{Path(original_image_filename).stem}_{defect_counter}"

            # Calculate moments to find the centroid.
            M = cv2.moments(contour)
            # Calculate centroid (cx, cy). Add epsilon to avoid division by zero if m00 is 0.
            cx = int(M['m10'] / (M['m00'] + 1e-5))
            cy = int(M['m01'] / (M['m00'] + 1e-5))

            # Get the bounding box (x, y, width, height).
            x, y, w, h = cv2.boundingRect(contour)
            # Calculate perimeter.
            perimeter_px = cv2.arcLength(contour, True)

            # Determine which zone this defect primarily belongs to.
            # This requires access to self.current_image_result.detected_zones.
            zone_name = "unknown" # Default zone name.
            if self.current_image_result and self.current_image_result.detected_zones:
                # Find the zone that contains the defect's centroid.
                for zn, z_info in self.current_image_result.detected_zones.items():
                    if z_info.mask is not None and z_info.mask[cy, cx] > 0:
                        zone_name = zn # Assign zone name.
                        break # Stop after finding the first matching zone.

            # Determine defect type (simple classification based on aspect ratio for now).
            # More sophisticated classification can be added (e.g., based on LEI vs DO2MR origin).
            aspect_ratio = float(w) / h if h > 0 else 0 # Calculate aspect ratio.
            defect_type = "Scratch" if aspect_ratio > 3.0 or aspect_ratio < 0.33 else "Region"

            # Identify which detection methods contributed to this defect.
            contributing_methods: List[str] = []
            # Iterate over individual method defect maps.
            for method_name, method_mask in all_defect_maps_by_method.items():
                if method_mask is not None:
                    # Check if the defect's centroid is within this method's detected region.
                    # A more robust check would be to see if the contour overlaps significantly.
                    # For simplicity, checking centroid:
                    if method_mask[cy, cx] > 0:
                        contributing_methods.append(method_name.split('_')[0]) # Get base method name.
            # Remove duplicates from contributing methods.
            contributing_methods = sorted(list(set(contributing_methods)))

            # Calculate confidence score based on the number of contributing methods.
            # Max possible methods is len(self.config.CONFIDENCE_WEIGHTS).
            # This is a simple confidence; can be refined with weights.
            num_total_methods = len(self.config.CONFIDENCE_WEIGHTS)
            confidence = len(contributing_methods) / num_total_methods if num_total_methods > 0 else 0.0
            confidence = min(confidence, 1.0) # Ensure confidence is not > 1.0.


            # Prepare DefectMeasurement objects.
            area_measurement = DefectMeasurement(value_px=area_px)
            perimeter_measurement = DefectMeasurement(value_px=perimeter_px)
            # For major/minor dimensions:
            # If scratch, use minAreaRect. For region, use equivalent diameter from area.
            major_dim_px: Optional[float] = None
            minor_dim_px: Optional[float] = None
            if defect_type == "Scratch":
                # Fit an oriented bounding box for scratches.
                if len(contour) >= 5: # minAreaRect needs at least 5 points.
                    rect = cv2.minAreaRect(contour) # ((center_x, center_y), (width, height), angle_of_rotation)
                    # Major dimension is the length, minor is the width.
                    major_dim_px = max(rect[1])
                    minor_dim_px = min(rect[1])
                else: # Fallback for very small contours.
                    major_dim_px = max(w,h)
                    minor_dim_px = min(w,h)
            else: # For "Region" defects.
                # Equivalent diameter.
                major_dim_px = np.sqrt(4 * area_px / np.pi)
                minor_dim_px = major_dim_px # For circular/blob-like, major and minor are same.

            major_dimension_measurement = DefectMeasurement(value_px=major_dim_px)
            minor_dimension_measurement = DefectMeasurement(value_px=minor_dim_px)


            # Convert to microns if possible.
            if self.pixels_per_micron and self.pixels_per_micron > 0:
                # Calculate area in square microns.
                area_measurement.value_um = area_px / (self.pixels_per_micron**2)
                # Calculate perimeter in microns.
                perimeter_measurement.value_um = perimeter_px / self.pixels_per_micron
                # Calculate major dimension in microns.
                major_dimension_measurement.value_um = major_dim_px / self.pixels_per_micron if major_dim_px is not None else None
                # Calculate minor dimension in microns.
                minor_dimension_measurement.value_um = minor_dim_px / self.pixels_per_micron if minor_dim_px is not None else None


            # Create DefectInfo object.
            defect_info = DefectInfo(
                defect_id=defect_counter, # Use the simple counter for now.
                zone_name=zone_name,
                defect_type=defect_type,
                centroid_px=(cx, cy),
                area=area_measurement,
                perimeter=perimeter_measurement,
                bounding_box_px=(x, y, w, h),
                major_dimension=major_dimension_measurement,
                minor_dimension=minor_dimension_measurement,
                confidence_score=confidence,
                detection_methods=contributing_methods,
                contour=contour # Store the contour itself.
            )
            # Add the defect information to the list.
            detected_defects.append(defect_info)

        # Log the number of defects found after analysis.
        _log_message(f"Analyzed {len(detected_defects)} defects from combined mask.")
        # Log the duration of defect analysis.
        _log_duration("Defect Contour Analysis", analysis_start_time, self.current_image_result)
        # Return the list of detected defects.
        return detected_defects

    # Ensure __init__ and other methods from Part 1 are here for context
    # Add the new methods defined above into the FiberInspector class structure.
    # For example:
    # class FiberInspector:
    #     def __init__(self, config: Optional[InspectorConfig] = None):
    #         # ... (from Part 1) ...
    #         pass
    #
    #     def _get_user_specifications(self):
    #         # ... (from Part 1) ...
    #         pass
    #
    #     def _initialize_zone_parameters(self):
    #         # ... (from Part 1) ...
    #         pass
    #
    #     def _get_image_paths_from_user(self) -> List[Path]:
    #         # ... (from Part 1) ...
    #         pass
    #
    #     def _load_single_image(self, image_path: Path) -> Optional[np.ndarray]:
    #         # ... (from Part 1) ...
    #         pass
    #
    #     # --- NEW METHODS FROM PART 2 ---
    #     def _preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
    #         # ... (implementation from above) ...
    #         pass
    #
    #     def _find_fiber_center_and_radius(self, processed_images: Dict[str, np.ndarray]) -> Optional[Tuple[Tuple[int, int], float]]:
    #         # ... (implementation from above) ...
    #         pass
    #
    #     def _calculate_pixels_per_micron(self, detected_cladding_radius_px: float) -> Optional[float]:
    #         # ... (implementation from above) ...
    #         pass
    #
    #     def _create_zone_masks(self, image_shape: Tuple[int, int], fiber_center_px: Tuple[int, int], detected_cladding_radius_px: float) -> Dict[str, DetectedZoneInfo]:
    #         # ... (implementation from above) ...
    #         pass
    #
    #     def _detect_region_defects_do2mr(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
    #         # ... (implementation from above) ...
    #         pass
    #
    #     def _detect_scratches_lei(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
    #         # ... (implementation from above) ...
    #         pass
    #
    #     def _detect_defects_canny(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
    #         # ... (implementation from above) ...
    #         pass
    #
    #     def _detect_defects_adaptive_thresh(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
    #         # ... (implementation from above) ...
    #         pass
    #
    #     def _combine_defect_masks(self, defect_maps: Dict[str, Optional[np.ndarray]], image_shape: Tuple[int,int]) -> np.ndarray:
    #         # ... (implementation from above) ...
    #         pass
    #
    #     def _analyze_defect_contours(self, combined_defect_mask: np.ndarray, original_image_filename: str, all_defect_maps_by_method: Dict[str, Optional[np.ndarray]]) -> List[DefectInfo]:
    #         # ... (implementation from above) ...
    #         pass
    #
    #     # ... (Part 3 methods will go here: reporting, batch processing, main orchestration) ...

# --- Placeholder for Main Execution (will be in Part 3) ---
if __name__ == "__main__":
    # This is where the script execution would start.
    # For Part 2, we are just defining the methods.
    # To test, you would instantiate FiberInspector from Part 1, then call these methods.
    _log_message("Processing and Detection Logic (Part 2) loaded.", level="DEBUG")

    # Example (conceptual - requires Part 1 context):
    # config = InspectorConfig()
    # inspector = FiberInspector(config) # Assuming FiberInspector is fully defined with Part 1
    # inspector._get_user_specifications()
    # image_paths = inspector._get_image_paths_from_user()
    #
    # if image_paths:
    #     test_image_path = image_paths[0]
    #     raw_image = inspector._load_single_image(test_image_path)
    #     if raw_image is not None:
    #         inspector.current_image_result = ImageResult(filename=test_image_path.name, timestamp=datetime.now(), fiber_specs_used=inspector.fiber_specs, operating_mode=inspector.operating_mode)
    #         processed_imgs = inspector._preprocess_image(raw_image)
    #         center_radius = inspector._find_fiber_center_and_radius(processed_imgs)
    #         if center_radius:
    #             center, radius_cladding = center_radius
    #             inspector._calculate_pixels_per_micron(radius_cladding)
    #             zones = inspector._create_zone_masks(raw_image.shape[:2], center, radius_cladding)
    #             inspector.current_image_result.detected_zones = zones
    #
    #             # Example: Detect defects in the 'cladding' zone
    #             if 'cladding' in zones and zones['cladding'].mask is not None:
    #                 gray_img_for_detection = processed_imgs.get('clahe_enhanced', processed_imgs['original_gray'])
    #                 do2mr_map = inspector._detect_region_defects_do2mr(gray_img_for_detection, zones['cladding'].mask, 'cladding')
    #                 lei_map = inspector._detect_scratches_lei(gray_img_for_detection, zones['cladding'].mask, 'cladding')
    #
    #                 if do2mr_map is not None: cv2.imwrite("do2mr_cladding_test.png", do2mr_map)
    #                 if lei_map is not None: cv2.imwrite("lei_cladding_test.png", lei_map)
    #
    #                 # Combine and analyze
    #                 defect_maps_for_combo = {"do2mr": do2mr_map, "lei": lei_map}
    #                 combined = inspector._combine_defect_masks(defect_maps_for_combo, raw_image.shape[:2])
    #                 if combined is not None: cv2.imwrite("combined_defects_test.png", combined)
    #                 defects_found = inspector._analyze_defect_contours(combined, test_image_path.name, defect_maps_for_combo)
    #                 _log_message(f"Found {len(defects_found)} defects in test.")

    pass
