#!/usr/bin/env python3
# calibration.py
""""
This module is responsible for calculating the pixel-to-micron conversion
ratio (um_per_px) using a calibration target image (e.g., a stage micrometer).
It can be run as a standalone script to generate/update 'calibration.json'.
The main inspection system will then use this calibration data.
"""

import cv2 # OpenCV for image processing.
import numpy as np # NumPy for numerical operations.
import json # Standard library for JSON parsing and serialization.
from pathlib import Path # Standard library for object-oriented path manipulation.
from typing import Dict, Any, Optional, List, Tuple # Standard library for type hinting.
import logging # Standard library for logging events.
import argparse # Standard library for parsing command-line arguments.

# --- Constants ---
DEFAULT_CALIBRATION_FILE = "calibration.json" # Default filename for saving calibration data.

def _load_image_grayscale(image_path_str: str) -> Optional[np.ndarray]:
    """
    Loads an image from the given path and converts it to grayscale.

    Args:
        image_path_str: Path to the image file.

    Returns:
        The loaded grayscale image as a NumPy array, or None if loading fails.
    """
    image_path = Path(image_path_str) # Convert string path to Path object.
    if not image_path.exists(): # Check if the image file exists.
        logging.error(f"Calibration image not found: {image_path}")
        return None # Return None if file not found.

    image = cv2.imread(str(image_path)) # Read the image using OpenCV.
    if image is None: # Check if image loading failed.
        logging.error(f"Failed to load calibration image: {image_path}")
        return None # Return None if loading failed.

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert BGR image to grayscale.
    logging.info(f"Calibration image '{image_path.name}' loaded successfully in grayscale.")
    return gray_image # Return the grayscale image.

def _detect_calibration_features(gray_image: np.ndarray) -> List[Tuple[float, float]]:
    """
    Detects features (e.g., dots) in the calibration image.
    This version uses SimpleBlobDetector, which is often good for dots.
    Alternative methods like HoughCircles or contour finding can be added.

    Args:
        gray_image: The grayscale calibration image.

    Returns:
        A list of (x, y) coordinates for the centroids of detected features.
    """
    # --- Method 1: SimpleBlobDetector ---
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params() # Initialize parameters object.

    # Change thresholds
    params.minThreshold = 10 # Minimum intensity threshold.
    params.maxThreshold = 200 # Maximum intensity threshold.

    # Filter by Area.
    params.filterByArea = True # Enable area filtering.
    params.minArea = 20 # Minimum blob area in pixels.
    params.maxArea = 5000 # Maximum blob area in pixels (adjust based on target).

    # Filter by Circularity
    params.filterByCircularity = True # Enable circularity filtering.
    params.minCircularity = 0.6 # Minimum circularity (1.0 is a perfect circle).

    # Filter by Convexity
    params.filterByConvexity = True # Enable convexity filtering.
    params.minConvexity = 0.80 # Minimum convexity.

    # Filter by Inertia
    params.filterByInertia = True # Enable inertia ratio filtering.
    params.minInertiaRatio = 0.1 # Minimum inertia ratio (1.0 for circle, 0 for line).

    detector = cv2.SimpleBlobDetector_create(params) # Create a blob detector with the parameters.
    keypoints = detector.detect(gray_image) # Detect blobs in the image.
    
    centroids: List[Tuple[float, float]] = [] # Initialize list to store centroids.
    if keypoints: # If keypoints (blobs) are found.
        for kp in keypoints: # Iterate through each keypoint.
            centroids.append(kp.pt) # Add the (x, y) coordinates of the keypoint to the list.
        logging.info(f"Detected {len(centroids)} features using SimpleBlobDetector.")
    else: # If no keypoints are found with SimpleBlobDetector.
        logging.warning("SimpleBlobDetector found no features. Trying HoughCircles as a fallback.")
        
        # --- Method 2: HoughCircles (Fallback) ---
        # Apply Gaussian blur to reduce noise, which helps HoughCircles.
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2) # Kernel size (9,9), sigma 2.
        
        # Detect circles using HoughCircles. Parameters might need tuning.
        # dp: Inverse ratio of accumulator resolution.
        # minDist: Minimum distance between the centers of detected circles.
        # param1: Higher threshold for the Canny edge detector.
        # param2: Accumulator threshold for the circle centers at the detection stage.
        # minRadius: Minimum circle radius.
        # maxRadius: Maximum circle radius.
        circles = cv2.HoughCircles(
            blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=60, param2=30, minRadius=5, maxRadius=50
        ) #

        if circles is not None: # If circles are found.
            circles_np = np.uint16(np.around(circles)) # Convert coordinates to integers.
            for i in circles_np[0, :]: # Iterate through detected circles.
                centroids.append((float(i[0]), float(i[1]))) # Add circle center as a feature centroid.
            logging.info(f"Detected {len(centroids)} features using HoughCircles fallback.")
        else: # If HoughCircles also fails.
            logging.warning("HoughCircles fallback also found no features.")
            
    if not centroids: # If no features were detected by any method.
        logging.error("No calibration features detected in the image.")
        raise ValueError("Could not detect any features in the calibration image.") # Raise error.

    return centroids # Return the list of detected feature centroids.

def calculate_um_per_px(
    centroids: List[Tuple[float, float]],
    known_spacing_um: float
) -> Optional[float]:
    """
    Calculates the um_per_px ratio from a list of feature centroids and
    a known spacing between them.

    Args:
        centroids: A list of (x, y) coordinates of detected features.
        known_spacing_um: The known physical distance between adjacent features in microns.

    Returns:
        The calculated um_per_px, or None if calculation is not possible.
    """
    if len(centroids) < 2: # Check if at least two features were detected.
        logging.error("Cannot calculate scale: at least two features are required.")
        return None # Return None if not enough features.

    distances_px: List[float] = [] # Initialize list to store pixel distances.
    # Calculate distances between all pairs of centroids.
    # This is a simple approach; for a grid, a more robust method would identify
    # horizontal/vertical neighbors and average those specific distances.
    for i in range(len(centroids)): # Iterate through centroids.
        for j in range(i + 1, len(centroids)): # Iterate through subsequent centroids.
            dist = np.sqrt(
                (centroids[i][0] - centroids[j][0])**2 +
                (centroids[i][1] - centroids[j][1])**2
            ) # Calculate Euclidean distance.
            distances_px.append(dist) # Add distance to list.

    if not distances_px: # Check if any distances were calculated.
        logging.error("No distances could be calculated between features.")
        return None # Return None.

    # --- Heuristic to find the characteristic spacing ---
    # This assumes the 'known_spacing_um' corresponds to the most frequent
    # or smallest significant inter-feature distance.
    # For a regular grid, this would be the distance to the nearest neighbors.
    
    # Sort distances to find the smallest ones, which likely represent single unit spacings.
    distances_px.sort()
    
    # A more robust method would be to use a histogram or clustering if many features are present
    # to find the dominant spacing corresponding to known_spacing_um.
    # For simplicity here, if we have many points, we might average the smallest N distances
    # that are reasonably close to each other, assuming they represent the unit spacing.
    
    # Consider distances that are likely to be the single 'known_spacing_um'.
    # This is a heuristic: filter out very large distances if many points are detected,
    # as they might be diagonals or multiples of the unit spacing.
    # If only a few points, all distances are considered more directly.
    
    # If many distances, try to isolate the fundamental spacing.
    if len(distances_px) > 10: # Arbitrary threshold indicating enough data for filtering.
        # Use a histogram to find the most frequent distance bin.
        hist, bin_edges = np.histogram(distances_px, bins='auto') # Automatic bin selection.
        if len(hist) > 0 and len(bin_edges) > 1 : # Check if histogram is valid.
            peak_bin_index = np.argmax(hist) # Find the bin with the most distances.
            # Characteristic distance is the average of the edges of the peak bin.
            characteristic_distance_px = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2.0
            logging.info(f"Characteristic pixel distance from histogram: {characteristic_distance_px:.2f} px")
        else: # If histogram is not valid.
             # Fallback: use the median of smaller distances if histogram fails
            characteristic_distance_px = np.median(distances_px[:max(1, len(distances_px)//2)])
            logging.warning(f"Histogram method for characteristic distance failed. Using median of smaller distances: {characteristic_distance_px:.2f} px")

    elif distances_px: # If fewer distances, use the smallest as characteristic (or average of a few smallest).
        characteristic_distance_px = distances_px[0] # Simplest: assumes the smallest distance is the unit.
        logging.info(f"Characteristic pixel distance (smallest): {characteristic_distance_px:.2f} px")
    else: # If no distances_px.
        logging.error("No valid pixel distances found for calibration.")
        return None # Return None.


    if characteristic_distance_px <= 1e-6: # Avoid division by zero or near-zero.
        logging.error(f"Calculated characteristic pixel distance is too small ({characteristic_distance_px:.2f} px). Calibration failed.")
        return None # Return None.

    um_per_px_calc = known_spacing_um / characteristic_distance_px # Calculate um/px.
    
    # Validate the calculated scale is reasonable
    # Typical microscope scales range from 0.1 to 10 µm/pixel
    if um_per_px_calc < 0.05 or um_per_px_calc > 20.0:
        logging.warning(f"Calculated scale {um_per_px_calc:.6f} µm/pixel seems unreasonable. Typical range is 0.1-10 µm/pixel.")
        logging.warning("Please verify your calibration target and known spacing value.")
    
    logging.info(f"Calculated scale: {um_per_px_calc:.6f} µm/pixel, based on known spacing {known_spacing_um} µm and characteristic distance {characteristic_distance_px:.2f} px.")
    return um_per_px_calc # Return the calculated scale.

def save_calibration_data(
    data: Dict[str, Any],
    file_path_str: str = DEFAULT_CALIBRATION_FILE
) -> bool:
    """
    Saves the calibration data to a JSON file.

    Args:
        data: Dictionary containing the calibration data (e.g., {"um_per_px": value}).
        file_path_str: Path to save the JSON file.

    Returns:
        True if saving was successful, False otherwise.
    """
    file_path = Path(file_path_str) # Convert string path to Path object.
    try:
        with open(file_path, "w", encoding="utf-8") as f: # Open file for writing.
            json.dump(data, f, indent=2) # Write data to JSON file with indentation.
        logging.info(f"Calibration data saved successfully to '{file_path}'.")
        return True # Return True on success.
    except IOError as e: # Handle I/O errors.
        logging.error(f"Failed to save calibration data to '{file_path}': {e}")
        return False # Return False on failure.

def load_calibration_data(
    file_path_str: str = DEFAULT_CALIBRATION_FILE
) -> Optional[Dict[str, Any]]:
    """
    Loads calibration data from a JSON file.

    Args:
        file_path_str: Path to the JSON calibration file.

    Returns:
        A dictionary with the calibration data, or None if loading fails.
    """
    file_path = Path(file_path_str) # Convert string path to Path object.
    if not file_path.exists(): # Check if file exists.
        logging.warning(f"Calibration file '{file_path}' not found.")
        return None # Return None if not found.

    try:
        with open(file_path, "r", encoding="utf-8") as f: # Open file for reading.
            data = json.load(f) # Load JSON data.
        logging.info(f"Calibration data loaded successfully from '{file_path}'.")
        return data # Return loaded data.
    except (json.JSONDecodeError, IOError) as e: # Handle JSON decoding or I/O errors.
        logging.error(f"Failed to load calibration data from '{file_path}': {e}")
        return None # Return None on failure.

def run_calibration_process(
    image_path_str: str,
    known_spacing_um: float,
    output_file_str: str = DEFAULT_CALIBRATION_FILE,
    visualize: bool = False
) -> Optional[float]:
    """
    Runs the full calibration process: load image, detect features,
    calculate scale, and save it.

    Args:
        image_path_str: Path to the calibration target image.
        known_spacing_um: Known physical spacing between features in microns.
        output_file_str: Path to save the resulting calibration.json.
        visualize: If True, displays the image with detected features.

    Returns:
        The calculated um_per_px value, or None if calibration fails.
    """
    logging.info(f"Starting calibration process for image: '{image_path_str}'")
    logging.info(f"Known feature spacing: {known_spacing_um} µm")

    gray_image = _load_image_grayscale(image_path_str) # Load the calibration image.
    if gray_image is None: # Check if loading failed.
        return None # Exit if image loading failed.

    try:
        centroids = _detect_calibration_features(gray_image) # Detect features.
    except ValueError as e: # Handle errors during feature detection.
        logging.error(f"Feature detection failed: {e}")
        return None # Exit if feature detection failed.
    
    if visualize: # If visualization is enabled.
        # Create a color image to draw on if the original was grayscale.
        vis_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR) if len(gray_image.shape) == 2 else gray_image.copy()
        for (x, y) in centroids: # Iterate through detected centroids.
            cv2.circle(vis_image, (int(x), int(y)), 10, (0, 0, 255), 2) # Draw a red circle around each centroid.
            cv2.putText(vis_image, f"({int(x)},{int(y)})", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1) # Add coordinates text.
        
        cv2.imshow("Detected Calibration Features", vis_image) # Display the image.
        logging.info("Displaying image with detected features. Press any key to continue...")
        cv2.waitKey(0) # Wait for a key press.
        cv2.destroyAllWindows() # Close the display window.

    um_per_px = calculate_um_per_px(centroids, known_spacing_um) # Calculate um/px.

    if um_per_px is not None: # If calculation was successful.
        calibration_data = { # Prepare data for saving.
            "um_per_px": um_per_px,
            "source_image": Path(image_path_str).name,
            "known_spacing_um": known_spacing_um,
            "num_features_detected": len(centroids)
        }
        if save_calibration_data(calibration_data, output_file_str): # Save the data.
            logging.info(f"Calibration successful. Scale: {um_per_px:.6f} µm/pixel.")
            return um_per_px # Return the calculated scale.
        else: # If saving failed.
            logging.error("Calibration calculation succeeded, but saving data failed.")
            return None # Return None.
    else: # If um/px calculation failed.
        logging.error("Calibration process failed: Could not calculate µm/pixel scale.")
        return None # Return None.

if __name__ == "__main__":
    # This block makes the script executable from the command line.
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s') # Configure basic logging.

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="D-Scope Blink: Pixel-to-Micron Calibration Tool") # Create argument parser.
    parser.add_argument( # Argument for calibration image path.
        "image_path",
        type=str,
        help="Path to the calibration target image (e.g., stage micrometer)."
    )
    parser.add_argument( # Argument for known feature spacing.
        "known_spacing",
        type=float,
        help="Known physical distance between features on the target, in micrometers (µm)."
    )
    parser.add_argument( # Argument for output calibration file path.
        "--output_file",
        type=str,
        default=DEFAULT_CALIBRATION_FILE,
        help=f"Path to save the calibration JSON data (default: {DEFAULT_CALIBRATION_FILE})."
    )
    parser.add_argument( # Argument to enable visualization.
        "--visualize",
        action="store_true", # Treat as a boolean flag.
        help="Display the image with detected calibration features."
    )

    args = parser.parse_args() # Parse the command-line arguments.

    # Run the calibration process using provided arguments.
    calculated_scale = run_calibration_process(
        args.image_path,
        args.known_spacing,
        args.output_file,
        args.visualize
    )

    if calculated_scale is not None: # If calibration was successful.
        print(f"\nCalibration completed successfully.")
        print(f"Calculated µm/pixel: {calculated_scale:.6f}")
        print(f"Calibration data saved to: {args.output_file}")
    else: # If calibration failed.
        print(f"\nCalibration failed. Please check logs and input parameters.")

    # --- Example of loading the saved calibration data (for testing) ---
    # print("\n--- Testing loading of saved calibration data ---")
    # loaded_data = load_calibration_data(args.output_file)
    # if loaded_data and "um_per_px" in loaded_data:
    #     print(f"Successfully loaded um_per_px from '{args.output_file}': {loaded_data['um_per_px']:.6f}")
    # elif loaded_data:
    #     print(f"Loaded data from '{args.output_file}', but 'um_per_px' key is missing: {loaded_data}")
    # else:
    #     print(f"Could not load data from '{args.output_file}' after calibration attempt.")
