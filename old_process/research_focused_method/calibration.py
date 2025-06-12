#!/usr/bin/env python3
# calibration.py

"""
Calibration Module
=================================
This is a multi-line string serving as a docstring, providing an overview of the module's purpose.
This module is responsible for calculating the pixel-to-micron conversion
It explains that the module calculates the pixel-to-micron ratio, a crucial step for physical measurements.
ratio (um_per_px) using a calibration target image (e.g., a stage micrometer).
It specifies that a calibration target, like a stage micrometer, is used for this process.
It can be run as a standalone script to generate/update 'calibration.json'.
It notes the script's ability to be run directly from the command line to create or update the calibration file.
The main inspection system will then use this calibration data.
It clarifies that other parts of the system will rely on the output of this script.
"""

import cv2 # Imports the OpenCV library, essential for all computer vision and image processing tasks.
import numpy as np # Imports the NumPy library, used for efficient numerical operations, especially with arrays.
import json # Imports the JSON library, used for reading and writing the calibration data to a .json file.
from pathlib import Path # Imports the Path object from the pathlib library for modern, object-oriented filesystem path manipulation.
from typing import Dict, Any, Optional, List, Tuple # Imports type hints for better code readability and static analysis.
import logging # Imports the logging module to record events, warnings, and errors during execution.
import argparse # Imports the argparse module to parse command-line arguments when run as a standalone script.

# --- Constants ---
# A section marker for defining constant values used throughout the script.
DEFAULT_CALIBRATION_FILE = "calibration.json" # Defines the default filename for saving calibration data, promoting consistency.

def _load_image_grayscale(image_path_str: str) -> Optional[np.ndarray]:
    """
    This docstring explains the function's purpose: loading an image and converting it to grayscale.
    Loads an image from the given path and converts it to grayscale.

    Args:
    This section of the docstring details the function's arguments.
        image_path_str: Path to the image file.

    Returns:
    This section of the docstring details what the function returns.
        The loaded grayscale image as a NumPy array, or None if loading fails.
    """
    image_path = Path(image_path_str) # Converts the input string path into a more robust Path object.
    if not image_path.exists(): # Checks if the file specified by the path actually exists on the filesystem.
        logging.error(f"Calibration image not found: {image_path}") # Logs an error message if the file does not exist.
        return None # Returns None to indicate that the image could not be loaded.

    image = cv2.imread(str(image_path)) # Reads the image file from the specified path using OpenCV.
    if image is None: # Checks if the image loading process failed (e.g., corrupted file).
        logging.error(f"Failed to load calibration image: {image_path}") # Logs an error message if OpenCV could not read the image.
        return None # Returns None to indicate the failure.

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converts the loaded image from Blue-Green-Red (BGR) color space to grayscale.
    logging.info(f"Calibration image '{image_path.name}' loaded successfully in grayscale.") # Logs a success message.
    return gray_image # Returns the resulting grayscale image as a NumPy array.

def _detect_calibration_features(gray_image: np.ndarray) -> List[Tuple[float, float]]:
    """
    This docstring explains the function's purpose: detecting features like dots in the calibration image.
    Detects features (e.g., dots) in the calibration image.
    This version uses SimpleBlobDetector, which is often good for dots.
    It clarifies the primary detection method used.
    Alternative methods like HoughCircles or contour finding can be added.
    It suggests other possible computer vision techniques that could be implemented.

    Args:
    This section details the function's arguments.
        gray_image: The grayscale calibration image.

    Returns:
    This section details what the function returns.
        A list of (x, y) coordinates for the centroids of detected features.
    """
    # --- Method 1: SimpleBlobDetector ---
    # A comment indicating the start of the primary feature detection method.
    # Setup SimpleBlobDetector parameters.
    # A comment explaining the next block of code sets up the detector's configuration.
    params = cv2.SimpleBlobDetector_Params() # Initializes a parameter object for the SimpleBlobDetector.

    # Change thresholds
    # A comment grouping the threshold parameter settings.
    params.minThreshold = 10 # Sets the minimum intensity value for a pixel to be considered part of a blob.
    params.maxThreshold = 200 # Sets the maximum intensity value for a pixel to be considered part of a blob.

    # Filter by Area.
    # A comment grouping the area-based filtering parameters.
    params.filterByArea = True # Enables filtering of blobs based on their pixel area.
    params.minArea = 20 # Sets the minimum area in pixels for a detected blob to be kept.
    params.maxArea = 5000 # Sets the maximum area in pixels, helping to filter out noise or very large objects.

    # Filter by Circularity
    # A comment grouping the circularity-based filtering parameters.
    params.filterByCircularity = True # Enables filtering of blobs based on how close their shape is to a perfect circle.
    params.minCircularity = 0.6 # Sets the minimum circularity value (1.0 is a perfect circle).

    # Filter by Convexity
    # A comment grouping the convexity-based filtering parameters.
    params.filterByConvexity = True # Enables filtering based on the ratio of blob area to its convex hull area.
    params.minConvexity = 0.80 # Sets the minimum convexity value for a detected blob.

    # Filter by Inertia
    # A comment grouping the inertia-based filtering parameters.
    params.filterByInertia = True # Enables filtering based on the blob's inertia ratio, which relates to its elongation.
    params.minInertiaRatio = 0.1 # Sets the minimum inertia ratio (1.0 for a circle, 0 for a line).

    detector = cv2.SimpleBlobDetector_create(params) # Creates an instance of the blob detector with the specified parameters.
    keypoints = detector.detect(gray_image) # Runs the detection algorithm on the grayscale image to find blobs (keypoints).
    
    centroids: List[Tuple[float, float]] = [] # Initializes an empty list to store the (x, y) coordinates of detected feature centers.
    if keypoints: # Checks if the detector found any keypoints.
        for kp in keypoints: # Iterates through each detected keypoint.
            centroids.append(kp.pt) # Appends the (x, y) coordinates of the keypoint's center to the centroids list.
        logging.info(f"Detected {len(centroids)} features using SimpleBlobDetector.") # Logs how many features were found.
    else: # This block executes if SimpleBlobDetector found no features.
        logging.warning("SimpleBlobDetector found no features. Trying HoughCircles as a fallback.") # Logs a warning and indicates a fallback method will be tried.
        
        # --- Method 2: HoughCircles (Fallback) ---
        # A comment indicating the start of the secondary (fallback) detection method.
        # Apply Gaussian blur to reduce noise, which helps HoughCircles.
        # Explains why blurring is being applied before the next step.
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2) # Applies a Gaussian blur to the image to smooth it and reduce noise.
        
        # Detect circles using HoughCircles. Parameters might need tuning.
        # A comment noting that the following parameters may require adjustment for different images.
        # dp: Inverse ratio of accumulator resolution.
        # A comment explaining the 'dp' parameter.
        # minDist: Minimum distance between the centers of detected circles.
        # A comment explaining the 'minDist' parameter.
        # param1: Higher threshold for the Canny edge detector.
        # A comment explaining the 'param1' parameter.
        # param2: Accumulator threshold for the circle centers at the detection stage.
        # A comment explaining the 'param2' parameter.
        # minRadius: Minimum circle radius.
        # A comment explaining the 'minRadius' parameter.
        # maxRadius: Maximum circle radius.
        # A comment explaining the 'maxRadius' parameter.
        circles = cv2.HoughCircles( # Calls the Hough Circle Transform function to find circles in the image.
            blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, # Passes the blurred image and detection parameters.
            param1=60, param2=30, minRadius=5, maxRadius=50 # Specifies thresholds and size constraints for circle detection.
        ) # Concludes the function call.

        if circles is not None: # Checks if the HoughCircles algorithm found any circles.
            circles_np = np.uint16(np.around(circles)) # Converts the circle coordinates and radii from floating-point to integers.
            for i in circles_np[0, :]: # Iterates through the array of detected circles.
                centroids.append((float(i[0]), float(i[1]))) # Appends the (x, y) center of each detected circle to the centroids list.
            logging.info(f"Detected {len(centroids)} features using HoughCircles fallback.") # Logs the number of features found by the fallback method.
        else: # Executes if the HoughCircles fallback also fails to find features.
            logging.warning("HoughCircles fallback also found no features.") # Logs a warning that both methods failed.
            
    if not centroids: # Checks if the centroids list is still empty after all detection attempts.
        logging.error("No calibration features detected in the image.") # Logs a critical error that no features could be found.
        raise ValueError("Could not detect any features in the calibration image.") # Raises a ValueError to halt the process, as calibration is impossible.

    return centroids # Returns the final list of detected feature centroids.

def calculate_um_per_px(
    centroids: List[Tuple[float, float]], # The function takes a list of feature coordinates.
    known_spacing_um: float # It also takes the known physical distance between features in micrometers.
) -> Optional[float]: # The function returns the calculated scale (a float) or None if it fails.
    """
    This docstring explains the function's purpose: calculating the microns-per-pixel ratio.
    Calculates the um_per_px ratio from a list of feature centroids and
    a known spacing between them.

    Args:
    This section details the function's arguments.
        centroids: A list of (x, y) coordinates of detected features.
        known_spacing_um: The known physical distance between adjacent features in microns.

    Returns:
    This section details what the function returns.
        The calculated um_per_px, or None if calculation is not possible.
    """
    if len(centroids) < 2: # Checks if there are fewer than two detected features.
        logging.error("Cannot calculate scale: at least two features are required.") # Logs an error because distance calculation requires at least two points.
        return None # Returns None to indicate that the calculation cannot proceed.

    distances_px: List[float] = [] # Initializes an empty list to store the calculated distances between features in pixels.
    # Calculate distances between all pairs of centroids.
    # A comment explaining the logic of the following nested loops.
    # This is a simple approach; for a grid, a more robust method would identify
    # horizontal/vertical neighbors and average those specific distances.
    # A comment suggesting a more advanced alternative for grid-like patterns.
    for i in range(len(centroids)): # The outer loop iterates through each centroid.
        for j in range(i + 1, len(centroids)): # The inner loop iterates through the remaining centroids to create unique pairs.
            dist = np.sqrt( # Calculates the Euclidean distance between the pair of centroids (i, j).
                (centroids[i][0] - centroids[j][0])**2 + # Calculates the squared difference in x-coordinates.
                (centroids[i][1] - centroids[j][1])**2 # Calculates the squared difference in y-coordinates.
            ) # Concludes the np.sqrt function call.
            distances_px.append(dist) # Adds the calculated pixel distance to the list.

    if not distances_px: # Checks if the list of distances is empty.
        logging.error("No distances could be calculated between features.") # Logs an error if no distances were computed.
        return None # Returns None as the scale cannot be determined.

    # --- Heuristic to find the characteristic spacing ---
    # A comment explaining the purpose of the next section is to find the most representative distance.
    # This assumes the 'known_spacing_um' corresponds to the most frequent
    # or smallest significant inter-feature distance.
    # A comment explaining the underlying assumption of this heuristic.
    # For a regular grid, this would be the distance to the nearest neighbors.
    # A comment providing a specific example for a regular grid.
    
    # Sort distances to find the smallest ones, which likely represent single unit spacings.
    # A comment explaining why the distances are being sorted.
    distances_px.sort() # Sorts the list of pixel distances in ascending order.
    
    # A more robust method would be to use a histogram or clustering if many features are present
    # to find the dominant spacing corresponding to known_spacing_um.
    # A comment suggesting more advanced alternative methods for improved robustness.
    # For simplicity here, if we have many points, we might average the smallest N distances
    # that are reasonably close to each other, assuming they represent the unit spacing.
    # A comment explaining the simpler approach taken in this code.
    
    # Consider distances that are likely to be the single 'known_spacing_um'.
    # Explains the goal of the logic that follows.
    # This is a heuristic: filter out very large distances if many points are detected,
    # as they might be diagonals or multiples of the unit spacing.
    # Details the heuristic being applied to filter out irrelevant distances.
    # If only a few points, all distances are considered more directly.
    # Explains how the logic adapts if few points are detected.
    
    # If many distances, try to isolate the fundamental spacing.
    # A comment explaining the condition for using the histogram-based approach.
    if len(distances_px) > 10: # A check to see if there are enough distances to warrant a statistical approach.
        # Use a histogram to find the most frequent distance bin.
        # A comment explaining the chosen statistical method.
        hist, bin_edges = np.histogram(distances_px, bins='auto') # Creates a histogram of the distances with automatically determined bins.
        if len(hist) > 0 and len(bin_edges) > 1 : # Checks if the histogram was successfully created.
            peak_bin_index = np.argmax(hist) # Finds the index of the histogram bin with the highest count (the mode).
            # Characteristic distance is the average of the edges of the peak bin.
            # Explains how the characteristic distance is derived from the histogram.
            characteristic_distance_px = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2.0 # Calculates the midpoint of the most frequent bin.
            logging.info(f"Characteristic pixel distance from histogram: {characteristic_distance_px:.2f} px") # Logs the result of the histogram method.
        else: # This block is a fallback if the histogram creation fails.
             # Fallback: use the median of smaller distances if histogram fails
             # Explains the fallback logic.
            characteristic_distance_px = np.median(distances_px[:max(1, len(distances_px)//2)]) # Calculates the median of the first half of the sorted distances.
            logging.warning(f"Histogram method for characteristic distance failed. Using median of smaller distances: {characteristic_distance_px:.2f} px") # Logs a warning that the fallback was used.

    elif distances_px: # This block executes if there are 10 or fewer distances.
        characteristic_distance_px = distances_px[0] # Takes the smallest calculated distance as the characteristic one.
        logging.info(f"Characteristic pixel distance (smallest): {characteristic_distance_px:.2f} px") # Logs the result of this simpler method.
    else: # This block executes if there are no distances at all (should be caught earlier, but for safety).
        logging.error("No valid pixel distances found for calibration.") # Logs a final error.
        return None # Returns None indicating failure.


    if characteristic_distance_px <= 1e-6: # Checks if the calculated distance is zero or extremely small to prevent division errors.
        logging.error(f"Calculated characteristic pixel distance is too small ({characteristic_distance_px:.2f} px). Calibration failed.") # Logs an error about the invalid distance.
        return None # Returns None to indicate failure.

    um_per_px_calc = known_spacing_um / characteristic_distance_px # Calculates the final microns-per-pixel ratio.
    
    # Validate the calculated scale is reasonable
    # A comment explaining the purpose of the following check.
    # Typical microscope scales range from 0.1 to 10 µm/pixel
    # A comment providing context on typical values for this ratio.
    if um_per_px_calc < 0.05 or um_per_px_calc > 20.0: # Checks if the calculated value falls outside a plausible range.
        logging.warning(f"Calculated scale {um_per_px_calc:.6f} µm/pixel seems unreasonable. Typical range is 0.1-10 µm/pixel.") # Logs a warning if the value is unusual.
        logging.warning("Please verify your calibration target and known spacing value.") # Advises the user to double-check their inputs.
    
    logging.info(f"Calculated scale: {um_per_px_calc:.6f} µm/pixel, based on known spacing {known_spacing_um} µm and characteristic distance {characteristic_distance_px:.2f} px.") # Logs the final successful result.
    return um_per_px_calc # Returns the calculated scale.

def save_calibration_data(
    data: Dict[str, Any], # The function takes a dictionary containing the data to save.
    file_path_str: str = DEFAULT_CALIBRATION_FILE # It takes an optional file path, defaulting to the defined constant.
) -> bool: # The function returns a boolean indicating success or failure.
    """
    This docstring explains the function's purpose: saving data to a JSON file.
    Saves the calibration data to a JSON file.

    Args:
    This section details the function's arguments.
        data: Dictionary containing the calibration data (e.g., {"um_per_px": value}).
        file_path_str: Path to save the JSON file.

    Returns:
    This section details what the function returns.
        True if saving was successful, False otherwise.
    """
    file_path = Path(file_path_str) # Converts the string path to a Path object.
    try: # Starts a try block to gracefully handle potential file I/O errors.
        with open(file_path, "w", encoding="utf-8") as f: # Opens the specified file in write mode ("w") with UTF-8 encoding.
            json.dump(data, f, indent=2) # Uses the json library to write the dictionary to the file, with an indent of 2 for readability.
        logging.info(f"Calibration data saved successfully to '{file_path}'.") # Logs a success message.
        return True # Returns True to indicate that the file was saved successfully.
    except IOError as e: # Catches any I/O errors that occur during the file writing process.
        logging.error(f"Failed to save calibration data to '{file_path}': {e}") # Logs the specific error that occurred.
        return False # Returns False to indicate that saving the file failed.

def load_calibration_data(
    file_path_str: str = DEFAULT_CALIBRATION_FILE # The function takes an optional file path, defaulting to the defined constant.
) -> Optional[Dict[str, Any]]: # The function returns a dictionary of data or None if it fails.
    """
    This docstring explains the function's purpose: loading data from a JSON file.
    Loads calibration data from a JSON file.

    Args:
    This section details the function's arguments.
        file_path_str: Path to the JSON calibration file.

    Returns:
    This section details what the function returns.
        A dictionary with the calibration data, or None if loading fails.
    """
    file_path = Path(file_path_str) # Converts the string path to a Path object.
    if not file_path.exists(): # Checks if the file exists at the given path.
        logging.warning(f"Calibration file '{file_path}' not found.") # Logs a warning if the file does not exist.
        return None # Returns None to indicate that the file could not be found.

    try: # Starts a try block to handle potential errors during file reading or JSON parsing.
        with open(file_path, "r", encoding="utf-8") as f: # Opens the file in read mode ("r").
            data = json.load(f) # Uses the json library to parse the file content into a Python dictionary.
        logging.info(f"Calibration data loaded successfully from '{file_path}'.") # Logs a success message.
        return data # Returns the loaded data dictionary.
    except (json.JSONDecodeError, IOError) as e: # Catches errors related to invalid JSON format or file reading issues.
        logging.error(f"Failed to load calibration data from '{file_path}': {e}") # Logs the specific error that occurred.
        return None # Returns None to indicate that loading failed.

def run_calibration_process(
    image_path_str: str, # The path to the image to be processed.
    known_spacing_um: float, # The known physical distance between features on the calibration target.
    output_file_str: str = DEFAULT_CALIBRATION_FILE, # The file path for the output JSON, with a default value.
    visualize: bool = False # A boolean flag to control whether to display visual feedback.
) -> Optional[float]: # The function returns the calculated scale or None on failure.
    """
    This docstring explains that this function orchestrates the entire calibration workflow.
    Runs the full calibration process: load image, detect features,
    calculate scale, and save it.

    Args:
    This section details the function's arguments.
        image_path_str: Path to the calibration target image.
        known_spacing_um: Known physical spacing between features in microns.
        output_file_str: Path to save the resulting calibration.json.
        visualize: If True, displays the image with detected features.

    Returns:
    This section details what the function returns.
        The calculated um_per_px value, or None if calibration fails.
    """
    logging.info(f"Starting calibration process for image: '{image_path_str}'") # Logs the start of the process.
    logging.info(f"Known feature spacing: {known_spacing_um} µm") # Logs the user-provided known spacing.

    gray_image = _load_image_grayscale(image_path_str) # Calls the function to load the image in grayscale.
    if gray_image is None: # Checks if the image loading failed.
        return None # Exits the function if no image could be loaded.

    try: # Starts a try block to handle the potential ValueError from feature detection.
        centroids = _detect_calibration_features(gray_image) # Calls the feature detection function.
    except ValueError as e: # Catches the error if no features are found.
        logging.error(f"Feature detection failed: {e}") # Logs the failure.
        return None # Exits the function if feature detection fails.
    
    if visualize: # Checks if the visualization flag is set to True.
        # Create a color image to draw on if the original was grayscale.
        # A comment explaining the purpose of the next line.
        vis_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR) if len(gray_image.shape) == 2 else gray_image.copy() # Creates a color copy for drawing.
        for (x, y) in centroids: # Iterates through each detected centroid.
            cv2.circle(vis_image, (int(x), int(y)), 10, (0, 0, 255), 2) # Draws a red circle around the centroid's location.
            cv2.putText(vis_image, f"({int(x)},{int(y)})", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1) # Adds text with the coordinates.
        
        cv2.imshow("Detected Calibration Features", vis_image) # Displays the image in a window titled "Detected Calibration Features".
        logging.info("Displaying image with detected features. Press any key to continue...") # Informs the user to press a key to proceed.
        cv2.waitKey(0) # Waits indefinitely for a key press from the user.
        cv2.destroyAllWindows() # Closes the image display window after a key is pressed.

    um_per_px = calculate_um_per_px(centroids, known_spacing_um) # Calls the function to calculate the scale.

    if um_per_px is not None: # Checks if the scale calculation was successful.
        calibration_data = { # Creates a dictionary to hold the results for saving.
            "um_per_px": um_per_px, # Stores the calculated scale.
            "source_image": Path(image_path_str).name, # Stores the name of the image used for calibration.
            "known_spacing_um": known_spacing_um, # Stores the known spacing provided by the user.
            "num_features_detected": len(centroids) # Stores the number of features that were detected.
        }
        if save_calibration_data(calibration_data, output_file_str): # Calls the function to save the data and checks for success.
            logging.info(f"Calibration successful. Scale: {um_per_px:.6f} µm/pixel.") # Logs the final success message.
            return um_per_px # Returns the calculated scale.
        else: # Executes if saving the data to the file failed.
            logging.error("Calibration calculation succeeded, but saving data failed.") # Logs the specific error condition.
            return None # Returns None to indicate the failure.
    else: # Executes if the scale calculation itself failed.
        logging.error("Calibration process failed: Could not calculate µm/pixel scale.") # Logs the calculation failure.
        return None # Returns None to indicate failure.

if __name__ == "__main__":
    # This block ensures the enclosed code only runs when the script is executed directly, not when imported as a module.
    # This block makes the script executable from the command line.
    # A comment clarifying the purpose of the __name__ == "__main__" block.
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s') # Configures the logging module to display INFO level messages and above.

    # --- Argument Parsing ---
    # A comment indicating the start of the command-line argument setup.
    parser = argparse.ArgumentParser(description=" Pixel-to-Micron Calibration Tool") # Creates an ArgumentParser object with a description.
    parser.add_argument( # Adds a new command-line argument.
        "image_path", # The name of the argument.
        type=str, # Specifies that the argument's value should be treated as a string.
        help="Path to the calibration target image (e.g., stage micrometer)." # Provides a help message for this argument.
    )
    parser.add_argument( # Adds a second command-line argument.
        "known_spacing", # The name of this argument.
        type=float, # Specifies that this argument's value should be treated as a floating-point number.
        help="Known physical distance between features on the target, in micrometers (µm)." # Provides a help message for this argument.
    )
    parser.add_argument( # Adds an optional command-line argument.
        "--output_file", # The name of the argument flag.
        type=str, # Specifies the argument's value is a string.
        default=DEFAULT_CALIBRATION_FILE, # Sets a default value if the argument is not provided.
        help=f"Path to save the calibration JSON data (default: {DEFAULT_CALIBRATION_FILE})." # Provides a help message.
    )
    parser.add_argument( # Adds another optional command-line argument that acts as a flag.
        "--visualize", # The name of the argument flag.
        action="store_true", # Specifies that if this flag is present, its value is True; otherwise, it's False.
        help="Display the image with detected calibration features." # Provides a help message.
    )

    args = parser.parse_args() # Parses the command-line arguments provided by the user.

    # Run the calibration process using provided arguments.
    # A comment explaining what the next function call does.
    calculated_scale = run_calibration_process( # Calls the main orchestration function.
        args.image_path, # Passes the image path from the parsed arguments.
        args.known_spacing, # Passes the known spacing from the parsed arguments.
        args.output_file, # Passes the output file path from the parsed arguments.
        args.visualize # Passes the visualization flag from the parsed arguments.
    )

    if calculated_scale is not None: # Checks if the calibration process completed successfully.
        print(f"\nCalibration completed successfully.") # Prints a success message to the console.
        print(f"Calculated µm/pixel: {calculated_scale:.6f}") # Prints the final calculated scale.
        print(f"Calibration data saved to: {args.output_file}") # Informs the user where the data was saved.
    else: # Executes if the calibration process failed.
        print(f"\nCalibration failed. Please check logs and input parameters.") # Prints a failure message to the console.

    # --- Example of loading the saved calibration data (for testing) ---
    # A comment indicating this section is for testing and is currently commented out.
    # print("\n--- Testing loading of saved calibration data ---")
    # A commented-out print statement for a section header.
    # loaded_data = load_calibration_data(args.output_file)
    # A commented-out call to the load function.
    # if loaded_data and "um_per_px" in loaded_data:
    # A commented-out check for successful loading.
    #     print(f"Successfully loaded um_per_px from '{args.output_file}': {loaded_data['um_per_px']:.6f}")
    # A commented-out print statement for success.
    # elif loaded_data:
    # A commented-out check for partial success.
    #     print(f"Loaded data from '{args.output_file}', but 'um_per_px' key is missing: {loaded_data}")
    # A commented-out print statement for partial success.
    # else:
    # A commented-out else block for complete failure.
    #     print(f"Could not load data from '{args.output_file}' after calibration attempt.")
    # A commented-out print statement for failure.