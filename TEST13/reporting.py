#!/usr/bin/env python3
# reporting.py

"""
Reporting Module
===============================
This is a docstring that provides a high-level overview of the module.
This module is responsible for generating all output reports for each processed image,
including annotated images, detailed CSV files for defects, and polar defect
distribution histograms.
"""

import cv2 # Imports the OpenCV library, essential for all image processing and drawing tasks like adding annotations.
import numpy as np # Imports the NumPy library, used for efficient numerical operations, especially with arrays for image data and calculations.
import matplotlib.pyplot as plt # Imports Matplotlib's pyplot interface, used for creating static, animated, and interactive visualizations like the polar histogram.
from matplotlib.projections.polar import PolarAxes # Imports the specific PolarAxes class from Matplotlib to allow for type hinting and clearer code when working with polar plots.
import pandas as pd # Imports the Pandas library, which is used for creating and manipulating data structures, particularly for generating CSV files from defect data.
from pathlib import Path # Imports the Path object from the standard library's pathlib module for modern, object-oriented handling of filesystem paths.
from typing import Dict, Any, Optional, List, Tuple, cast # Imports various typing hints from the standard library to improve code readability and allow for static analysis.
import logging # Imports the standard logging library to record events, warnings, and errors during the report generation process.
import datetime # Imports the standard datetime library to work with dates and times, used here for timestamping reports.


# This block attempts to import a configuration function from another module in the project.
try:
    # This line assumes a 'config_loader.py' file exists in the same directory or is in the Python path.
    from config_loader import get_config # Imports the 'get_config' function, which is expected to return the global configuration dictionary for the application.
# This 'except' block handles the case where the import fails, which is useful for standalone testing.
except ImportError:
    # This line logs a warning that the primary configuration loader could not be found.
    logging.warning("Could not import get_config from config_loader. Using dummy config for standalone testing.")
    # This defines a dummy function to substitute for the real 'get_config' during testing.
    def get_config() -> Dict[str, Any]: # The function is type-hinted to return a dictionary.
        """Returns a dummy configuration for standalone testing."""
        # This returns a hardcoded dictionary that mimics the structure of the actual configuration file.
        return { # A dictionary containing simplified configuration settings.
            "reporting": { # A nested dictionary for reporting-specific parameters.
                "annotated_image_dpi": 150, # Sets the resolution for saved images in dots per inch.
                "defect_label_font_scale": 0.4, # Sets the font size for text labels on defects.
                "defect_label_thickness": 1, # Sets the thickness of the font for defect labels.
                "pass_fail_stamp_font_scale": 1.5, # Sets a larger font size for the main PASS/FAIL status stamp.
                "pass_fail_stamp_thickness": 2, # Sets a thicker font for the PASS/FAIL stamp to make it stand out.
                "zone_outline_thickness": 1, # Sets the thickness of the lines used to draw zone boundaries.
                "defect_outline_thickness": 1, # Sets the thickness of the boxes drawn around defects.
                "display_timestamp_on_image": True, # A boolean flag to control whether a timestamp is added to the annotated image.
                "timestamp_format": "%Y-%m-%d %H:%M:%S" # A string that defines the format for the timestamp if it is displayed.
            },
            "zone_definitions_iec61300_3_35": { # A nested dictionary for zone definitions based on an industry standard.
                "single_mode_pc": [ # An example entry for a specific fiber type.
                    {"name": "Core", "color_bgr": [255,0,0]}, # Defines the 'Core' zone with its name and color in BGR format (Blue).
                    {"name": "Cladding", "color_bgr": [0,255,0]}, # Defines the 'Cladding' zone with its color (Green).
                    {"name": "Adhesive", "color_bgr": [0,255,255]}, # Defines the 'Adhesive' zone with its color (Yellow).
                    {"name": "Contact", "color_bgr": [255,0,255]}  # Defines the 'Contact' zone with its color (Magenta).
                ]
            },
            # This is a placeholder comment indicating where other dummy keys could be added for testing.
            # Add other keys as needed for standalone testing.
        }

# This comment indicates the start of the section containing the main report generation functions.
# --- Report Generation Functions ---

# This defines the function for creating the annotated image.
def generate_annotated_image(
    original_bgr_image: np.ndarray, # Parameter: The original image as a NumPy array in BGR color format.
    analysis_results: Dict[str, Any], # Parameter: A dictionary containing the results from the analysis module, like defects and status.
    localization_data: Dict[str, Any], # Parameter: A dictionary with data about the fiber's location (center, radii).
    zone_masks: Dict[str, np.ndarray], # Parameter: A dictionary where keys are zone names and values are their corresponding binary mask arrays.
    fiber_type_key: str, # Parameter: A string key to identify the current fiber type (e.g., "single_mode_pc").
    output_path: Path, # Parameter: A Path object representing the full file path to save the output image.
    report_timestamp: Optional[datetime.datetime] = None # Optional Parameter: A datetime object for the report's timestamp.
) -> bool: # The function is type-hinted to return a boolean, indicating success or failure.
    """
    This docstring explains the function's purpose, arguments, and return value.
    Generates and saves an annotated image showing zones, defects, and pass/fail status.

    Args:
        original_bgr_image: The original BGR image.
        analysis_results: Dictionary containing characterized defects and pass/fail status.
        localization_data: Dictionary with fiber localization info (centers, radii/ellipses).
        zone_masks: Dictionary of binary masks for each zone.
        fiber_type_key: Key for the fiber type (e.g., "single_mode_pc") for zone colors.
        output_path: Path object where the annotated image will be saved.
        report_timestamp: Optional datetime object for displaying on the image.

    Returns:
        True if the image was saved successfully, False otherwise.
    """
    annotated_image = original_bgr_image.copy() # Creates a writable copy of the input image to avoid modifying the original array.
    config = get_config() # Retrieves the application's configuration settings by calling the get_config function.
    report_cfg = config.get("reporting", {}) # Extracts the "reporting" sub-dictionary from the main config, defaulting to an empty dict if not found.
    zone_defs_all_types = config.get("zone_definitions_iec61300_3_35", {}) # Extracts the zone definitions for all fiber types.
    current_fiber_zone_defs = zone_defs_all_types.get(fiber_type_key, []) # Gets the specific zone definitions for the current fiber type being processed.

    # This comment indicates the start of the code block for drawing zone outlines.
    # --- Draw Zones ---
    # Creates a dictionary mapping zone names to their BGR color tuples for easy lookup.
    zone_color_map = {z["name"]: tuple(z["color_bgr"]) for z in current_fiber_zone_defs if "color_bgr" in z}
    # Retrieves the thickness for zone outlines from the config, with a default value of 1.
    zone_outline_thickness = report_cfg.get("zone_outline_thickness", 1) # Thickness for zone outlines.

    # This loop iterates over each zone name and its corresponding mask in the 'zone_masks' dictionary.
    for zone_name, zone_mask_np in zone_masks.items(): # Iterate through zone masks.
        # Retrieves the color for the current zone from the map, defaulting to gray if not found.
        color = zone_color_map.get(zone_name, (128, 128, 128)) # Default to gray if color not defined.
        # This OpenCV function finds the continuous outer boundaries of shapes in the binary zone mask.
        contours, _ = cv2.findContours(zone_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours.
        # This function draws the found contours onto the annotated image with the specified color and thickness.
        cv2.drawContours(annotated_image, contours, -1, color, zone_outline_thickness) # Draw zone contours.
        
        # This conditional block executes if any contours were found for the current zone.
        if contours: # If contours exist for labeling.
            # Finds the largest contour by area, which is assumed to be the primary boundary of the zone.
            largest_contour = max(contours, key=cv2.contourArea) # Get largest contour.
            # This line attempts to find a good position for the zone label by selecting the topmost point of the largest contour.
            # This is a simple heuristic that may need refinement for complex or oddly shaped zones.
            label_pos_candidate = tuple(largest_contour[largest_contour[:,:,1].argmin()][0]) # Top-most point.
            
            # This block adjusts the calculated label position to prevent it from being drawn off-screen.
            # Ensures the x-coordinate of the label is not too close to the left edge.
            label_x = max(5, label_pos_candidate[0] - 20) # Ensure not too far left
            # Ensures the y-coordinate is not too close to the top edge.
            label_y = max(10, label_pos_candidate[1] - 5) # Ensure not too far up
            # Checks if the label is too close to the bottom edge and adjusts it upwards if necessary.
            if label_y > annotated_image.shape[0] - 10: # If too close to bottom
                 label_y = annotated_image.shape[0] - 10 # Moves the label position up to stay within the image boundary.

            # This OpenCV function draws the zone name text onto the annotated image at the calculated position.
            cv2.putText(annotated_image, zone_name, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, report_cfg.get("defect_label_font_scale", 0.4) * 0.9, # Uses a slightly smaller font scale for the zone label.
                        color, report_cfg.get("defect_label_thickness", 1), cv2.LINE_AA) # Add zone label.

    # This comment indicates the start of the code block for drawing defect annotations.
    # --- Draw Defects ---
    # Retrieves the list of characterized defects from the analysis results dictionary.
    defects_list = analysis_results.get("characterized_defects", []) # Get list of characterized defects.
    # Retrieves the font scale for defect labels from the reporting configuration.
    defect_font_scale = report_cfg.get("defect_label_font_scale", 0.4) # Font scale for defect labels.
    # Retrieves the line thickness for drawing defect outlines from the configuration.
    defect_line_thickness = report_cfg.get("defect_outline_thickness", 1) # Thickness for defect outlines.

    # This loop iterates through each defect found in the analysis.
    for defect in defects_list: # Iterate through defects.
        # Retrieves the classification of the defect (e.g., "Scratch", "Pit/Dig").
        classification = defect.get("classification", "Unknown") # Get defect classification.
        # Sets the color for the defect outline based on its classification (Magenta for Scratch, Orange for others).
        defect_color = (255, 0, 255) if classification == "Scratch" else (0, 165, 255) # Magenta for Scratch, Orange for Pit/Dig.

        # Retrieves the defect's contour points, which define its precise shape.
        contour_pts = defect.get("contour_points_px", None)
        # This block executes if contour points are available for the defect.
        if contour_pts:
            # Converts the list of contour points into a NumPy array with the correct shape for OpenCV functions.
            contour_np = np.array(contour_pts, dtype=np.int32).reshape(-1, 1, 2)
            # 'cv2.minAreaRect' requires at least 3 points to define a rectangle.
            if contour_np.shape[0] >= 3: # Check if there are enough points for minAreaRect
                # Calculates the minimum area bounding rectangle that encloses the defect contour, which can be rotated.
                rot_rect_params = cv2.minAreaRect(contour_np)
                # Calculates the four corner points of this rotated rectangle.
                box_points = cv2.boxPoints(rot_rect_params)
                # Converts the corner point coordinates to integers, as required by 'cv2.drawContours'.
                box_points_int = np.array(box_points, dtype=np.int32) # Use np.int32 for drawContours 
                # Draws the outline of the rotated bounding box on the image. This line previously had a potential error if types were wrong.
                cv2.drawContours(annotated_image, [box_points_int], 0, defect_color, defect_line_thickness) # Error here
            # This is a fallback for contours that are too small to form a rectangle (e.g., a line or a single point).
            elif contour_np.size > 0: # Fallback for contours with < 3 points but > 0 points (e.g., a line)
                                      # This comment explains the condition covers cases where the contour exists but has few points.
                # Calculates an upright (non-rotated) bounding rectangle for these small contours.
                x, y, w, h = cv2.boundingRect(contour_np) # Use boundingRect for these cases
                # Checks if the bounding rectangle has a valid width and height.
                if w > 0 and h > 0:
                    # Draws the upright rectangle on the image.
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), defect_color, defect_line_thickness)
                # If the contour consists of exactly two points, it draws a line between them.
                elif contour_np.shape[0] == 2: # Draw a line if it's two points
                    cv2.line(annotated_image, tuple(contour_np[0,0]), tuple(contour_np[1,0]), defect_color, defect_line_thickness)
                # If the contour is just a single point, it draws a small filled circle to mark it.
                elif contour_np.shape[0] == 1: # Draw a circle if it's one point
                    cv2.circle(annotated_image, tuple(contour_np[0,0]), defect_line_thickness, defect_color, -1) # Small circle
            # This 'else' block is a further fallback if the contour data is invalid or empty.
            else: # Fallback to bounding box from defect data if contour_np is effectively empty
                # Retrieves the pre-calculated bounding box dimensions directly from the defect dictionary.
                x, y, w, h = (
                    defect.get("bbox_x_px", 0), # x-coordinate of the top-left corner.
                    defect.get("bbox_y_px", 0), # y-coordinate of the top-left corner.
                    defect.get("bbox_w_px", 0), # Width of the bounding box.
                    defect.get("bbox_h_px", 0), # Height of the bounding box.
                )
                # Draws an upright rectangle only if the width and height are valid (greater than 0).
                if w > 0 and h > 0: # Only draw if width and height are valid
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), defect_color, defect_line_thickness)
        # This 'else' block is the ultimate fallback if 'contour_points_px' was not provided in the defect data at all.
        else: # Fallback to bounding box if contour_points_px is not present
            # Retrieves the pre-calculated bounding box dimensions directly from the defect dictionary.
            x, y, w, h = (
                defect.get("bbox_x_px", 0), # x-coordinate of the top-left corner.
                defect.get("bbox_y_px", 0), # y-coordinate of the top-left corner.
                defect.get("bbox_w_px", 0), # Width of the bounding box.
                defect.get("bbox_h_px", 0), # Height of the bounding box.
            )
            # Draws an upright rectangle only if the width and height are valid.
            if w > 0 and h > 0: # Only draw if width and height are valid
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), defect_color, defect_line_thickness)

        # This block of code prepares and draws the text label for each defect.
        # Retrieves the unique ID for the defect.
        defect_id = defect.get("defect_id", "N/A") # Get defect ID.
        # Initializes an empty string for the defect's primary dimension.
        primary_dim_str = "" # Initialize primary dimension string.
        # This series of 'if' statements checks for different measures of size (length in µm, diameter in µm, length in pixels) and formats the first one it finds.
        if defect.get("length_um") is not None: # More robust check
            primary_dim_str = f"{defect['length_um']:.1f}µm" # Formats length in micrometers to one decimal place.
        elif defect.get("effective_diameter_um") is not None: # More robust check
            primary_dim_str = f"{defect['effective_diameter_um']:.1f}µm" # Formats diameter in micrometers.
        elif defect.get("length_px") is not None: # More robust check
            primary_dim_str = f"{defect['length_px']:.0f}px" # Formats length in pixels as an integer.
        
        # Constructs the final label text, e.g., "1:Scr,22.0µm". It uses the last part of the ID, the first 3 letters of the classification, and the size.
        label_text = f"{defect_id.split('_')[-1]}:{classification[:3]},{primary_dim_str}" # Create label text.
        # Gets the x-coordinate for the label from the defect's bounding box.
        label_x = defect.get("bbox_x_px", 0) # Get label x position.
        # Gets the y-coordinate, placing it 5 pixels above the bounding box.
        label_y = defect.get("bbox_y_px", 0) - 5 # Get label y position (slightly above bbox).
        # Adjusts the label's y-position to be below the defect if it's too close to the top of the image.
        if label_y < 10: label_y = defect.get("bbox_y_px",0) + defect.get("bbox_h_px",10) + 10 # Adjust if too close to top.

        # Draws the constructed label text onto the image using the specified font, scale, color, and thickness.
        cv2.putText(annotated_image, label_text, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, defect_font_scale, defect_color,
                    report_cfg.get("defect_label_thickness", 1), cv2.LINE_AA) # Add defect label.

    # This comment indicates the start of the code block for adding the overall PASS/FAIL status.
    # --- Add PASS/FAIL Stamp ---
    # Retrieves the overall inspection status from the analysis results.
    status = analysis_results.get("overall_status", "UNKNOWN") # Get overall status.
    # Sets the color for the stamp text: Green for PASS, Red for FAIL or any other status.
    status_color = (0, 255, 0) if status == "PASS" else (0, 0, 255) # Green for PASS, Red for FAIL/other.
    # Retrieves the font scale for the stamp from the configuration.
    stamp_font_scale = report_cfg.get("pass_fail_stamp_font_scale", 1.5) # Font scale for stamp.
    # Retrieves the font thickness for the stamp from the configuration.
    stamp_thickness = report_cfg.get("pass_fail_stamp_thickness", 2) # Thickness for stamp.
    
    # Calculates the pixel size (width, height) of the status text to position it correctly.
    text_size_status, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, stamp_font_scale, stamp_thickness) # Get text size.
    # Sets the x-position for the stamp near the left edge of the image.
    text_x_status = 10 # X position for stamp.
    # Sets the y-position for the stamp near the top edge, accounting for the text height.
    text_y_status = text_size_status[1] + 10 # Y position for stamp.
    # Draws the large PASS/FAIL status text in the top-left corner of the image.
    cv2.putText(annotated_image, status, (text_x_status, text_y_status),
                cv2.FONT_HERSHEY_SIMPLEX, stamp_font_scale, status_color, stamp_thickness, cv2.LINE_AA) # Add PASS/FAIL stamp.

    # This block adds other summary information below the PASS/FAIL stamp.
    # Extracts just the filename from the full image path stored in the analysis results.
    img_filename = Path(analysis_results.get("image_filename", "unknown.png")).name # Get image filename.
    # Retrieves the total number of defects found.
    total_defects = analysis_results.get("total_defect_count", 0) # Get total defect count.
    
    # Calculates the starting y-position for this additional info, placing it below the status stamp.
    info_text_y_start = text_y_status + text_size_status[1] + 15 # Starting Y for info text.
    
    # Draws the image filename on the annotated image.
    cv2.putText(annotated_image, f"File: {img_filename}", (10, info_text_y_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA) # Add filename.
    # Moves the y-position down for the next line of text.
    info_text_y_start += 20
    # Draws the total defect count on the annotated image.
    cv2.putText(annotated_image, f"Defects: {total_defects}", (10, info_text_y_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA) # Add defect count.
    # Moves the y-position down again.
    info_text_y_start += 20

    # This conditional block adds a generation timestamp to the image if enabled in the config and if a timestamp was provided.
    if report_timestamp and report_cfg.get("display_timestamp_on_image", False):
        # Retrieves the desired format for the timestamp string from the config.
        ts_format = report_cfg.get("timestamp_format", "%Y-%m-%d %H:%M:%S")
        # Formats the datetime object into a string according to the specified format.
        timestamp_str = report_timestamp.strftime(ts_format)
        # Draws the formatted timestamp string on the annotated image.
        cv2.putText(annotated_image, f"Generated: {timestamp_str}", (10, info_text_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)


    # This comment indicates the start of the final block for saving the image file.
    # --- Save the Annotated Image ---
    # This block attempts to write the final annotated image to the disk and handles potential errors.
    try:
        # This comment notes that while DPI is a configured value, cv2.imwrite does not always embed it in the file.
        # It's kept for potential future use with other libraries like Matplotlib.
        # dpi_val = report_cfg.get("annotated_image_dpi", 150) # Already have this, but unused by cv2.imwrite directly
        # This is the core command that saves the NumPy array representing the image to a file at the specified path.
        cv2.imwrite(str(output_path), annotated_image) # Save the annotated image.
        # Logs a success message to the console/log file.
        logging.info(f"Annotated image saved successfully to '{output_path}'.")
        # Returns True to indicate that the function completed successfully.
        return True # Return True on success.
    # This 'except' block catches any exceptions that occur during the file saving process.
    except Exception as e: # Handle errors during saving.
        # Logs a detailed error message including the file path and the specific exception.
        logging.error(f"Failed to save annotated image to '{output_path}': {e}")
        # Returns False to indicate that the function failed.
        return False # Return False on failure.

# This defines the function for creating the detailed CSV report of all defects.
def generate_defect_csv_report(
    analysis_results: Dict[str, Any], # Parameter: The dictionary containing the analysis results, including the list of defects.
    output_path: Path # Parameter: A Path object representing the full file path to save the output CSV.
) -> bool: # The function is type-hinted to return a boolean, indicating success or failure.
    """
    This docstring explains the function's purpose, arguments, and return value.
    Generates a CSV file listing all detected defects and their properties.

    Args:
        analysis_results: Dictionary containing characterized defects.
        output_path: Path object where the CSV report will be saved.

    Returns:
        True if the CSV was saved successfully, False otherwise.
    """
    # Retrieves the list of characterized defects from the analysis results dictionary.
    defects_list = analysis_results.get("characterized_defects", []) # Get list of characterized defects.
    # This block handles the special case where there are no defects to report.
    if not defects_list: # If no defects.
        # Logs a message indicating that no CSV will be generated because there are no defects.
        logging.info(f"No defects to report for {output_path.name}. CSV not generated.")
        # This 'try' block attempts to create an empty CSV file with only a header row.
        try: 
            # Defines the full list of all possible column headers for the CSV file to ensure consistency.
            cols = ["defect_id", "zone", "classification", "confidence_score",
                    "centroid_x_px", "centroid_y_px", "area_px", "length_px", "width_px",
                    "aspect_ratio_oriented", "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px",
                    "area_um2", "length_um", "width_um", "effective_diameter_um",
                    "rotated_rect_center_px", "rotated_rect_angle_deg"]
            # Creates an empty Pandas DataFrame using the defined column headers.
            df = pd.DataFrame([], columns=cols) 
            # Writes the empty DataFrame to a CSV file, which results in a file with only the header row.
            df.to_csv(output_path, index=False, encoding='utf-8') # Save empty CSV.
            # Logs a success message for saving the empty report.
            logging.info(f"Empty defect report CSV (with headers) saved to '{output_path}'.")
            # Returns True to indicate success.
            return True 
        # This 'except' block catches any errors that occur while trying to save the empty CSV.
        except Exception as e: 
            # Logs an error message.
            logging.error(f"Failed to save empty defect report to '{output_path}': {e}")
            # Returns False to indicate failure.
            return False 

    # This 'try' block handles the main case where there are defects to report.
    try:
        # Creates a Pandas DataFrame directly from the list of defect dictionaries.
        df = pd.DataFrame(defects_list) 
        
        # Defines the desired order and selection of columns for the final CSV report.
        report_columns = [ 
            "defect_id", "zone", "classification", "confidence_score",
            "centroid_x_px", "centroid_y_px",
            "length_um", "width_um", "effective_diameter_um", "area_um2",
            "length_px", "width_px", "area_px",
            "aspect_ratio_oriented",
            "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px",
            "rotated_rect_center_px", "rotated_rect_angle_deg"
            # "contour_points_px" # This comment indicates that contour points are too verbose and are intentionally excluded.
        ]
        # This list comprehension creates a final list of columns that exist in both the desired list and the actual DataFrame, preventing errors.
        final_columns = [col for col in report_columns if col in df.columns] 

        # Creates a new DataFrame containing only the selected columns in the desired order.
        df_report = df[final_columns] 

        # Writes the report DataFrame to a CSV file at the specified path, without including the DataFrame index.
        df_report.to_csv(output_path, index=False, encoding='utf-8') 
        # Logs a success message.
        logging.info(f"Defect CSV report saved successfully to '{output_path}'.")
        # Returns True to indicate success.
        return True 
    # This 'except' block catches any errors during the DataFrame creation or CSV writing process.
    except Exception as e: 
        # Logs a detailed error message.
        logging.error(f"Failed to save defect CSV report to '{output_path}': {e}")
        # Returns False to indicate failure.
        return False 

# This defines the function for creating the polar histogram plot.
def generate_polar_defect_histogram(
    analysis_results: Dict[str, Any], # Parameter: The dictionary with analysis results, including defects.
    localization_data: Dict[str, Any], # Parameter: The dictionary with fiber location info, especially the center point.
    zone_masks: Dict[str, np.ndarray], # Parameter: A dictionary of zone masks used to draw the zone boundaries on the plot.
    fiber_type_key: str, # Parameter: The string key for the current fiber type, used for getting zone colors.
    output_path: Path # Parameter: A Path object for the output location of the histogram PNG image.
) -> bool: # The function is type-hinted to return a boolean indicating success or failure.
    """
    This docstring explains the function's purpose, arguments, and return value.
    Generates and saves a polar histogram showing defect distribution.

    Args:
        analysis_results: Dictionary containing characterized defects.
        localization_data: Dictionary with fiber localization info (center is crucial).
        zone_masks: Dictionary of zone masks for plotting boundaries.
        fiber_type_key: Key for fiber type to get zone colors.
        output_path: Path object where the histogram PNG will be saved.

    Returns:
        True if the histogram was saved successfully, False otherwise.
    """
    # Retrieves the list of characterized defects from the analysis results.
    defects_list = analysis_results.get("characterized_defects", [])
    # Retrieves the pixel coordinates of the fiber's center.
    fiber_center_xy = localization_data.get("cladding_center_xy")

    # If there are no defects, there's nothing to plot, so the function exits successfully.
    if not defects_list:
        logging.info(f"No defects to plot for polar histogram for {output_path.name}.")
        return True

    # The fiber center is essential for converting defect coordinates to polar coordinates. If it's missing, the plot cannot be generated.
    if fiber_center_xy is None:
        logging.error("Cannot generate polar histogram: Fiber center not localized.")
        return False

    # Retrieves the global configuration.
    config = get_config()
    # Extracts the "reporting" sub-dictionary for parameters like DPI.
    report_cfg = config.get("reporting", {}) # Get reporting config for DPI
    # Extracts all zone definitions.
    zone_defs_all_types = config.get("zone_definitions_iec61300_3_35", {})
    # Gets the specific zone definitions for the current fiber type.
    current_fiber_zone_defs = zone_defs_all_types.get(fiber_type_key, [])
    # Creates a dictionary mapping zone names to their BGR colors.
    zone_color_map_bgr = {z["name"]: tuple(z["color_bgr"]) for z in current_fiber_zone_defs if "color_bgr" in z}

    # Unpacks the fiber center coordinates into separate x and y variables.
    center_x, center_y = fiber_center_xy
    # Initializes an empty list to store the angles of the defects in radians.
    angles_rad: List[float] = []
    # Initializes an empty list to store the radial distances of the defects in pixels.
    radii_px: List[float] = []
    # Initializes an empty list to store the colors for each defect point on the plot.
    defect_plot_colors_rgb: List[Tuple[float,float,float]] = []

    # This loop iterates through each defect to calculate its polar coordinates.
    for defect in defects_list:
        # Retrieves the defect's centroid x-coordinate.
        cx_px = defect.get("centroid_x_px")
        # Retrieves the defect's centroid y-coordinate.
        cy_px = defect.get("centroid_y_px")

        # If a defect is missing its centroid, it cannot be plotted, so it's skipped.
        if cx_px is None or cy_px is None: # Skip if centroid is missing
            logging.warning(f"Defect {defect.get('defect_id')} missing centroid, cannot plot in polar histogram.")
            continue # Skips to the next defect in the loop.

        # Calculates the defect's position relative to the fiber center (dx, dy).
        dx = cx_px - center_x
        dy = cy_px - center_y

        # Calculates the angle (theta) of the defect using arctan2, which handles all quadrants correctly.
        angle = np.arctan2(dy, dx)
        # Calculates the radial distance (r) of the defect using the Pythagorean theorem.
        radius = np.sqrt(dx**2 + dy**2)

        # Appends the calculated angle to the list of angles.
        angles_rad.append(angle)
        # Appends the calculated radius to the list of radii.
        radii_px.append(radius)

        # Retrieves the defect's classification.
        classification = defect.get("classification", "Unknown")
        # Assigns a BGR color based on the classification.
        bgr_color = (255, 0, 255) if classification == "Scratch" else (0, 165, 255) # Magenta for Scratch, Orange for Pit/Dig
        # Converts the BGR color to RGB and normalizes it to the [0, 1] range required by Matplotlib.
        rgb_color_normalized = (bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0)
        # Appends the normalized RGB color to the list of plot colors.
        defect_plot_colors_rgb.append(rgb_color_normalized)

    # This comment block indicates lines that were being checked or modified.
    # --- Start of lines to modify/check ---
    # Creates a new Matplotlib figure and a polar subplot. 'figsize' controls the output image size in inches.
    fig, ax_untyped = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    # This line explicitly type-hints the 'ax' variable as a PolarAxes object for better code completion and static analysis.
    ax: PolarAxes = ax_untyped # Explicitly type hint ax
    # --- End of lines to modify/check (ax definition) ---

    # This block executes if there are any defects to plot.
    if angles_rad and radii_px:
        # Creates a scatter plot of the defects on the polar axes, using the calculated angles, radii, and colors.
        ax.scatter(angles_rad, radii_px, c=defect_plot_colors_rgb, s=50, alpha=0.75, edgecolors='k')

    # Initializes a variable to track the maximum radius needed to display all zones.
    max_display_radius = 0
    # Initializes a set to keep track of zone labels that have been added to the legend to avoid duplicates.
    plotted_zone_labels = set() # To avoid duplicate labels in legend
    # This loop iterates through the zone masks to draw their boundaries on the plot. Sorting ensures a consistent legend order.
    for zone_name, zone_mask_np in sorted(zone_masks.items(), key=lambda item: item[0]): # Sort for consistent legend order
        # Checks if the zone mask actually contains any pixels.
        if np.sum(zone_mask_np) > 0:
            # Finds the coordinates of all pixels belonging to the zone.
            y_coords, x_coords = np.where(zone_mask_np > 0)
            # This block executes if coordinates were found.
            if y_coords.size > 0:
                # Calculates the distance of each zone pixel from the fiber center.
                distances_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                # Finds the maximum of these distances to determine the zone's outer radius.
                zone_outer_radius_px = np.max(distances_from_center) if distances_from_center.size > 0 else 0
                # Updates the maximum display radius if this zone is larger than the current maximum.
                max_display_radius = max(max_display_radius, zone_outer_radius_px)

                # Retrieves the BGR color for the zone, defaulting to gray.
                zone_bgr = zone_color_map_bgr.get(zone_name, (128,128,128))
                # Converts the BGR color to normalized RGB for Matplotlib.
                zone_rgb_normalized = (zone_bgr[2]/255.0, zone_bgr[1]/255.0, zone_bgr[0]/255.0)

                # Initializes the label for the legend as None.
                label_to_use = None
                # This block assigns a label for the legend only if it hasn't been plotted yet and the zone has a size.
                if zone_name not in plotted_zone_labels and zone_outer_radius_px > 0:
                    # Sets the label to the zone's name.
                    label_to_use = zone_name
                    # Adds the zone name to the set of plotted labels to prevent duplication.
                    plotted_zone_labels.add(zone_name)

                # Plots a dashed circle on the polar axes to represent the outer boundary of the zone.
                ax.plot(np.linspace(0, 2 * np.pi, 100), [zone_outer_radius_px] * 100,
                        color=zone_rgb_normalized, linestyle='--', label=label_to_use)

    # This comment indicates that the code below it was part of the original script.
# ... [previous code in the function] ...
    # Sets a default maximum radius for the plot's radial axis.
    current_r_max = 100 # Default r_max if no zones and no defect radii
    # If zone boundaries were drawn, set the plot's max radius to be slightly larger than the largest zone.
    if max_display_radius > 0: 
        current_r_max = max_display_radius * 1.1
    # If there are no zones but there are defects, set the max radius to be slightly larger than the furthest defect.
    elif radii_px: # Check if radii_px is not empty
        current_r_max = max(radii_px) * 1.2 

    # Sets the limit of the radial axis (the 'r-axis') of the polar plot.
    ax.set_rlim(0, current_r_max) 

    # This comment indicates a corrected line of code.
    # Corrected line:
    # This was the original, problematic line. 'get_rlim()' is not the correct method for setting ticks based on the current limits in some versions/contexts.
    # ax.set_rticks(np.linspace(0, ax.get_rlim()[1], 5)) # Original problematic line
    # This is the corrected line. For PolarAxes, get_ylim() correctly returns the radial limits (min, max) which can then be used to generate tick positions.
    ax.set_rticks(np.linspace(0, ax.get_ylim()[1], 5)) # Corrected: Use get_ylim() for PolarAxes

    # Sets the angular position (in degrees) where the radial axis labels (the numbers) are displayed.
    ax.set_rlabel_position(22.5) 
    # This comment indicates that the code below is the rest of the original function.
# ... [rest of the function] ...

    # Turns on the grid lines for the polar plot.
    ax.grid(True)
    # Sets the title for the plot, using the output filename stem for context. 'va' and 'pad' adjust the position.
    ax.set_title(f"Defect Distribution: {output_path.stem.replace('_histogram','')}", va='bottom', pad=20)
    # This block checks if there are any labels to display in the legend before creating it.
    if any(label is not None for label in ax.get_legend_handles_labels()[1]):
        # Creates a legend and positions it outside the top-right of the plot area for clarity.
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0)) # Adjusted legend position slightly


    # This 'try' block attempts to save the generated plot to a file.
    try:
        # Adjusts plot parameters for a tight layout, preventing labels from being cut off.
        plt.tight_layout()
        # Saves the figure to the specified output path, using the DPI setting from the configuration.
        fig.savefig(output_path, dpi=report_cfg.get("annotated_image_dpi", 150)) # Use DPI from config
        # Closes the figure to free up system memory, which is crucial when processing many images in a loop.
        plt.close(fig)
        # Logs a success message.
        logging.info(f"Polar defect histogram saved successfully to '{output_path}'.")
        # Returns True to indicate success.
        return True
    # This 'except' block catches any errors that occur during the saving process.
    except Exception as e:
        # Logs a detailed error message.
        logging.error(f"Failed to save polar defect histogram to '{output_path}': {e}")
        # Ensures the figure is closed even if saving fails, to prevent memory leaks.
        plt.close(fig)
        # Returns False to indicate failure.
        return False

# This defines an auxiliary function to add diameter annotations to an image.
def add_diameter_annotations(image: np.ndarray, localization_data: Dict[str, Any], 
                           um_per_px: Optional[float] = None) -> np.ndarray: # The function returns the modified image array.
    """
    This docstring explains the function's purpose, arguments, and return value.
    Add diameter measurements to the image for core and cladding.
    
    Args:
        image: Image to annotate
        localization_data: Dictionary with fiber localization info
        um_per_px: Microns per pixel conversion factor
        
    Returns:
        Annotated image
    """
    # Creates a copy of the input image to draw on.
    annotated = image.copy()
    
    # This block adds the core diameter annotation if the core radius is available.
    if 'core_radius_px' in localization_data and localization_data['core_radius_px'] > 0:
        # Calculates the core diameter in pixels.
        core_diameter_px = localization_data['core_radius_px'] * 2
        # Creates the initial text string with the pixel measurement.
        core_text = f"Core: {core_diameter_px:.1f}px"
        
        # If a pixel-to-micron conversion factor is available, it adds the micron measurement to the text.
        if um_per_px:
            # Converts the pixel diameter to micrometers.
            core_diameter_um = core_diameter_px * um_per_px
            # Appends the formatted micron measurement to the text string.
            core_text += f" ({core_diameter_um:.1f}µm)"
        
        # Draws the core diameter text onto the image in the top-left corner.
        cv2.putText(annotated, core_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # This block adds the cladding diameter annotation if the cladding radius is available.
    if 'cladding_radius_px' in localization_data and localization_data['cladding_radius_px'] > 0:
        # Calculates the cladding diameter in pixels.
        clad_diameter_px = localization_data['cladding_radius_px'] * 2
        # Creates the initial text string with the pixel measurement.
        clad_text = f"Cladding: {clad_diameter_px:.1f}px"
        
        # If a pixel-to-micron conversion factor is available, it adds the micron measurement to the text.
        if um_per_px:
            # Converts the pixel diameter to micrometers.
            clad_diameter_um = clad_diameter_px * um_per_px
            # Appends the formatted micron measurement to the text string.
            clad_text += f" ({clad_diameter_um:.1f}µm)"
        
        # Draws the cladding diameter text onto the image, positioned below the core diameter text.
        cv2.putText(annotated, clad_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Returns the image with the annotations added.
    return annotated

# This comment indicates the start of the main execution block, which is used for testing.
# --- Main function for testing this module (optional) ---
# The '__name__ == "__main__"' construct ensures this code only runs when the script is executed directly, not when imported as a module.
if __name__ == "__main__":
    # Configures the logging system to display messages of DEBUG level and higher, with a specific format including timestamp and module name.
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s') 

    # This comment indicates the start of the section where dummy data is created for testing purposes.
    # --- Dummy Data for Testing (mimicking outputs from analysis.py and image_processing.py) ---
    # Defines a base name for the test image files.
    dummy_image_base_name = "test_report_image"
    # Defines the file extension for the test image.
    dummy_image_ext = ".png"
    # This comment explains that a dummy image will be created for the test.
    # Create a dummy image if it doesn't exist.
    # This comment provides a note about handling file names during testing.
    # For testing, ensure a unique name if run multiple times or handle existing files.
    # Creates a 300x400 NumPy array filled with a light gray color, representing a blank image.
    img = np.full((300, 400, 3), (200, 200, 200), dtype=np.uint8) 
    # Draws a gray circle on the image to simulate the "Cladding".
    cv2.circle(img, (200,150), 80, (180,180,180), -1) # "Cladding"
    # Draws a smaller, darker gray circle inside the first one to simulate the "Core".
    cv2.circle(img, (200,150), 30, (150,150,150), -1) # "Core"
    
    # This block creates a temporary directory and saves the generated dummy image into it.
    # Defines the name for a temporary directory to store test images.
    temp_test_image_dir = Path("temp_test_images")
    # Creates the directory if it doesn't already exist.
    temp_test_image_dir.mkdir(exist_ok=True)
    # Creates a full Path object for the dummy image file.
    dummy_image_path_obj = temp_test_image_dir / f"{dummy_image_base_name}{dummy_image_ext}"
    # Saves the generated NumPy array as a PNG image file.
    cv2.imwrite(str(dummy_image_path_obj), img)
    # Logs a message indicating that the dummy image has been created.
    logging.info(f"Created dummy image at {dummy_image_path_obj}")


    # Reads the newly created dummy image back from the disk into a NumPy array.
    dummy_original_bgr = cv2.imread(str(dummy_image_path_obj)) 
    # This block checks if the image was loaded successfully and exits if it failed.
    if dummy_original_bgr is None: 
        logging.error(f"Failed to load dummy image for reporting test: {dummy_image_path_obj}")
        exit() # Exits the script if the test image cannot be loaded.

    # Gets the current date and time.
    current_timestamp = datetime.datetime.now()
    # Formats the timestamp into a string suitable for use in filenames.
    timestamp_str_file = current_timestamp.strftime("%Y%m%d_%H%M%S")


    # This large dictionary simulates the output that the 'analysis' module would produce.
    dummy_analysis_results = { 
        "image_filename": str(dummy_image_path_obj), # Includes the path to the image being "analyzed".
        "overall_status": "FAIL", # Sets a simulated overall inspection result.
        "failure_reasons": ["Zone 'Core': Scratch 'D1_S1' size (11.00µm) exceeds limit (3.0µm)"], # Provides a reason for the failure.
        "total_defect_count": 3, # Specifies the total number of defects found.
        "characterized_defects": [ # A list of dictionaries, where each dictionary represents one defect.
            {
                "defect_id": "D1_S1", "zone": "Core", "classification": "Scratch", "confidence_score": 0.95, # Properties of the first defect.
                "centroid_x_px": 190, "centroid_y_px": 140, "area_px": 50, "length_px": 22, "width_px": 2.5, # Pixel-based measurements.
                "aspect_ratio_oriented": 8.8, "bbox_x_px": 180, "bbox_y_px": 130, "bbox_w_px": 15, "bbox_h_px": 25, # Bounding box info.
                "rotated_rect_center_px": (190.0, 140.0), "rotated_rect_angle_deg": 45.0, # Rotated rectangle info.
                "contour_points_px": [[180,130],[195,130],[195,155],[180,155]], # Precise shape of the defect.
                "area_um2": 12.5, "length_um": 11.0, "width_um": 1.25 # Micron-based measurements.
            },
            { # This defect is designed to test the fallback logic for drawing contours with only two points.
                "defect_id": "D1_L1", "zone": "Cladding", "classification": "Scratch", "confidence_score": 0.88, # Properties of the second defect.
                "centroid_x_px": 230, "centroid_y_px": 170, "area_px": 0, "length_px": 10, "width_px": 0, 
                "aspect_ratio_oriented": 0, "bbox_x_px": 225, "bbox_y_px": 165, "bbox_w_px": 10, "bbox_h_px": 10,
                "rotated_rect_center_px": (230.0, 170.0), "rotated_rect_angle_deg": 0.0,
                "contour_points_px": [[225,165],[235,175]], # Only two points, which should be drawn as a line.
                "area_um2": 0, "length_um":5.0, "width_um":0
            },
            { # This defect is designed to test the fallback logic for defects with no contour points at all.
                "defect_id": "D2_P2", "zone": "Contact", "classification": "Pit/Dig", "confidence_score": 0.70, # Properties of the third defect.
                "centroid_x_px": 150, "centroid_y_px": 100, "area_px": 20, "length_px": 5, "width_px": 4, # Measurements.
                "bbox_x_px": 148, "bbox_y_px": 98, "bbox_w_px": 5, "bbox_h_px": 5, # Only bounding box info is provided.
                "area_um2": 5.0, "effective_diameter_um": 2.51
                # The absence of "contour_points_px" here is intentional for testing the fallback.
            }
        ]
    }
    # This dictionary simulates the output of the 'image_processing' module's localization step.
    dummy_localization_data = { 
        "cladding_center_xy": (200, 150), # The center of the fiber.
        "cladding_radius_px": 80.0, # The radius of the cladding.
        "core_center_xy": (200, 150), # The center of the core.
        "core_radius_px": 30.0 # The radius of the core.
    }
    # This string specifies the type of fiber being "inspected".
    dummy_fiber_type = "single_mode_pc" 
    # Initializes an empty dictionary to hold the generated zone masks.
    dummy_zone_masks_hist = {} 
    # Gets the height and width of the dummy image.
    _h, _w = dummy_original_bgr.shape[:2] 
    # Creates coordinate matrices that will be used to generate circular masks efficiently.
    _Y, _X = np.ogrid[:_h, :_w] 
    # Extracts the center coordinates from the dummy localization data.
    _center_x, _center_y = dummy_localization_data["cladding_center_xy"] 
    # Calculates the squared distance of every pixel from the center point. This is more efficient than calculating the square root for every pixel.
    _dist_sq = (_X - _center_x)**2 + (_Y - _center_y)**2 
    
    # Extracts the core and cladding radii from the dummy data.
    _core_r_px = dummy_localization_data["core_radius_px"] 
    _clad_r_px = dummy_localization_data["cladding_radius_px"] 
    # Creates a binary mask for the 'Core' zone by finding all pixels where the squared distance is less than the squared core radius.
    dummy_zone_masks_hist["Core"] = (_dist_sq < _core_r_px**2).astype(np.uint8) * 255 
    # Creates a binary mask for the 'Cladding' zone as an annulus (ring) between the core and cladding radii.
    dummy_zone_masks_hist["Cladding"] = ((_dist_sq >= _core_r_px**2) & (_dist_sq < _clad_r_px**2)).astype(np.uint8) * 255 
    # This comment explains that the 'Contact' zone is simplified for testing purposes.
    # Simplified Contact zone for testing: everything outside cladding up to a certain radius for visualization
    # Defines an outer radius for the contact zone for the test.
    _contact_outer_r_px = _clad_r_px * 1.5 
    # Creates a binary mask for the 'Contact' zone as an annulus outside the cladding.
    dummy_zone_masks_hist["Contact"] = ((_dist_sq >= _clad_r_px**2) & (_dist_sq < _contact_outer_r_px**2)).astype(np.uint8) * 255


    # Defines a Path object for the directory where test output files will be saved.
    test_output_dir = Path("test_reporting_output") 
    # Creates the output directory if it doesn't already exist.
    test_output_dir.mkdir(exist_ok=True) 

    # --- Test Case 1: Generate Annotated Image ---
    # Logs a header for the first test case.
    logging.info("\n--- Test Case 1: Generate Annotated Image ---")
    # Constructs the full output path for the annotated image, including the timestamp.
    annotated_img_path = test_output_dir / f"{dummy_image_base_name}_{timestamp_str_file}_annotated.png"
    # Calls the function to be tested with all the dummy data created above.
    success_annotated = generate_annotated_image( 
        dummy_original_bgr, dummy_analysis_results, dummy_localization_data, 
        dummy_zone_masks_hist, dummy_fiber_type, annotated_img_path,
        report_timestamp=current_timestamp # Passes the current timestamp to be displayed on the image.
    )
    # Logs the result (True or False) of the function call.
    logging.info(f"Annotated image generation success: {success_annotated}")

    # --- Test Case 2: Generate CSV Report ---
    # Logs a header for the second test case.
    logging.info("\n--- Test Case 2: Generate CSV Report ---")
    # Constructs the full output path for the CSV report file.
    csv_report_path = test_output_dir / f"{dummy_image_base_name}_{timestamp_str_file}_report.csv" 
    # Calls the CSV generation function with the dummy analysis results.
    success_csv = generate_defect_csv_report(dummy_analysis_results, csv_report_path) 
    # Logs the result of the function call.
    logging.info(f"CSV report generation success: {success_csv}")

    # --- Test Case 3: Generate Polar Defect Histogram ---
    # Logs a header for the third test case.
    logging.info("\n--- Test Case 3: Generate Polar Defect Histogram ---")
    # Constructs the full output path for the polar histogram image.
    histogram_path = test_output_dir / f"{dummy_image_base_name}_{timestamp_str_file}_histogram.png" 
    # Calls the histogram generation function with the necessary dummy data.
    success_hist = generate_polar_defect_histogram( 
        dummy_analysis_results, dummy_localization_data, dummy_zone_masks_hist, dummy_fiber_type, histogram_path
    )
    # Logs the result of the function call.
    logging.info(f"Polar histogram generation success: {success_hist}")
    
    # This block tests the reporting functions with a scenario where no defects were found.
    # Logs a header for the fourth test case.
    logging.info("\n--- Test Case 4: Reporting with NO defects ---")
    # Creates a copy of the dummy analysis results to modify for this specific test case.
    dummy_analysis_no_defects = dummy_analysis_results.copy() # Start with a copy
    # Modifies the copied data to simulate a no-defect scenario.
    dummy_analysis_no_defects["characterized_defects"] = [] # Empties the list of defects.
    dummy_analysis_no_defects["total_defect_count"] = 0 # Sets the defect count to zero.
    dummy_analysis_no_defects["overall_status"] = "PASS" # Changes the status to PASS.
    dummy_analysis_no_defects["failure_reasons"] = [] # Clears the list of failure reasons.
    
    # Creates a new timestamp string to make the output files for this test case unique.
    no_defect_timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Defines the output path for the empty CSV report.
    no_defect_csv_path = test_output_dir / f"{dummy_image_base_name}_{no_defect_timestamp_str}_no_defects_report.csv"
    # Calls the CSV function, which should generate an empty file with headers.
    generate_defect_csv_report(dummy_analysis_no_defects, no_defect_csv_path)

    # Defines the output path for the histogram (which should not be created or should be empty).
    no_defect_hist_path = test_output_dir / f"{dummy_image_base_name}_{no_defect_timestamp_str}_no_defects_histogram.png"
    # Calls the histogram function, which should gracefully handle having no defects to plot.
    generate_polar_defect_histogram(dummy_analysis_no_defects, dummy_localization_data, dummy_zone_masks_hist, dummy_fiber_type, no_defect_hist_path)
    
    # Defines the output path for the annotated image for the no-defect case.
    no_defect_annotated_img_path = test_output_dir / f"{dummy_image_base_name}_{no_defect_timestamp_str}_no_defects_annotated.png"
    # Calls the annotated image function, which should produce an image with only zones and a "PASS" stamp.
    generate_annotated_image(
        dummy_original_bgr, dummy_analysis_no_defects, dummy_localization_data,
        dummy_zone_masks_hist, dummy_fiber_type, no_defect_annotated_img_path,
        report_timestamp=datetime.datetime.now()
    )
    
    # This block tests how the polar histogram function handles defects that are missing centroid data.
    # Logs a header for the fifth test case.
    logging.info("\n--- Test Case 5: Polar histogram with NO defect centroids ---")
    # Creates another copy of the original dummy analysis results.
    dummy_analysis_no_centroids = dummy_analysis_results.copy()
    # Creates a deep copy of the defect list so that nested dictionaries can be safely modified.
    defects_no_centroids = [d.copy() for d in dummy_analysis_results["characterized_defects"]]
    # This loop iterates through the copied defects and removes their centroid information.
    for defect in defects_no_centroids:
        if "centroid_x_px" in defect: del defect["centroid_x_px"] # Deletes the x-centroid key.
        if "centroid_y_px" in defect: del defect["centroid_y_px"] # Deletes the y-centroid key.
    # Assigns the modified list of defects back to the analysis results for this test case.
    dummy_analysis_no_centroids["characterized_defects"] = defects_no_centroids
    
    # Creates a unique timestamp string for this test case's output file.
    no_centroids_timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Defines the output path for the histogram.
    no_centroids_hist_path = test_output_dir / f"{dummy_image_base_name}_{no_centroids_timestamp_str}_no_centroids_histogram.png"
    # Calls the histogram function, which should now log warnings and skip plotting the defects.
    generate_polar_defect_histogram(dummy_analysis_no_centroids, dummy_localization_data, dummy_zone_masks_hist, dummy_fiber_type, no_centroids_hist_path)


    # This final block cleans up the temporary files created during the test run.
    # Checks if the dummy image file still exists.
    if dummy_image_path_obj.exists():
        # Deletes the dummy image file.
        dummy_image_path_obj.unlink()
        # Logs a message indicating the file has been cleaned up.
        logging.info(f"Cleaned up dummy image: {dummy_image_path_obj}")
        # This 'try' block attempts to remove the temporary directory.
        try:
            # This command will only succeed if the directory is empty.
            temp_test_image_dir.rmdir() # Remove directory if empty
            # Logs that the directory was removed.
            logging.info(f"Cleaned up temporary image directory: {temp_test_image_dir}")
        # This 'except' block catches the error that occurs if the directory is not empty.
        except OSError:
            # Logs a message explaining that the directory was not removed.
            logging.info(f"Temporary image directory not empty, not removed: {temp_test_image_dir}")