#!/usr/bin/env python3
# main.py

"""
Main Orchestration Script
========================================
This is a multi-line docstring that serves as the documentation for the entire module.
It describes the script's role as the central entry point for the D-Scope Blink system.
It highlights that this script manages command-line arguments, controls the batch processing workflow,
and integrates all other specialized modules of the application.
"""
import cv2 # Imports the OpenCV library, aliased as 'cv2', for all computer vision and image processing tasks.
import argparse # Imports the standard library for parsing command-line arguments, allowing user interaction from the terminal.
import logging # Imports the standard library for logging events, which is crucial for tracking the application's execution and debugging.
import time # Imports the standard library for time-related functions, used here primarily for tracking performance of processing steps.
import datetime # Imports the standard library for working with date and time objects, used for creating timestamped output directories and log files.
from pathlib import Path # Imports the Path object from the pathlib library for modern, object-oriented manipulation of filesystem paths.
import sys # Imports the standard library for system-specific parameters and functions, used for interacting with the system (e.g., stderr, sys.exit).
import pandas as pd # Imports the pandas library, aliased as 'pd', which is used for creating and managing the final summary CSV report in a structured DataFrame.
from typing import Dict, Any, Optional, List # Imports specific types from the 'typing' module for type hinting, improving code readability and maintainability.

# --- D-Scope Blink Modules ---
# This comment block indicates the start of imports for custom modules specific to the D-Scope Blink application.
# These imports assume the modules are in the same directory or accessible via PYTHONPATH.
# This comment explains the dependency on the project's file structure or environment configuration.
try:
    # This 'try' block attempts to import optional or advanced modules that might not be installed in all environments.
    from advanced_visualization import InteractiveVisualizer
    # Imports the InteractiveVisualizer class, which provides advanced, interactive plotting capabilities.
    VISUALIZATION_AVAILABLE = True
    # If the import succeeds, this boolean flag is set to True, enabling the visualization feature.
except ImportError:
    # This 'except' block catches the ImportError if the 'advanced_visualization' module is not found.
    VISUALIZATION_AVAILABLE = False
    # If the import fails, the boolean flag is set to False, gracefully disabling the feature.
    
try:
    # This 'try' block attempts to import all the essential custom modules required for the core functionality of the application.
    from config_loader import load_config, get_processing_profile, get_zone_definitions # Imports functions from the config_loader module to handle configuration files.
    from calibration import load_calibration_data # Imports a function to load the pixel-to-micron calibration data.
    from image_processing import ( # Imports a set of functions from the image_processing module, which contains the core vision algorithms.
        load_and_preprocess_image,
        # Function to load an image from disk and apply initial preprocessing steps.
        locate_fiber_structure,
        # Function to find the core and cladding circles/ellipses in the image.
        generate_zone_masks,
        # Function to create masks for different inspection zones (core, cladding, etc.).
        detect_defects
        # Function that runs algorithms to find defects within a specific zone.
    )
    from analysis import characterize_and_classify_defects, apply_pass_fail_rules # Imports functions from the analysis module to measure defects and apply business logic.
    from reporting import generate_annotated_image, generate_defect_csv_report, generate_polar_defect_histogram # Imports functions from the reporting module to create all output files.
except ImportError as e: # This 'except' block catches an ImportError if any of the essential modules are missing.
    error_msg = (
        # A multi-line string is created to hold a detailed, user-friendly error message.
        f"[CRITICAL ERROR] D-Scope Blink could not start due to missing or problematic modules.\n"
        # The message clearly states the critical nature of the error.
        f"Details: {e}\n"
        # It includes the specific exception details for easier debugging.
        f"Please ensure all required Python modules (config_loader.py, calibration.py, "
        # The message lists the required custom modules.
        f"image_processing.py, analysis.py, reporting.py, and their dependencies like OpenCV, Pandas, Numpy) "
        # It also lists the required third-party libraries.
        f"are correctly installed and accessible in your Python environment (PYTHONPATH).\n"
        # It provides a hint on how to resolve the issue (checking the PYTHONPATH).
        f"Refer to the installation documentation for troubleshooting."
        # It points the user to the documentation for further help.
    )
    print(error_msg, file=sys.stderr)
    # The formatted error message is printed to the standard error stream.
    sys.exit(1) # The script terminates with a non-zero exit code, indicating that a critical error occurred.

# Import numpy for type hinting if not already (it's used in image_processing)
# This comment explains why numpy is being imported here, primarily for type hinting purposes.
import numpy as np
# Imports the numpy library, aliased as 'np', which is fundamental for numerical operations, especially with image arrays.

def setup_logging(log_level_str: str, log_to_console: bool, output_dir: Path) -> None:
    """
    This docstring explains the function's purpose: to configure the logging system for the application.
    Args:
        This section documents the function's parameters.
        log_level_str: The desired logging level as a string (e.g., "INFO", "DEBUG").
        log_to_console: Boolean indicating whether to log to the console.
        output_dir: The base directory where log files will be saved.
    """
    numeric_log_level = getattr(logging, log_level_str.upper(), logging.INFO) # Converts the string log level (e.g., "DEBUG") to its corresponding numeric value from the logging module (e.g., 10).
    
    log_format = '[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s' # Defines the format for log messages, including timestamp, level, module, line number, and the message itself.
    date_format = '%Y-%m-%d %H:%M:%S' # Defines the specific format for the timestamp used in the log messages.

    handlers: List[logging.Handler] = [] # Initializes an empty list to hold the configured logging handlers (e.g., file handler, console handler).

    # --- File Handler ---
    # This comment indicates the start of the file handler configuration section.
    # Create a unique log file name with a timestamp in the specified output directory.
    # This comment explains the logic for naming the log file to avoid overwriting previous logs.
    log_file_name = f"inspection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # Creates a unique log filename using the current date and time.
    log_file_path = output_dir / "logs" / log_file_name # Constructs the full path for the log file within a 'logs' subdirectory of the output directory.
    log_file_path.parent.mkdir(parents=True, exist_ok=True) # Creates the 'logs' subdirectory if it doesn't already exist.
    
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8') # Creates a file handler object that will write logs to the specified file path, using UTF-8 encoding.
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format)) # Assigns the predefined log format to the file handler.
    handlers.append(file_handler) # Adds the configured file handler to the list of active handlers.

    # --- Console Handler (Optional) ---
    # This comment indicates the start of the optional console handler configuration.
    if log_to_console: # Checks if the 'log_to_console' flag is True.
        console_handler = logging.StreamHandler(sys.stdout) # Creates a console handler that will write log messages to the standard output (the terminal).
        console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format)) # Assigns the predefined log format to the console handler.
        handlers.append(console_handler) # Adds the configured console handler to the list of active handlers.

    logging.basicConfig(level=numeric_log_level, handlers=handlers, force=True) # Configures the root logger with the specified level and the list of handlers. `force=True` allows re-configuration.
    logging.info(f"Logging configured. Level: {log_level_str}. Log file: {log_file_path}")
    # Logs an informational message to confirm that the logging setup was successful.

def process_single_image(
    image_path: Path,
    # The path to the specific image file being processed.
    output_dir_image: Path,
    # The dedicated directory where all results for this single image will be saved.
    profile_config: Dict[str, Any],
    # The configuration dictionary for the active processing profile (e.g., "deep_inspection").
    global_config: Dict[str, Any],
    # The full configuration dictionary containing all settings.
    calibration_um_per_px: Optional[float],
    # The generic microns-per-pixel scale loaded from a calibration file; can be None.
    user_core_dia_um: Optional[float],
    # The user-provided core diameter in microns; can be None.
    user_clad_dia_um: Optional[float],
    # The user-provided cladding diameter in microns; can be None.
    fiber_type_key: str
    # A string key identifying the fiber type (e.g., "single_mode_pc") for applying specific rules.
) -> Dict[str, Any]: # The function is type-hinted to always return a dictionary containing a summary of the results.
    """
    This docstring explains the function's purpose: to orchestrate the entire processing pipeline for a single fiber optic image.
    Args:
        This section documents the function's parameters.
        image_path: Path to the image file.
        output_dir_image: Directory to save results for this specific image.
        profile_config: The active processing profile configuration.
        global_config: The full global configuration dictionary.
        calibration_um_per_px: Calibrated um_per_px from file (can be None).
        user_core_dia_um: User-provided core diameter in microns (can be None).
        user_clad_dia_um: User-provided cladding diameter in microns (can be None).
        fiber_type_key: Key for the fiber type being processed.
    Returns:
        This section documents the function's return value.
        A dictionary containing summary results for the image.
    """
    image_start_time = time.perf_counter() # Records the starting time to measure the total processing duration for this image.
    logging.info(f"--- Processing image: {image_path.name} ---") # Logs a message indicating the start of processing for the specific image file.
    output_dir_image.mkdir(parents=True, exist_ok=True) # Ensures that the output directory for this specific image's results exists, creating it if necessary.

    # --- 1. Load and Preprocess Image ---
    # This comment block indicates the first major step in the pipeline.
    logging.info("Step 1: Loading and Preprocessing...") # Logs the current step to provide progress information.
    preprocess_results = load_and_preprocess_image(str(image_path), profile_config) # Calls the function to load the image and apply initial enhancements.
    if preprocess_results is None: # Checks if the preprocessing step failed and returned None.
        logging.error(f"Failed to load/preprocess image {image_path.name}. Skipping.") # Logs a critical error message.
        return {
            # Returns a dictionary indicating the failure, so the batch process can continue with the next image.
            "image_filename": image_path.name,
            # The name of the failed image.
            "pass_fail_status": "ERROR_LOAD_PREPROCESS",
            # A status code indicating the specific failure point.
            "processing_time_s": round(time.perf_counter() - image_start_time, 2),
            # The time spent before the failure occurred.
            "total_defect_count": 0,
            # Default value for defect count.
            "core_defect_count": 0,
            # Default value for core defect count.
            "cladding_defect_count": 0,
            # Default value for cladding defect count.
            "failure_reason_summary": "Load/preprocess failed"
            # A human-readable summary of the error.
        }
    original_bgr, original_gray, processed_image = preprocess_results
    # Unpacks the results from the successful preprocessing step into separate variables.

    # --- 2. Locate Fiber Structure (Cladding and Core) ---
    # This comment block indicates the second major step.
    logging.info("Step 2: Locating Fiber Structure...") # Logs the current step.
    localization_data = locate_fiber_structure(processed_image, profile_config, original_gray_image=original_gray)# Calls the function to find the fiber's geometric features.
    if localization_data is None or "cladding_center_xy" not in localization_data: # Checks if the localization failed to find the essential fiber center.
        logging.error(f"Failed to localize fiber structure in {image_path.name}. Skipping.")
        # Logs a critical error that the fiber could not be found.
        return {
            # Returns a dictionary indicating the localization failure.
            "image_filename": image_path.name,
            # The name of the failed image.
            "pass_fail_status": "ERROR_LOCALIZATION",
            # A status code indicating the failure point.
            "processing_time_s": round(time.perf_counter() - image_start_time, 2),
            # Time spent before failure.
            "total_defect_count": 0,
            # Default defect count.
            "core_defect_count": 0,
            # Default core defect count.
            "cladding_defect_count": 0,
            # Default cladding defect count.
            "failure_reason_summary": "Localization failed"
            # A human-readable summary of the error.
        }
    
    current_image_um_per_px = calibration_um_per_px # Sets the initial scale to the generic calibration value.
    if user_clad_dia_um is not None: # Checks if the user provided a known cladding diameter for this batch.
        detected_cladding_radius_px = localization_data.get("cladding_radius_px") # Retrieves the detected cladding radius in pixels from the localization results.
        if detected_cladding_radius_px and detected_cladding_radius_px > 0: # Checks if a valid circular radius was found.
            detected_cladding_diameter_px = detected_cladding_radius_px * 2.0 # Calculates the diameter in pixels.
            current_image_um_per_px = user_clad_dia_um / detected_cladding_diameter_px # Calculates a more accurate, image-specific scale (µm/px) using the user's value and the detected pixel size.
            logging.info(f"Using image-specific scale for {image_path.name}: {current_image_um_per_px:.4f} µm/px (user_clad_dia={user_clad_dia_um}µm, detected_clad_dia={detected_cladding_diameter_px:.1f}px).")
            # Logs that an image-specific scale is being used.
        elif localization_data.get("cladding_ellipse_params"): # If a circle wasn't found, checks if elliptical parameters are available (for angled fibers).
            ellipse_axes = localization_data["cladding_ellipse_params"][1] # Gets the minor and major axes of the detected ellipse.
            detected_cladding_diameter_px = ellipse_axes[1] # Assumes the major axis of the ellipse corresponds to the cladding diameter.
            if detected_cladding_diameter_px > 0: # Checks if the detected diameter is a valid positive number.
                current_image_um_per_px = user_clad_dia_um / detected_cladding_diameter_px # Calculates the image-specific scale using the ellipse's major axis.
                logging.info(f"Using image-specific scale (ellipse) for {image_path.name}: {current_image_um_per_px:.4f} µm/px (user_clad_dia={user_clad_dia_um}µm, detected_major_axis={detected_cladding_diameter_px:.1f}px).")
                # Logs that an ellipse-based scale is being used.
            else: # Handles the case where the ellipse major axis is not a valid number.
                 logging.warning(f"Detected cladding diameter (ellipse major axis) is zero for {image_path.name}. Cannot calculate image-specific scale. Falling back to generic calibration: {calibration_um_per_px}")
                 # Logs a warning that the calculation failed and the system is falling back to the less accurate generic scale.
        else: # Handles the case where neither a circle nor an ellipse could be reliably measured.
            logging.warning(f"Could not determine detected cladding diameter for {image_path.name} to calculate image-specific scale. Falling back to generic calibration: {calibration_um_per_px}")
            # Logs a warning about falling back to the generic scale.
    elif current_image_um_per_px: # If the user did not provide a diameter, this checks if a generic calibration scale is available.
        logging.info(f"Using generic calibration scale for {image_path.name}: {current_image_um_per_px:.4f} µm/px.")
        # Logs that the generic scale from the calibration file will be used.
    else: # This is the case where no scale information is available at all.
        logging.info(f"No µm/px scale available for {image_path.name}. Measurements will be in pixels.")
        # Informs the user that all reported measurements will be in pixels, not microns.


    analysis_summary = {
        # Initializes a dictionary to hold all analysis results for this image.
        "image_filename": image_path.name,
        # Stores the filename for reference.
        "cladding_diameter_px": None,
        # Placeholder for the detected cladding diameter in pixels.
        "core_diameter_px": None,
        # Placeholder for the detected core diameter in pixels.
        "characterized_defects": [],
        # An empty list to be filled with detailed information about each detected defect.
        "overall_status": "UNKNOWN",
        # The initial pass/fail status is set to "UNKNOWN".
        "total_defect_count": 0,
        # The initial total defect count.
        "failure_reasons": [],
        # An empty list to be filled with reasons for a "FAIL" status.
        "um_per_px_used": current_image_um_per_px
        # Stores the final µm/px scale that was used for this image's analysis.
    }
    
    # Add detected diameters to analysis summary
    # This comment indicates the following block updates the summary with geometric data.
    if localization_data:
        # Checks if localization data exists.
        cladding_diameter_px = localization_data.get("cladding_radius_px", 0) * 2
        # Calculates the cladding diameter from its radius, defaulting to 0 if not found.
        core_diameter_px = localization_data.get("core_radius_px", 0) * 2
        # Calculates the core diameter from its radius, defaulting to 0 if not found.
        
        logging.info(f"Detected diameters for {image_path.name}:")
        # Logs a header for the diameter information.
        logging.info(f"  - Cladding diameter: {cladding_diameter_px:.1f} pixels")
        # Logs the calculated cladding diameter in pixels.
        logging.info(f"  - Core diameter: {core_diameter_px:.1f} pixels")
        # Logs the calculated core diameter in pixels.
        
        analysis_summary['cladding_diameter_px'] = cladding_diameter_px
        # Stores the cladding diameter in the analysis summary dictionary.
        analysis_summary['core_diameter_px'] = core_diameter_px
        # Stores the core diameter in the analysis summary dictionary.

    # --- 3. Generate Zone Masks ---
    # This comment block indicates the third major step.
    zone_start_time = time.perf_counter()
    # Records the start time for the zone mask generation step to measure its performance.
    logging.info("Step 3: Generating Zone Masks...")
    # Logs the current step.
    try:
        # This 'try' block attempts to retrieve zone definitions which might fail if the fiber type is misconfigured.
        zone_definitions_for_type = get_zone_definitions(fiber_type_key)
        # Calls a function from the config_loader module to get the specific zone rules for the current fiber type.
    except ValueError as e: # Catches a ValueError if the fiber type key does not exist in the configuration.
        logging.error(f"Configuration error for fiber type '{fiber_type_key}': {e}. Cannot generate zone masks for {image_path.name}. Skipping.")
        # Logs a critical configuration error.
        return {
            # Returns a dictionary indicating the configuration failure.
            "image_filename": image_path.name,
            "pass_fail_status": "ERROR_CONFIG_ZONES",
            "processing_time_s": round(time.perf_counter() - image_start_time, 2),
            "total_defect_count": 0,
            "core_defect_count": 0,
            "cladding_defect_count": 0,
            "failure_reason_summary": f"Config error for fiber type '{fiber_type_key}': {e}"
        }

    zone_masks = generate_zone_masks( # Calls the function from image_processing to create the binary masks for each zone.
        processed_image.shape, localization_data, zone_definitions_for_type,
        # Passes the image shape, fiber location data, and zone definitions.
        current_image_um_per_px, user_core_dia_um, user_clad_dia_um
        # Also passes the scale and user-provided dimensions to ensure masks are dimensionally accurate.
    )
    zone_duration = time.perf_counter() - zone_start_time
    # Calculates the time taken for zone mask generation.
    logging.debug(f"Zone mask generation took {zone_duration:.3f}s")
    # Logs the duration of the step for performance analysis (at DEBUG level).
    
    if not zone_masks: # Checks if the zone mask generation failed and returned an empty result.
        logging.error(f"Failed to generate zone masks for {image_path.name}. Skipping.")
        # Logs a critical error.
        return {
            # Returns a dictionary indicating the failure.
            "image_filename": image_path.name,
            "pass_fail_status": "ERROR_ZONES",
            "processing_time_s": round(time.perf_counter() - image_start_time, 2),
            "total_defect_count": 0,
            "core_defect_count": 0,
            "cladding_defect_count": 0,
            "failure_reason_summary": "Zone mask generation failed"
        }

    # --- 4. Defect Detection in Each Zone ---
    # This comment block indicates the fourth and most crucial step of the pipeline.
    logging.info("Step 4: Detecting Defects in Zones...")
    # Logs the current step.
    all_zone_defect_masks: Dict[str, np.ndarray] = {}
    # Initializes a dictionary to store the resulting defect mask for each zone.
    combined_final_defect_mask = np.zeros_like(processed_image, dtype=np.uint8)
    # Creates an empty black image that will accumulate all detected defects from all zones.
    combined_confidence_map = np.zeros_like(processed_image, dtype=np.float32)
    # Creates an empty float image to accumulate confidence scores from different detection algorithms.
    global_algo_params = global_config.get("algorithm_parameters", {})
    # Retrieves a dictionary of global algorithm parameters from the main configuration.

    for zone_name, zone_mask_np in zone_masks.items():
        # Iterates through each zone (e.g., "Core", "Cladding") and its corresponding binary mask.
        if np.sum(zone_mask_np) == 0:
            # Checks if the zone mask is empty (i.e., the zone has zero area).
            logging.debug(f"Zone '{zone_name}' is empty. Skipping defect detection.")
            # Logs that this zone is being skipped.
            all_zone_defect_masks[zone_name] = np.zeros_like(processed_image, dtype=np.uint8)
            # Adds an empty defect mask for this zone to the results dictionary.
            continue
            # Skips to the next iteration of the loop.
        
        logging.debug(f"Detecting defects in zone: '{zone_name}'...")
        # Logs which zone is currently being analyzed for defects.
        # CORRECTED CALL to detect_defects: Added zone_name 
        # This comment, likely from development, notes a correction made to the function call.
        defects_in_zone_mask, zone_confidence_map = detect_defects(
            # Calls the core defect detection function from the image_processing module.
            processed_image, zone_mask_np, zone_name, profile_config, global_algo_params
            # Passes the image, the zone's mask, the zone's name, and algorithm configurations.
        )
        all_zone_defect_masks[zone_name] = defects_in_zone_mask
        # Stores the binary mask of defects found in the current zone.
        combined_final_defect_mask = cv2.bitwise_or(combined_final_defect_mask, defects_in_zone_mask)
        # Merges the defects from the current zone into the single combined mask using a bitwise OR operation.
        combined_confidence_map = np.maximum(combined_confidence_map, zone_confidence_map)
        # Merges the confidence map from the current zone into the combined map, keeping the highest confidence value for each pixel.
    
    # --- 5. Characterize, Classify Defects and Apply Pass/Fail ---
    # This comment block indicates the fifth step, where raw defect pixels are analyzed.
    logging.info("Step 5: Analyzing Defects and Applying Rules...")
    # Logs the current step.
    characterized_defects, overall_status, total_defect_count = characterize_and_classify_defects(
        # Calls a function from the analysis module to process the combined defect mask.
        combined_final_defect_mask, 
        # The mask containing all detected defects.
        zone_masks, 
        # The dictionary of all zone masks, used to determine which zone a defect is in.
        profile_config, 
        # The configuration for the current processing profile.
        current_image_um_per_px, 
        # The µm/px scale for converting pixel measurements to microns.
        image_path.name,
        # The name of the image file.
        confidence_map=combined_confidence_map
        # The combined confidence map, used to assign a confidence score to each defect.
    )
    
    # CORRECTED CALL to apply_pass_fail_rules: Changed zone_definitions_for_type to fiber_type_key 
    # This comment notes a correction made to the function call, improving how rules are applied.
    overall_status, failure_reasons = apply_pass_fail_rules(characterized_defects, fiber_type_key)
    # Calls a function from the analysis module to apply the pass/fail criteria based on the characterized defects and the specific fiber type.

    analysis_summary = { # Creates the final analysis summary dictionary for this image.
        "image_filename": image_path.name,
        # The image's filename.
        "characterized_defects": characterized_defects,
        # The list of all defects with their detailed properties (size, location, type, etc.).
        "overall_status": overall_status, 
        # The final "PASS" or "FAIL" status.
        "total_defect_count": total_defect_count, 
        # The total number of confirmed defects.
        "failure_reasons": failure_reasons, 
        # The list of specific rule violations if the status is "FAIL".
        "um_per_px_used": current_image_um_per_px
        # The scale used for measurements.
    }

    # --- 6. Generate Reports ---
    # This comment block indicates the sixth step, creating output files for the user.
    logging.info("Step 6: Generating Reports...")
    # Logs the current step.
    annotated_img_path = output_dir_image / f"{image_path.stem}_annotated.png" # Defines the full path for the annotated output image.
    generate_annotated_image( # Calls the function from the reporting module to create the visual report.
        original_bgr, analysis_summary, localization_data, zone_masks, fiber_type_key, annotated_img_path
        # Passes all necessary data to draw the zones, defects, and pass/fail status on the original image.
    )
    csv_report_path = output_dir_image / f"{image_path.stem}_report.csv" # Defines the full path for the detailed CSV report for this image.
    generate_defect_csv_report(analysis_summary, csv_report_path) # Calls the function to write the per-defect data to a CSV file.
    
    histogram_path = output_dir_image / f"{image_path.stem}_histogram.png" # Defines the full path for the polar histogram plot.
    generate_polar_defect_histogram( # Calls the function to create a polar plot showing the spatial distribution of defects.
        analysis_summary, localization_data, zone_masks, fiber_type_key, histogram_path
        # Passes the necessary data to plot defect locations relative to the fiber center.
    )
    
    processing_time_s = time.perf_counter() - image_start_time # Calculates the total processing time for this single image.
    logging.info(f"--- Finished processing {image_path.name}. Duration: {processing_time_s:.2f}s ---")
    # Logs a summary message with the total duration.

    # --- 7. Advanced Visualization (Optional) ---
    # This comment block indicates an optional final step.
    if VISUALIZATION_AVAILABLE and global_config.get("general_settings", {}).get("enable_visualization", False):
        # Checks if the visualization library is available AND if the feature is enabled in the configuration file.
        try:
            # This 'try' block gracefully handles any errors that might occur during visualization.
            visualizer = InteractiveVisualizer()
            # Creates an instance of the visualizer class.
            visualizer.show_inspection_results(
                # Calls the method to display the results interactively.
                original_bgr,
                # The original color image.
                all_zone_defect_masks,
                # The per-zone defect masks.
                zone_masks,
                # The zone boundary masks.
                analysis_summary,
                # The full analysis summary.
                interactive=False  # Sets the visualization to be non-blocking, so the batch process doesn't halt.
            )
        except Exception as e:
            # Catches any potential exceptions during the visualization process.
            logging.warning(f"Visualization failed for {image_path.name}: {e}")
            # Logs a warning that visualization failed but allows the program to continue.
            
    summary_for_batch = { # Creates a concise summary dictionary that will be used for the final batch report.
        "image_filename": image_path.name,
        # The filename of the processed image.
        "pass_fail_status": overall_status, 
        # The final "PASS" or "FAIL" status.
        "processing_time_s": round(processing_time_s, 2),
        # The total processing time for the image, rounded to two decimal places.
        "total_defect_count": total_defect_count, 
        # The total number of defects found.
        "core_defect_count": sum(1 for d in characterized_defects if d["zone"] == "Core"),
        # Counts the number of defects located specifically in the "Core" zone.
        "cladding_defect_count": sum(1 for d in characterized_defects if d["zone"] == "Cladding"),
        # Counts the number of defects located specifically in the "Cladding" zone.
        "failure_reason_summary": "; ".join(failure_reasons) if failure_reasons else "N/A" 
        # Joins the list of failure reasons into a single string, or shows "N/A" if the image passed.
    }
    return summary_for_batch # Returns the concise summary dictionary to the main batch processing loop.

# CORRECTED Function Definition: Changed args_namespace type hint to Any 
# This comment indicates a correction was made to the function's type hint for better compatibility.
def execute_inspection_run(args_namespace: Any) -> None:
    """
    This docstring explains the function's purpose: to house the core inspection logic that processes a batch of images based on provided arguments.
    It clarifies that this function contains the main processing flow of the application.
    """
    # --- Output Directory Setup ---
    # This comment block indicates the section for setting up output directories.
    base_output_dir = Path(args_namespace.output_dir) # Converts the output directory path from a string to a more robust Path object.
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') # Generates a timestamp string for creating a unique run folder.
    current_run_output_dir = base_output_dir / f"run_{run_timestamp}" # Creates a unique path for the current inspection run inside the base output directory.
    current_run_output_dir.mkdir(parents=True, exist_ok=True) # Creates the unique run directory, including any necessary parent directories.

    # --- Configuration and Logging Setup ---
    # This comment block indicates the setup for essential services like configuration loading and logging.
    try:
        # This 'try' block handles potential errors during the critical configuration loading step.
        config_file_path = str(args_namespace.config_file)
        # Gets the configuration file path from the arguments namespace.
        global_config = load_config(config_file_path) # Calls the function to load the main JSON configuration file into a dictionary.
    except (FileNotFoundError, ValueError) as e: # Catches errors if the config file is not found or is malformed.
        print(f"[CRITICAL] Failed to load configuration: {e}. Exiting.", file=sys.stderr)
        # Prints a critical error message to standard error.
        try:
            # This nested 'try' attempts to set up emergency logging to record the configuration error.
            fallback_log_dir = Path(".") / "error_logs"
            # Defines a fallback directory for error logs in the current working directory.
            fallback_log_dir.mkdir(parents=True, exist_ok=True)
            # Creates the fallback log directory.
            setup_logging("ERROR", True, fallback_log_dir)
            # Sets up logging to capture the critical error.
            logging.critical(f"Failed to load configuration: {e}. Exiting.")
            # Logs the fatal error.
        except Exception as log_e:
            # Catches errors that might even occur during the fallback logging setup.
            print(f"[CRITICAL] Logging setup failed during config error: {log_e}", file=sys.stderr)
            # Prints a final, desperate error message if logging itself fails.
        sys.exit(1) # Exits the application with an error code because it cannot proceed without a valid configuration.

    general_settings = global_config.get("general_settings", {}) # Safely retrieves the 'general_settings' dictionary from the global config, defaulting to an empty dict if not found.
    setup_logging( # Calls the function to initialize the application's logging system.
        general_settings.get("log_level", "INFO"),
        # Gets the log level from config, defaulting to "INFO".
        general_settings.get("log_to_console", True),
        # Gets the console logging flag from config, defaulting to True.
        current_run_output_dir 
        # Provides the unique output directory for this run to save the log file.
    )

    logging.info("D-Scope Blink: Inspection System Started.")
    # Logs that the system has started successfully.
    logging.info(f"Input Directory: {args_namespace.input_dir}")
    # Logs the input directory being used for this run.
    logging.info(f"Output Directory (this run): {current_run_output_dir}")
    # Logs the unique output directory created for this run's results.
    logging.info(f"Using Profile: {args_namespace.profile}")
    # Logs the processing profile (e.g., "fast_scan") being used.
    logging.info(f"Fiber Type Key for Rules: {args_namespace.fiber_type}")
    # Logs the fiber type key which determines which set of pass/fail rules to apply.
    
    # Add fiber type validation and correction
    # This comment indicates the start of a block to handle user-friendly aliases for fiber types.
    fiber_type_corrections = {
        # A dictionary mapping common user typos or abbreviations to the official fiber type keys used in the config.
        "single_mode": "single_mode_pc",
        "multi_mode": "multi_mode_pc",
        "sm": "single_mode_pc",
        "mm": "multi_mode_pc",
        "singlemode": "single_mode_pc",
        "multimode": "multi_mode_pc"
    }
    
    if args_namespace.fiber_type in fiber_type_corrections:
        # Checks if the user-provided fiber type is one of the aliases.
        corrected_type = fiber_type_corrections[args_namespace.fiber_type]
        # Retrieves the correct, official key from the dictionary.
        logging.warning(f"Correcting fiber type '{args_namespace.fiber_type}' to '{corrected_type}'")
        # Logs a warning to inform the user that their input was automatically corrected.
        args_namespace.fiber_type = corrected_type
        # Updates the fiber type in the arguments namespace to the corrected value.


    if args_namespace.core_dia_um: logging.info(f"User Provided Core Diameter: {args_namespace.core_dia_um} µm")
    # If the user provided a core diameter, it is logged.
    if args_namespace.clad_dia_um: logging.info(f"User Provided Cladding Diameter: {args_namespace.clad_dia_um} µm")
    # If the user provided a cladding diameter, it is logged.

    try:
        # This 'try' block attempts to load the specific processing profile from the configuration.
        active_profile_config = get_processing_profile(args_namespace.profile) # Calls the function to retrieve the configuration settings for the selected profile.
    except ValueError as e: # Catches a ValueError if the profile name doesn't exist in the configuration file.
        logging.critical(f"Failed to get processing profile '{args_namespace.profile}': {e}. Exiting.")
        # Logs a critical error that the application cannot proceed.
        sys.exit(1) # Exits the application with an error code.

    # --- Load Calibration Data ---
    # This comment block indicates the section for loading the µm/px scale.
    calibration_file_path = str(args_namespace.calibration_file)
    # Gets the calibration file path from the command-line arguments.
    calibration_data = load_calibration_data(calibration_file_path) # Calls the function to load the calibration JSON file.
    loaded_um_per_px: Optional[float] = None # Initializes the variable for the scale factor as None.
    if calibration_data: # Checks if the calibration data was loaded successfully.
        loaded_um_per_px = calibration_data.get("um_per_px") # Safely retrieves the 'um_per_px' value from the loaded data.
        if loaded_um_per_px: # Checks if the 'um_per_px' key was found and has a value.
            logging.info(f"Loaded µm/pixel scale from '{calibration_file_path}': {loaded_um_per_px:.4f} µm/px.")
            # Logs the successfully loaded generic scale factor.
        else: # This block executes if the calibration file is present but missing the required key.
            logging.warning(f"Calibration file '{calibration_file_path}' loaded, but 'um_per_px' key is missing or invalid.")
            # Logs a warning that the file is incomplete.
    else: # This block executes if the calibration file could not be loaded at all.
        logging.warning(f"No calibration data loaded from '{calibration_file_path}'. Measurements may be in pixels if user dimensions not provided.")
        # Logs a warning that measurements may not be in microns.

    # --- Image Discovery ---
    # This comment block indicates the section for finding all images to be processed.
    input_path = Path(args_namespace.input_dir) # Converts the input directory string to a Path object.
    if not input_path.is_dir(): # Validates that the provided input path is actually a directory.
        logging.critical(f"Input path '{input_path}' is not a valid directory. Exiting.")
        # Logs a critical error if the path is invalid.
        sys.exit(1) # Exits the application with an error code.

    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    # Defines a list of supported image file extensions.
    image_paths_to_process: List[Path] = [] # Initializes an empty list to store the paths of all images found.
    for ext in image_extensions: # Loops through each supported file extension.
        image_paths_to_process.extend(list(input_path.glob(f"*{ext}"))) # Finds all files with the current lowercase extension and adds them to the list.
        image_paths_to_process.extend(list(input_path.glob(f"*{ext.upper()}"))) # Finds all files with the current uppercase extension to be case-insensitive.
    
    image_paths_to_process = sorted(list(set(image_paths_to_process)))
    # Converts the list to a set to remove any duplicates, then back to a list and sorts it alphabetically.

    if not image_paths_to_process: # Checks if the list of images to process is empty.
        logging.info(f"No images found in directory: {input_path}")
        # Logs an informational message that no images were found.
        sys.exit(0) # Exits the program gracefully with a success code, as there is no work to do.

    logging.info(f"Found {len(image_paths_to_process)} images to process in '{input_path}'.")
    # Logs the total number of images that will be processed.

    # --- Batch Processing ---
    # This comment block indicates the start of the main processing loop for all images.
    batch_start_time = time.perf_counter() # Records the start time for the entire batch operation.
    all_image_summaries: List[Dict[str, Any]] = [] # Initializes an empty list to collect the summary dictionary from each processed image.

    for i, image_file_path in enumerate(image_paths_to_process): # Iterates through each found image path, with an index 'i'.
        logging.info(f"--- Starting image {i+1}/{len(image_paths_to_process)}: {image_file_path.name} ---")
        # Logs a progress message indicating which image is being processed now.
        image_specific_output_subdir = current_run_output_dir / image_file_path.stem
        # Creates a path for a dedicated subdirectory for the current image's results, named after the image file (without extension).
        current_image_processing_start_time = time.perf_counter() # Records the start time for processing this specific image, used for error case timing.
        
        try:
            # This 'try' block wraps the processing of a single image to catch any unexpected, critical errors.
            summary = process_single_image( # Calls the main function to process one image.
                image_file_path,
                # The path to the current image.
                image_specific_output_subdir,
                # The output directory for this image's results.
                active_profile_config,
                # The configuration for the active processing profile.
                global_config,
                # The entire application configuration.
                loaded_um_per_px,
                # The generic µm/px scale from the calibration file.
                args_namespace.core_dia_um,
                # The user-provided core diameter.
                args_namespace.clad_dia_um,
                # The user-provided cladding diameter.
                args_namespace.fiber_type
                # The key for the fiber type being processed.
            )
            # process_single_image is designed to always return a dict.
            # This comment explains the expected behavior of the called function.
            # Adding a safeguard for unforeseen None returns.
            # This comment explains the purpose of the following 'if' statement.
            if summary is None: 
                # A safeguard check in case the function unexpectedly returns None.
                 logging.error(f"Critical internal error: process_single_image returned None for {image_file_path.name}. This should not happen.")
                 # Logs a severe error as this indicates a bug in the 'process_single_image' function.
                 summary = {
                    # Creates a manual error summary dictionary to ensure the batch can continue.
                    "image_filename": image_file_path.name,
                    "pass_fail_status": "ERROR_UNEXPECTED_NONE_RETURN",
                    "processing_time_s": round(time.perf_counter() - current_image_processing_start_time, 2),
                    "total_defect_count": 0,
                    "core_defect_count": 0,
                    "cladding_defect_count": 0,
                    "failure_reason_summary": "Internal error: process_single_image returned None."
                }
            all_image_summaries.append(summary) 
            # Appends the returned summary dictionary (either from successful processing or an error) to the list for the final report.
        except Exception as e: # Catches any other unhandled exception during the image processing.
            logging.error(f"Unexpected error processing {image_file_path.name}: {e}", exc_info=True) # Logs the full error traceback for detailed debugging.
            failure_summary = { 
                # Creates a summary dictionary for this unexpected failure.
                "image_filename": image_file_path.name,
                "pass_fail_status": "ERROR_UNHANDLED_EXCEPTION",
                "processing_time_s": round(time.perf_counter() - current_image_processing_start_time, 2),
                "total_defect_count": 0,
                "core_defect_count": 0,
                "cladding_defect_count": 0,
                "failure_reason_summary": f"Unhandled exception: {str(e)}"
            }
            all_image_summaries.append(failure_summary)
            # Appends the failure summary to the list to ensure it's recorded in the final report.
            
    # --- Final Summary Report ---
    # This comment block indicates the final step of the batch run: creating a summary report.
    if all_image_summaries: # Checks if there are any image summaries to report on.
        summary_df = pd.DataFrame(all_image_summaries) # Creates a pandas DataFrame from the list of summary dictionaries.
        summary_report_path = current_run_output_dir / "batch_summary_report.csv" # Defines the path for the final summary CSV file.
        try:
            # This 'try' block handles potential errors while writing the final report file.
            summary_df.to_csv(summary_report_path, index=False, encoding='utf-8') # Saves the DataFrame to a CSV file, without the DataFrame index.
            logging.info(f"Batch summary report saved to: {summary_report_path}")
            # Logs the location of the final summary report.
        except Exception as e: # Catches any file I/O errors during the save operation.
            logging.error(f"Failed to save batch summary report: {e}")
            # Logs an error message if the report could not be saved.
    else: # This block executes if no summaries were generated at all.
        logging.warning("No image summaries were generated for the batch report.")
        # Logs a warning that the final report will be empty.

    batch_duration = time.perf_counter() - batch_start_time # Calculates the total duration of the entire batch process.
    logging.info(f"--- D-Scope Blink: Batch Processing Complete ---")
    # Logs a completion message for the entire run.
    logging.info(f"Total images processed: {len(image_paths_to_process)}")
    # Logs the total number of images processed.
    logging.info(f"Total batch duration: {batch_duration:.2f} seconds.")
    # Logs the total time taken for the batch.
    logging.info(f"All reports for this run saved in: {current_run_output_dir}")
    # Reminds the user where to find all the generated output files.

# CORRECTED Function Definition: Changed args_namespace type hint to Any 
# This comment notes a correction to the function's type hint.
def main_with_args(args_namespace: Any) -> None:
    """
    This docstring explains the function's purpose: to serve as an entry point that uses a pre-filled arguments object.
    It clarifies that this makes the core logic callable by other Python scripts, not just the command line.
    """
    execute_inspection_run(args_namespace)
    # Calls the main execution function, passing along the pre-filled arguments.

def main():
    """
    This docstring describes the function's role as the main entry point when the script is run from the command line.
    It drives the D-Scope Blink system by parsing command-line arguments.
    """
    # --- Argument Parsing ---
    # This comment block indicates the section where command-line arguments are defined and parsed.
    parser = argparse.ArgumentParser(description="D-Scope Blink: Automated Fiber Optic End Face Inspection System.") # Creates an ArgumentParser object with a description of the program.
    parser.add_argument("input_dir", type=str, help="Path to the directory containing images to inspect.") # Defines a required positional argument for the input directory.
    parser.add_argument("output_dir", type=str, help="Path to the directory where results will be saved.") # Defines a required positional argument for the output directory.
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the JSON configuration file (default: config.json).") # Defines an optional argument for the config file path, with a default value.
    parser.add_argument("--calibration_file", type=str, default="calibration.json", help="Path to the JSON calibration file (default: calibration.json).") # Defines an optional argument for the calibration file path, with a default value.
    parser.add_argument("--profile", type=str, default="deep_inspection", choices=["fast_scan", "deep_inspection"], help="Processing profile to use (default: deep_inspection).") # Defines an optional argument for the processing profile, restricted to specific choices.
    parser.add_argument("--fiber_type", type=str, default="single_mode_pc", help="Key for fiber type specific rules, e.g., 'single_mode_pc', 'multi_mode_pc' (must match config.json).") # Defines an optional argument for the fiber type.
    parser.add_argument("--core_dia_um", type=float, default=None, help="Optional: Known core diameter in microns for this batch.") # Defines an optional argument to specify the core diameter.
    parser.add_argument("--clad_dia_um", type=float, default=None, help="Optional: Known cladding diameter in microns for this batch.") # Defines an optional argument to specify the cladding diameter.
    
    args = parser.parse_args() # Parses the command-line arguments provided by the user into a namespace object.
    execute_inspection_run(args) # Calls the core logic of the program, passing the parsed arguments.

if __name__ == "__main__":
    # This is a standard Python construct that checks if the script is being run directly (not imported as a module).
    main() # If the script is being run directly, it calls the main() function to start the application.