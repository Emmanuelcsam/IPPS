#!/usr/bin/env python3
# main.py

"""
D-Scope Blink: Main Orchestration Script
========================================
This script is the main entry point for the D-Scope Blink Automated Fiber Optic
Inspection System. It handles command-line arguments, orchestrates the batch
processing workflow, and integrates all other modules (config, calibration,
image processing, analysis, and reporting).
"""
import cv2 # OpenCV for image processing tasks.
import argparse # Standard library for parsing command-line arguments.
import logging # Standard library for logging events.
import time # Standard library for time-related functions (performance tracking).
import datetime # Standard library for date and time objects.
from pathlib import Path # Standard library for object-oriented path manipulation.
import sys # Standard library for system-specific parameters and functions.
import pandas as pd # Pandas for creating the final summary CSV report.
from typing import Dict, Any, Optional, List # For type hinting

# --- D-Scope Blink Modules ---
# These imports assume the modules are in the same directory or accessible via PYTHONPATH.
try:
    from advanced_visualization import InteractiveVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    
try:
    from config_loader import load_config, get_processing_profile, get_zone_definitions # Import config loading functions.
    from calibration import load_calibration_data # Import calibration data loading function.
    from image_processing import ( # Import image processing functions.
        load_and_preprocess_image,
        locate_fiber_structure,
        generate_zone_masks,
        detect_defects
    )
    from analysis import characterize_and_classify_defects, apply_pass_fail_rules # Import analysis functions.
    from reporting import generate_annotated_image, generate_defect_csv_report, generate_polar_defect_histogram # Import reporting functions.
except ImportError as e: # Handle import errors if modules are not found.
    error_msg = (
        f"[CRITICAL ERROR] D-Scope Blink could not start due to missing or problematic modules.\n"
        f"Details: {e}\n"
        f"Please ensure all required Python modules (config_loader.py, calibration.py, "
        f"image_processing.py, analysis.py, reporting.py, and their dependencies like OpenCV, Pandas, Numpy) "
        f"are correctly installed and accessible in your Python environment (PYTHONPATH).\n"
        f"Refer to the installation documentation for troubleshooting."
    )
    print(error_msg, file=sys.stderr)
    sys.exit(1) # Exit if essential modules cannot be imported.

# Import numpy for type hinting if not already (it's used in image_processing)
import numpy as np

def setup_logging(log_level_str: str, log_to_console: bool, output_dir: Path) -> None:
    """
    Configures the logging system for the application.

    Args:
        log_level_str: The desired logging level as a string (e.g., "INFO", "DEBUG").
        log_to_console: Boolean indicating whether to log to the console.
        output_dir: The base directory where log files will be saved.
    """
    numeric_log_level = getattr(logging, log_level_str.upper(), logging.INFO) # Convert string log level to numeric.
    
    log_format = '[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s' # Define log message format.
    date_format = '%Y-%m-%d %H:%M:%S' # Define date format for logs.

    handlers: List[logging.Handler] = [] # Initialize list for log handlers.

    # --- File Handler ---
    # Create a unique log file name with a timestamp in the specified output directory.
    log_file_name = f"d_scope_blink_inspection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = output_dir / "logs" / log_file_name # Define full log file path.
    log_file_path.parent.mkdir(parents=True, exist_ok=True) # Create 'logs' subdirectory if it doesn't exist.
    
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8') # Create file handler.
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format)) # Set formatter for file handler.
    handlers.append(file_handler) # Add file handler to list.

    # --- Console Handler (Optional) ---
    if log_to_console: # If logging to console is enabled.
        console_handler = logging.StreamHandler(sys.stdout) # Create console handler.
        console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format)) # Set formatter for console handler.
        handlers.append(console_handler) # Add console handler to list.

    logging.basicConfig(level=numeric_log_level, handlers=handlers, force=True) # Configure basic logging with handlers. `force=True` helps in re-running in notebooks/interactive sessions.
    logging.info(f"Logging configured. Level: {log_level_str}. Log file: {log_file_path}")

def process_single_image(
    image_path: Path,
    output_dir_image: Path,
    profile_config: Dict[str, Any],
    global_config: Dict[str, Any],
    calibration_um_per_px: Optional[float],
    user_core_dia_um: Optional[float],
    user_clad_dia_um: Optional[float],
    fiber_type_key: str # e.g. "single_mode_pc"
) -> Optional[Dict[str, Any]]:
    """
    Orchestrates the full processing pipeline for a single image.

    Args:
        image_path: Path to the image file.
        output_dir_image: Directory to save results for this specific image.
        profile_config: The active processing profile configuration.
        global_config: The full global configuration dictionary.
        calibration_um_per_px: Calibrated um_per_px from file (can be None).
        user_core_dia_um: User-provided core diameter in microns (can be None).
        user_clad_dia_um: User-provided cladding diameter in microns (can be None).
        fiber_type_key: Key for the fiber type being processed.

    Returns:
        A dictionary containing summary results for the image, or None on critical failure.
    """
    image_start_time = time.perf_counter() # Start timer for image processing.
    logging.info(f"--- Processing image: {image_path.name} ---")
    output_dir_image.mkdir(parents=True, exist_ok=True) # Ensure image-specific output directory exists.

    # --- 1. Load and Preprocess Image ---
    logging.info("Step 1: Loading and Preprocessing...")
    preprocess_results = load_and_preprocess_image(str(image_path), profile_config) # Load and preprocess image.
    if preprocess_results is None: # If preprocessing failed.
        logging.error(f"Failed to load/preprocess image {image_path.name}. Skipping.")
        return {"image_filename": image_path.name, "status": "ERROR_LOAD_PREPROCESS", "processing_time_s": time.perf_counter() - image_start_time, "total_defect_count": 0, "failure_reasons": ["Load/preprocess failed"]}
    original_bgr, original_gray, processed_image = preprocess_results # Unpack preprocessing results.

    # --- 2. Locate Fiber Structure (Cladding and Core) ---
    logging.info("Step 2: Locating Fiber Structure...")
    localization_data = locate_fiber_structure(processed_image, profile_config) # Locate fiber structure.
    if localization_data is None or "cladding_center_xy" not in localization_data: # If localization failed.
        logging.error(f"Failed to localize fiber structure in {image_path.name}. Skipping.")
        return {"image_filename": image_path.name, "status": "ERROR_LOCALIZATION", "processing_time_s": time.perf_counter() - image_start_time, "total_defect_count": 0, "failure_reasons": ["Localization failed"]}
    
    current_image_um_per_px = calibration_um_per_px # Default to generic calibration.
    if user_clad_dia_um is not None: # If user provided cladding diameter.
        detected_cladding_radius_px = localization_data.get("cladding_radius_px") # Get detected cladding radius.
        if detected_cladding_radius_px and detected_cladding_radius_px > 0: # If radius is valid.
            detected_cladding_diameter_px = detected_cladding_radius_px * 2.0 # Calculate diameter.
            current_image_um_per_px = user_clad_dia_um / detected_cladding_diameter_px # Calculate image-specific um/px.
            logging.info(f"Using image-specific scale for {image_path.name}: {current_image_um_per_px:.4f} µm/px (user_clad_dia={user_clad_dia_um}µm, detected_clad_dia={detected_cladding_diameter_px:.1f}px).")
        elif localization_data.get("cladding_ellipse_params"): # If ellipse parameters available.
            ellipse_axes = localization_data["cladding_ellipse_params"][1] # Get ellipse axes (minor, major).
            detected_cladding_diameter_px = ellipse_axes[1] # Major axis.
            if detected_cladding_diameter_px > 0: # If diameter is valid.
                current_image_um_per_px = user_clad_dia_um / detected_cladding_diameter_px # Calculate image-specific um/px.
                logging.info(f"Using image-specific scale (ellipse) for {image_path.name}: {current_image_um_per_px:.4f} µm/px (user_clad_dia={user_clad_dia_um}µm, detected_major_axis={detected_cladding_diameter_px:.1f}px).")
            else: # If diameter is not valid.
                 logging.warning(f"Detected cladding diameter (ellipse major axis) is zero for {image_path.name}. Cannot calculate image-specific scale. Falling back to generic calibration: {calibration_um_per_px}")
        else: # If radius/ellipse not valid.
            logging.warning(f"Could not determine detected cladding diameter for {image_path.name} to calculate image-specific scale. Falling back to generic calibration: {calibration_um_per_px}")
    elif current_image_um_per_px: # If using generic calibration.
        logging.info(f"Using generic calibration scale for {image_path.name}: {current_image_um_per_px:.4f} µm/px.")
    else: # If no scale available.
        logging.info(f"No µm/px scale available for {image_path.name}. Measurements will be in pixels.")


    # --- 3. Generate Zone Masks ---
    logging.info("Step 3: Generating Zone Masks...")
    try:
        zone_definitions_for_type = get_zone_definitions(fiber_type_key)
    except ValueError as e: # Handle if fiber type not found in config.
        logging.error(f"Configuration error for fiber type '{fiber_type_key}': {e}. Cannot generate zone masks for {image_path.name}. Skipping.")
        return {"image_filename": image_path.name, "status": "ERROR_CONFIG_ZONES", "processing_time_s": time.perf_counter() - image_start_time, "total_defect_count": 0, "failure_reasons": [f"Config error for fiber type {fiber_type_key}"]}

    zone_masks = generate_zone_masks( # Generate zone masks.
        processed_image.shape, localization_data, zone_definitions_for_type,
        current_image_um_per_px, user_core_dia_um, user_clad_dia_um
    )
    if not zone_masks: # If zone mask generation failed.
        logging.error(f"Failed to generate zone masks for {image_path.name}. Skipping.")
        return {"image_filename": image_path.name, "status": "ERROR_ZONES", "processing_time_s": time.perf_counter() - image_start_time, "total_defect_count": 0, "failure_reasons": ["Zone mask generation failed"]}

    # --- 4. Defect Detection in Each Zone ---
    logging.info("Step 4: Detecting Defects in Zones...")
    all_zone_defect_masks: Dict[str, np.ndarray] = {}
    combined_final_defect_mask = np.zeros_like(processed_image, dtype=np.uint8)
    combined_confidence_map = np.zeros_like(processed_image, dtype=np.float32)
    global_algo_params = global_config.get("algorithm_parameters", {})

    for zone_name, zone_mask_np in zone_masks.items():
        if np.sum(zone_mask_np) == 0:
            logging.debug(f"Zone '{zone_name}' is empty. Skipping defect detection.")
            all_zone_defect_masks[zone_name] = np.zeros_like(processed_image, dtype=np.uint8)
            continue
        
        logging.debug(f"Detecting defects in zone: '{zone_name}'...")
        defects_in_zone_mask, zone_confidence_map = detect_defects(
            processed_image, zone_mask_np, profile_config, global_algo_params
        )
        all_zone_defect_masks[zone_name] = defects_in_zone_mask
        combined_final_defect_mask = cv2.bitwise_or(combined_final_defect_mask, defects_in_zone_mask)
        combined_confidence_map = np.maximum(combined_confidence_map, zone_confidence_map)
    
    # --- 5. Characterize, Classify Defects and Apply Pass/Fail ---
    logging.info("Step 5: Analyzing Defects and Applying Rules...")
    # Updated call to characterize_and_classify_defects, expecting three return values:
    # characterized_defects, overall_status (preliminary), total_defect_count
    characterized_defects, overall_status, total_defect_count = characterize_and_classify_defects(
        combined_final_defect_mask, 
        zone_masks, 
        profile_config, 
        current_image_um_per_px, 
        image_path.name,
        confidence_map=combined_confidence_map
    )
    
    # Apply pass/fail rules. This call will update overall_status and provide failure_reasons.
    # zone_definitions_for_type was fetched in Step 3.
    overall_status, failure_reasons = apply_pass_fail_rules(characterized_defects, zone_definitions_for_type)

    analysis_summary = { # Create analysis summary dictionary.
        "image_filename": image_path.name,
        "characterized_defects": characterized_defects,
        "overall_status": overall_status, # Final status from apply_pass_fail_rules
        "total_defect_count": total_defect_count, # Count from characterize_and_classify_defects
        "failure_reasons": failure_reasons, # Reasons from apply_pass_fail_rules
        "um_per_px_used": current_image_um_per_px
    }

    # --- 6. Generate Reports ---
    logging.info("Step 6: Generating Reports...")
    annotated_img_path = output_dir_image / f"{image_path.stem}_annotated.png" # Define path for annotated image.
    generate_annotated_image( # Generate annotated image.
        original_bgr, analysis_summary, localization_data, zone_masks, fiber_type_key, annotated_img_path
    )
    csv_report_path = output_dir_image / f"{image_path.stem}_report.csv" # Define path for CSV report.
    generate_defect_csv_report(analysis_summary, csv_report_path) # Generate CSV report.
    
    histogram_path = output_dir_image / f"{image_path.stem}_histogram.png" # Define path for histogram.
    generate_polar_defect_histogram( # Generate polar histogram.
        analysis_summary, localization_data, zone_masks, fiber_type_key, histogram_path
    )
    
    processing_time_s = time.perf_counter() - image_start_time # Calculate processing time.
    logging.info(f"--- Finished processing {image_path.name}. Duration: {processing_time_s:.2f}s ---")

    # --- 7. Advanced Visualization (Optional) ---
    if VISUALIZATION_AVAILABLE and global_config.get("general_settings", {}).get("enable_visualization", False):
        try:
            visualizer = InteractiveVisualizer()
            visualizer.show_inspection_results(
                original_bgr,
                all_zone_defect_masks,
                zone_masks,
                analysis_summary,
                interactive=False  # Non-blocking for batch processing
            )
        except Exception as e:
            logging.warning(f"Visualization failed for {image_path.name}: {e}")
            
    summary_for_batch = { # Create summary dictionary for batch.
        "image_filename": image_path.name,
        "pass_fail_status": overall_status, # Use final overall_status from apply_pass_fail_rules
        "processing_time_s": round(processing_time_s, 2),
        "total_defect_count": total_defect_count, # Use total_defect_count from characterize_and_classify_defects
        "core_defect_count": sum(1 for d in characterized_defects if d["zone"] == "Core"),
        "cladding_defect_count": sum(1 for d in characterized_defects if d["zone"] == "Cladding"),
        "failure_reason_summary": "; ".join(failure_reasons) if failure_reasons else "N/A" # Use final failure_reasons
    }
    return summary_for_batch # Return summary.

def execute_inspection_run(args_namespace: argparse.Namespace) -> None:
    """
    Core inspection logic that takes an args-like namespace object.
    This function contains the main processing flow.
    """
    # --- Output Directory Setup ---
    base_output_dir = Path(args_namespace.output_dir) # Convert output dir string to Path object.
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') # Generate timestamp.
    current_run_output_dir = base_output_dir / f"run_{run_timestamp}" # Define path for current run output.
    current_run_output_dir.mkdir(parents=True, exist_ok=True) # Create directory.

    # --- Configuration and Logging Setup ---
    try:
        # Convert config_file path to string if it's a Path object from ArgsSimulator
        config_file_path = str(args_namespace.config_file)
        global_config = load_config(config_file_path) # Load global configuration.
    except (FileNotFoundError, ValueError) as e: # Handle config loading errors.
        print(f"[CRITICAL] Failed to load configuration: {e}. Exiting.", file=sys.stderr)
        # Attempt to set up basic logging if possible, or just print
        try:
            # Try to set up basic logging to a default location or console
            fallback_log_dir = Path(".") / "d_scope_blink_error_logs"
            fallback_log_dir.mkdir(parents=True, exist_ok=True)
            setup_logging("ERROR", True, fallback_log_dir)
            logging.critical(f"Failed to load configuration: {e}. Exiting.")
        except Exception as log_e:
            print(f"[CRITICAL] Logging setup failed during config error: {log_e}", file=sys.stderr)
        sys.exit(1) # Exit if config loading fails.

    general_settings = global_config.get("general_settings", {}) # Get general settings from config.
    setup_logging( # Setup logging.
        general_settings.get("log_level", "INFO"),
        general_settings.get("log_to_console", True),
        current_run_output_dir # Log files will go into the run-specific output directory.
    )

    logging.info("D-Scope Blink: Inspection System Started.")
    logging.info(f"Input Directory: {args_namespace.input_dir}")
    logging.info(f"Output Directory (this run): {current_run_output_dir}")
    logging.info(f"Using Profile: {args_namespace.profile}")
    logging.info(f"Fiber Type Key for Rules: {args_namespace.fiber_type}")
    if args_namespace.core_dia_um: logging.info(f"User Provided Core Diameter: {args_namespace.core_dia_um} µm")
    if args_namespace.clad_dia_um: logging.info(f"User Provided Cladding Diameter: {args_namespace.clad_dia_um} µm")

    try:
        active_profile_config = get_processing_profile(args_namespace.profile) # Get active processing profile.
    except ValueError as e: # Handle if profile not found.
        logging.critical(f"Failed to get processing profile '{args_namespace.profile}': {e}. Exiting.")
        sys.exit(1) # Exit.

    # --- Load Calibration Data ---
    # Convert calibration_file path to string if it's a Path object from ArgsSimulator
    calibration_file_path = str(args_namespace.calibration_file)
    calibration_data = load_calibration_data(calibration_file_path) # Load calibration data.
    loaded_um_per_px: Optional[float] = None # Initialize loaded um/px.
    if calibration_data: # If calibration data loaded.
        loaded_um_per_px = calibration_data.get("um_per_px") # Get um/px value.
        if loaded_um_per_px: # If um/px value exists.
            logging.info(f"Loaded µm/pixel scale from '{calibration_file_path}': {loaded_um_per_px:.4f} µm/px.")
        else: # If um/px key missing.
            logging.warning(f"Calibration file '{calibration_file_path}' loaded, but 'um_per_px' key is missing or invalid.")
    else: # If calibration data not loaded.
        logging.warning(f"No calibration data loaded from '{calibration_file_path}'. Measurements may be in pixels if user dimensions not provided.")

    # --- Image Discovery ---
    input_path = Path(args_namespace.input_dir) # Convert input dir string to Path object.
    if not input_path.is_dir(): # Check if input path is a directory.
        logging.critical(f"Input path '{input_path}' is not a valid directory. Exiting.")
        sys.exit(1) # Exit.

    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    image_paths_to_process: List[Path] = [] # Initialize list for image paths.
    for ext in image_extensions: # Iterate through extensions.
        image_paths_to_process.extend(list(input_path.glob(f"*{ext}"))) # Add images with current extension.
        image_paths_to_process.extend(list(input_path.glob(f"*{ext.upper()}"))) # Add with uppercase extension.
    
    image_paths_to_process = sorted(list(set(image_paths_to_process)))

    if not image_paths_to_process: # If no images found.
        logging.info(f"No images found in directory: {input_path}")
        sys.exit(0) # Exit gracefully.

    logging.info(f"Found {len(image_paths_to_process)} images to process in '{input_path}'.")

    # --- Batch Processing ---
    batch_start_time = time.perf_counter() # Start timer for batch processing.
    all_image_summaries: List[Dict[str, Any]] = [] # Initialize list for summaries of all images.

    for i, image_file_path in enumerate(image_paths_to_process): # Iterate through image paths.
        logging.info(f"--- Starting image {i+1}/{len(image_paths_to_process)}: {image_file_path.name} ---")
        image_specific_output_subdir = current_run_output_dir / image_file_path.stem # Define path for image-specific output.
        
        try:
            summary = process_single_image( # Process single image.
                image_file_path,
                image_specific_output_subdir,
                active_profile_config,
                global_config,
                loaded_um_per_px,
                args_namespace.core_dia_um,
                args_namespace.clad_dia_um,
                args_namespace.fiber_type
            )
            # Append summary directly, assuming process_single_image always returns a dict
            all_image_summaries.append(summary) 
        except Exception as e: # Handle unexpected errors during single image processing.
            # Updated logging message as per the snippet
            logging.error(f"Unexpected error processing {image_file_path.name}: {e}")
            # Updated error summary structure as per the snippet
            failure_summary = { 
                "image_filename": image_file_path.name,
                "status": "ERROR_PROCESSING", # Key "status"
                "processing_time_s": 0,
                "total_defect_count": 0,
                "failure_reasons": [str(e)] # Key "failure_reasons" as a list
            }
            all_image_summaries.append(failure_summary)
            
    # --- Final Summary Report ---
    if all_image_summaries: # If summaries exist.
        summary_df = pd.DataFrame(all_image_summaries) # Create DataFrame from summaries.
        summary_report_path = current_run_output_dir / "batch_summary_report.csv" # Define path for summary report.
        try:
            summary_df.to_csv(summary_report_path, index=False, encoding='utf-8') # Save summary report to CSV.
            logging.info(f"Batch summary report saved to: {summary_report_path}")
        except Exception as e: # Handle errors during saving.
            logging.error(f"Failed to save batch summary report: {e}")
    else: # If no summaries (e.g., no images processed or all failed critically before summary).
        logging.warning("No image summaries were generated for the batch report.")

    batch_duration = time.perf_counter() - batch_start_time # Calculate total batch processing time.
    logging.info(f"--- D-Scope Blink: Batch Processing Complete ---")
    logging.info(f"Total images processed: {len(image_paths_to_process)}")
    logging.info(f"Total batch duration: {batch_duration:.2f} seconds.")
    logging.info(f"All reports for this run saved in: {current_run_output_dir}")

def main_with_args(args_namespace: argparse.Namespace) -> None:
    """
    Entry point that uses a pre-filled args_namespace object.
    This is callable by other scripts.
    """
    execute_inspection_run(args_namespace)

def main():
    """
    Main function to drive the D-Scope Blink inspection system from Command Line.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="D-Scope Blink: Automated Fiber Optic End Face Inspection System.") # Create argument parser.
    parser.add_argument("input_dir", type=str, help="Path to the directory containing images to inspect.") # Input directory argument.
    parser.add_argument("output_dir", type=str, help="Path to the directory where results will be saved.") # Output directory argument.
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the JSON configuration file (default: config.json).") # Config file argument.
    parser.add_argument("--calibration_file", type=str, default="calibration.json", help="Path to the JSON calibration file (default: calibration.json).") # Calibration file argument.
    parser.add_argument("--profile", type=str, default="deep_inspection", choices=["fast_scan", "deep_inspection"], help="Processing profile to use (default: deep_inspection).") # Processing profile argument.
    parser.add_argument("--fiber_type", type=str, default="single_mode_pc", help="Key for fiber type specific rules, e.g., 'single_mode_pc', 'multi_mode_pc' (must match config.json).") # Fiber type argument.
    parser.add_argument("--core_dia_um", type=float, default=None, help="Optional: Known core diameter in microns for this batch.") # Core diameter argument.
    parser.add_argument("--clad_dia_um", type=float, default=None, help="Optional: Known cladding diameter in microns for this batch.") # Cladding diameter argument.
    
    args = parser.parse_args() # Parse command-line arguments.
    execute_inspection_run(args) # Call the core logic

if __name__ == "__main__":
    main() # Call the main function when script is executed.