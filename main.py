#!/usr/bin/env python3
# main.py

"""
Main Orchestration Script
========================================
It handles command-line arguments, orchestrates the batch
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
        f"[CRITICAL ERROR] could not start due to missing or problematic modules.\n"
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
    log_file_name = f"inspection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    fiber_type_key: str
) -> Dict[str, Any]: # Ensure it always returns a Dict
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
        A dictionary containing summary results for the image.
    """
    image_start_time = time.perf_counter() # Start timer for image processing.
    logging.info(f"--- Processing image: {image_path.name} ---")
    output_dir_image.mkdir(parents=True, exist_ok=True) # Ensure image-specific output directory exists.


    # --- 1. Load and Preprocess Image ---
    logging.info("Step 1: Loading and Preprocessing...") # Log current step.
    preprocess_results = load_and_preprocess_image(str(image_path), profile_config) # Call function to load and preprocess.
    if preprocess_results is None: # Check if preprocessing failed.
        logging.error(f"Failed to load/preprocess image {image_path.name}. Skipping.") # Log error.
        return {
            "image_filename": image_path.name,
            "pass_fail_status": "ERROR_LOAD_PREPROCESS",
            "processing_time_s": round(time.perf_counter() - image_start_time, 2),
            "total_defect_count": 0,
            "core_defect_count": 0,
            "cladding_defect_count": 0,
            "failure_reason_summary": "Load/preprocess failed"
        }
    # Unpack results from preprocessing.
    original_bgr, original_gray, processed_image = preprocess_results

    # --- 2. Locate Fiber Structure (Cladding and Core) ---
    logging.info("Step 2: Locating Fiber Structure...") # Log current step.
    localization_data = locate_fiber_structure(processed_image, profile_config, original_gray_image=original_gray)# Locate fiber structure.
    if localization_data is None or "cladding_center_xy" not in localization_data: # If localization failed.
        logging.error(f"Failed to localize fiber structure in {image_path.name}. Skipping.")
        return {
            "image_filename": image_path.name,
            "pass_fail_status": "ERROR_LOCALIZATION",
            "processing_time_s": round(time.perf_counter() - image_start_time, 2),
            "total_defect_count": 0,
            "core_defect_count": 0,
            "cladding_defect_count": 0,
            "failure_reason_summary": "Localization failed"
        }
    
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


    analysis_summary = {
        "image_filename": image_path.name,
        "cladding_diameter_px": None,
        "core_diameter_px": None,
        "characterized_defects": [],
        "overall_status": "UNKNOWN",
        "total_defect_count": 0,
        "failure_reasons": [],
        "um_per_px_used": current_image_um_per_px
    }
    
    # Add detected diameters to analysis summary
    if localization_data:
        cladding_diameter_px = localization_data.get("cladding_radius_px", 0) * 2
        core_diameter_px = localization_data.get("core_radius_px", 0) * 2
        
        logging.info(f"Detected diameters for {image_path.name}:")
        logging.info(f"  - Cladding diameter: {cladding_diameter_px:.1f} pixels")
        logging.info(f"  - Core diameter: {core_diameter_px:.1f} pixels")
        
        analysis_summary['cladding_diameter_px'] = cladding_diameter_px
        analysis_summary['core_diameter_px'] = core_diameter_px

    # --- 3. Generate Zone Masks ---
    zone_start_time = time.perf_counter()
    logging.info("Step 3: Generating Zone Masks...")
    try:
        zone_definitions_for_type = get_zone_definitions(fiber_type_key)
    except ValueError as e: # Handle if fiber type not found in config.
        logging.error(f"Configuration error for fiber type '{fiber_type_key}': {e}. Cannot generate zone masks for {image_path.name}. Skipping.")
        return {
            "image_filename": image_path.name,
            "pass_fail_status": "ERROR_CONFIG_ZONES",
            "processing_time_s": round(time.perf_counter() - image_start_time, 2),
            "total_defect_count": 0,
            "core_defect_count": 0,
            "cladding_defect_count": 0,
            "failure_reason_summary": f"Config error for fiber type '{fiber_type_key}': {e}"
        }

    zone_masks = generate_zone_masks( # Generate zone masks.
        processed_image.shape, localization_data, zone_definitions_for_type,
        current_image_um_per_px, user_core_dia_um, user_clad_dia_um
    )
    zone_duration = time.perf_counter() - zone_start_time
    logging.debug(f"Zone mask generation took {zone_duration:.3f}s")
    
    if not zone_masks: # If zone mask generation failed.
        logging.error(f"Failed to generate zone masks for {image_path.name}. Skipping.")
        return {
            "image_filename": image_path.name,
            "pass_fail_status": "ERROR_ZONES",
            "processing_time_s": round(time.perf_counter() - image_start_time, 2),
            "total_defect_count": 0,
            "core_defect_count": 0,
            "cladding_defect_count": 0,
            "failure_reason_summary": "Zone mask generation failed"
        }

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
        # CORRECTED CALL to detect_defects: Added zone_name [cite: 109, 317]
        defects_in_zone_mask, zone_confidence_map = detect_defects(
            processed_image, zone_mask_np, zone_name, profile_config, global_algo_params
        )
        all_zone_defect_masks[zone_name] = defects_in_zone_mask
        combined_final_defect_mask = cv2.bitwise_or(combined_final_defect_mask, defects_in_zone_mask)
        combined_confidence_map = np.maximum(combined_confidence_map, zone_confidence_map)
    
    # --- 5. Characterize, Classify Defects and Apply Pass/Fail ---
    logging.info("Step 5: Analyzing Defects and Applying Rules...")
    characterized_defects, overall_status, total_defect_count = characterize_and_classify_defects(
        combined_final_defect_mask, 
        zone_masks, 
        profile_config, 
        current_image_um_per_px, 
        image_path.name,
        confidence_map=combined_confidence_map
    )
    
    # CORRECTED CALL to apply_pass_fail_rules: Changed zone_definitions_for_type to fiber_type_key [cite: 113, 321]
    overall_status, failure_reasons = apply_pass_fail_rules(characterized_defects, fiber_type_key)

    analysis_summary = { # Create analysis summary dictionary.
        "image_filename": image_path.name,
        "characterized_defects": characterized_defects,
        "overall_status": overall_status, 
        "total_defect_count": total_defect_count, 
        "failure_reasons": failure_reasons, 
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
        "pass_fail_status": overall_status, 
        "processing_time_s": round(processing_time_s, 2),
        "total_defect_count": total_defect_count, 
        "core_defect_count": sum(1 for d in characterized_defects if d["zone"] == "Core"),
        "cladding_defect_count": sum(1 for d in characterized_defects if d["zone"] == "Cladding"),
        "failure_reason_summary": "; ".join(failure_reasons) if failure_reasons else "N/A" 
    }
    return summary_for_batch # Return summary.

def execute_inspection_run(args_namespace: Any) -> None:
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
        config_file_path = str(args_namespace.config_file)
        global_config = load_config(config_file_path) # Load global configuration.
    except (FileNotFoundError, ValueError) as e: # Handle config loading errors.
        print(f"[CRITICAL] Failed to load configuration: {e}. Exiting.", file=sys.stderr)
        try:
            fallback_log_dir = Path(".") / "error_logs"
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
        current_run_output_dir 
    )

    logging.info("Inspection System Started.")
    logging.info(f"Input Directory: {args_namespace.input_dir}")
    logging.info(f"Output Directory (this run): {current_run_output_dir}")
    logging.info(f"Using Profile: {args_namespace.profile}")
    logging.info(f"Fiber Type Key for Rules: {args_namespace.fiber_type}")
    
    # Add fiber type validation and correction
    fiber_type_corrections = {
        "single_mode": "single_mode_pc",
        "multi_mode": "multi_mode_pc",
        "sm": "single_mode_pc",
        "mm": "multi_mode_pc",
        "singlemode": "single_mode_pc",
        "multimode": "multi_mode_pc"
    }
    
    if args_namespace.fiber_type in fiber_type_corrections:
        corrected_type = fiber_type_corrections[args_namespace.fiber_type]
        logging.warning(f"Correcting fiber type '{args_namespace.fiber_type}' to '{corrected_type}'")
        args_namespace.fiber_type = corrected_type


    if args_namespace.core_dia_um: logging.info(f"User Provided Core Diameter: {args_namespace.core_dia_um} µm")
    if args_namespace.clad_dia_um: logging.info(f"User Provided Cladding Diameter: {args_namespace.clad_dia_um} µm")

    try:
        active_profile_config = get_processing_profile(args_namespace.profile) # Get active processing profile.
    except ValueError as e: # Handle if profile not found.
        logging.critical(f"Failed to get processing profile '{args_namespace.profile}': {e}. Exiting.")
        sys.exit(1) # Exit.

    # --- Load Calibration Data ---
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



    # Initialize advanced models if enabled
    advanced_models_initialized = False

    if global_config.get("algorithm_parameters", {}).get("enable_advanced_models", True):
        logging.info("Initializing advanced detection models...")
        
        # Initialize Anomalib
        if ANOMALIB_FULL_AVAILABLE:
            try:
                anomalib_config_path = global_config["algorithm_parameters"].get("anomalib_config_path")
                anomalib_detector = AnomalibDefectDetector(anomalib_config_path)
                
                # Load pre-trained models
                model_paths = {
                    "padim": Path("models/anomalib/padim/openvino"),
                    "patchcore": Path("models/anomalib/patchcore/openvino")
                }
                
                existing_models = {k: v for k, v in model_paths.items() if v.exists()}
                if existing_models:
                    anomalib_detector.create_ensemble_detector(existing_models)
                    global_config["algorithm_parameters"]["anomalib_detector_instance"] = anomalib_detector
                    logging.info(f"Anomalib ensemble initialized with models: {list(existing_models.keys())}")
            except Exception as e:
                logging.warning(f"Failed to initialize Anomalib: {e}")
        
        # Initialize PaDiM Specific
        if PADIM_SPECIFIC_AVAILABLE:
            try:
                padim_model = FiberPaDiM(backbone='resnet18', device='cuda' if torch.cuda.is_available() else 'cpu')
                
                # Load pre-trained model if available
                padim_model_path = global_config["algorithm_parameters"].get("padim_model_path")
                if padim_model_path and Path(padim_model_path).exists():
                    checkpoint = torch.load(padim_model_path)
                    padim_model.patch_means = checkpoint['patch_means']
                    padim_model.C = checkpoint['C']
                    padim_model.C_inv = checkpoint['C_inv']
                    padim_model.R = checkpoint['R']
                    padim_model.fitted = True
                    logging.info("Loaded pre-trained PaDiM model")
                
                global_config["algorithm_parameters"]["padim_specific_instance"] = padim_model
            except Exception as e:
                logging.warning(f"Failed to initialize PaDiM specific: {e}")
        
        # Initialize SegDecNet
        if SEGDECNET_AVAILABLE:
            try:
                segdecnet_model_path = global_config["algorithm_parameters"].get("segdecnet_model_path")
                segdecnet = FiberSegDecNet(
                    model_path=segdecnet_model_path if Path(segdecnet_model_path).exists() else None,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                global_config["algorithm_parameters"]["segdecnet_instance"] = segdecnet
                logging.info("SegDecNet initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize SegDecNet: {e}")
        
        advanced_models_initialized = True
        logging.info("Advanced model initialization complete")

    # --- Batch Processing ---
    batch_start_time = time.perf_counter() # Start timer for batch processing.
    all_image_summaries: List[Dict[str, Any]] = [] # Initialize list for summaries of all images.

    # Parallel processing setup
    from multiprocessing import Pool, cpu_count
    from functools import partial

    def process_image_wrapper(args):
        """Wrapper for multiprocessing"""
        image_path, output_dir, profile_config, global_config, um_per_px, core_dia, clad_dia, fiber_type = args
        try:
            return process_single_image(
                image_path, output_dir, profile_config, global_config,
                um_per_px, core_dia, clad_dia, fiber_type
            )
        except Exception as e:
            logging.error(f"Error processing {image_path.name}: {e}")
            return {
                "image_filename": image_path.name,
                "pass_fail_status": "ERROR_PROCESSING",
                "processing_time_s": 0,
                "total_defect_count": 0,
                "core_defect_count": 0,
                "cladding_defect_count": 0,
                "failure_reason_summary": str(e)
            }

    # Determine number of processes
    num_processes = min(cpu_count() - 1, len(image_paths_to_process))
    num_processes = max(1, num_processes)  # At least 1 process

    if num_processes > 1 and len(image_paths_to_process) > 1:
        logging.info(f"Using parallel processing with {num_processes} processes")
        
        # Prepare arguments for each image
        process_args = [
            (
                image_path,
                current_run_output_dir / image_path.stem,
                active_profile_config,
                global_config,
                loaded_um_per_px,
                args_namespace.core_dia_um,
                args_namespace.clad_dia_um,
                args_namespace.fiber_type
            )
            for image_path in image_paths_to_process
        ]
        
        # Process in parallel
        with Pool(processes=num_processes) as pool:
            all_image_summaries = pool.map(process_image_wrapper, process_args)
    else:
        # Fall back to sequential processing for single image or if parallel not beneficial
        logging.info("Using sequential processing")
        all_image_summaries = []
        for i, image_file_path in enumerate(image_paths_to_process):
            logging.info(f"--- Starting image {i+1}/{len(image_paths_to_process)}: {image_file_path.name} ---")
            image_specific_output_subdir = current_run_output_dir / image_file_path.stem
            summary = process_image_wrapper((
                image_file_path,
                image_specific_output_subdir,
                active_profile_config,
                global_config,
                loaded_um_per_px,
                args_namespace.core_dia_um,
                args_namespace.clad_dia_um,
                args_namespace.fiber_type
            ))
            all_image_summaries.append(summary)
        image_specific_output_subdir = current_run_output_dir / image_file_path.stem
        current_image_processing_start_time = time.perf_counter() # Timer for this specific image processing attempt
        
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
            # process_single_image is designed to always return a dict.
            # Adding a safeguard for unforeseen None returns.
            if summary is None: 
                 logging.error(f"Critical internal error: process_single_image returned None for {image_file_path.name}. This should not happen.")
                 summary = {
                    "image_filename": image_file_path.name,
                    "pass_fail_status": "ERROR_UNEXPECTED_NONE_RETURN",
                    "processing_time_s": round(time.perf_counter() - current_image_processing_start_time, 2),
                    "total_defect_count": 0,
                    "core_defect_count": 0,
                    "cladding_defect_count": 0,
                    "failure_reason_summary": "Internal error: process_single_image returned None."
                }
            all_image_summaries.append(summary) 
        except Exception as e: # Handle unexpected errors during single image processing.
            logging.error(f"Unexpected error processing {image_file_path.name}: {e}", exc_info=True) # Log full traceback
            failure_summary = { 
                "image_filename": image_file_path.name,
                "pass_fail_status": "ERROR_UNHANDLED_EXCEPTION",
                "processing_time_s": round(time.perf_counter() - current_image_processing_start_time, 2),
                "total_defect_count": 0,
                "core_defect_count": 0,
                "cladding_defect_count": 0,
                "failure_reason_summary": f"Unhandled exception: {str(e)}"
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
    logging.info(f"Batch Processing Complete ---")
    logging.info(f"Total images processed: {len(image_paths_to_process)}")
    logging.info(f"Total batch duration: {batch_duration:.2f} seconds.")
    logging.info(f"All reports for this run saved in: {current_run_output_dir}")

# CORRECTED Function Definition: Changed args_namespace type hint to Any [cite: 129, 340]
def main_with_args(args_namespace: Any) -> None:
    """
    Entry point that uses a pre-filled args_namespace object.
    This is callable by other scripts.
    """
    execute_inspection_run(args_namespace)

def main():
    """
    Main function to drive the inspection system from Command Line.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Automated Fiber Optic End Face Inspection System.") # Create argument parser.
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