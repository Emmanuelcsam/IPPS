#!/usr/bin/env python3
# main.py

"""
Main Orchestration Script with Performance Optimizations
========================================
Enhanced version with image resizing optimization, progress monitoring,
and improved error handling for OpenCV contrib modules.
"""
import cv2
import argparse
import logging
import time
import datetime
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from functools import wraps
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Performance timer decorator
def performance_timer(func):
    """Decorator to measure and log function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        logging.info(f"{func.__name__} took {elapsed_time:.2f} seconds")
        return result
    return wrapper

# Image processor for performance optimization
class ImageProcessor:
    """Handles image processing optimizations"""
    
    def __init__(self, max_processing_size=1024):
        self.max_processing_size = max_processing_size
    
    def resize_for_processing(self, image, max_size=None):
        """
        Resize image for faster processing if needed.
        Returns tuple of (processed_image, scale_factor)
        """
        if max_size is None:
            max_size = self.max_processing_size
            
        h, w = image.shape[:2]
        scale_factor = 1.0
        
        if max(h, w) > max_size:
            scale_factor = max_size / max(h, w)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logging.debug(f"Image resized from {w}x{h} to {new_w}x{new_h} (scale: {scale_factor:.2f})")
            return resized, scale_factor
        
        return image, scale_factor

# Safe thinning implementation with fallback
def safe_thinning(binary_image):
    """Safe thinning with fallback if opencv-contrib not available"""
    try:
        import cv2
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
            return cv2.ximgproc.thinning(binary_image)
        else:
            logging.debug("cv2.ximgproc.thinning not available, using morphological skeleton")
            return morphological_skeleton(binary_image)
    except AttributeError:
        logging.debug("OpenCV contrib not available, using fallback skeleton")
        return morphological_skeleton(binary_image)

def morphological_skeleton(binary_image):
    """Fallback skeleton using morphological operations"""
    skeleton = np.zeros_like(binary_image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    while True:
        eroded = cv2.erode(binary_image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary_image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary_image = eroded.copy()
        
        if cv2.countNonZero(binary_image) == 0:
            break
    
    return skeleton

# Check for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. PaDiM and SegDecNet models will be disabled.")

try:
    from anomalib_integration import AnomalibDefectDetector
    ANOMALIB_FULL_AVAILABLE = True
except ImportError:
    ANOMALIB_FULL_AVAILABLE = False
    logging.warning("Anomalib full integration not available")

try:
    from padim_specific import FiberPaDiM
    PADIM_SPECIFIC_AVAILABLE = True
except ImportError:
    PADIM_SPECIFIC_AVAILABLE = False

try:
    from segdecnet_integration import FiberSegDecNet
    SEGDECNET_AVAILABLE = True
except ImportError:
    SEGDECNET_AVAILABLE = False

try:
    from advanced_scratch_detection import AdvancedScratchDetector
    ADVANCED_SCRATCH_AVAILABLE = True
except ImportError:
    ADVANCED_SCRATCH_AVAILABLE = False

try:
    from anomaly_detection import AnomalyDetector
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False

try:
    from advanced_visualization import InteractiveVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Import core modules
try:
    from config_loader import load_config, get_processing_profile, get_zone_definitions
    from calibration import load_calibration_data
    from image_processing import (
        load_and_preprocess_image,
        locate_fiber_structure,
        generate_zone_masks,
        detect_defects
    )
    from analysis import characterize_and_classify_defects, apply_pass_fail_rules
    from reporting import generate_annotated_image, generate_defect_csv_report, generate_polar_defect_histogram
    # Import performance optimizer if available
    try:
        from performance_optimizer import ImageProcessor as OptimizedImageProcessor
        PERFORMANCE_OPTIMIZER_AVAILABLE = True
    except ImportError:
        PERFORMANCE_OPTIMIZER_AVAILABLE = False
        logging.debug("Performance optimizer module not available, using built-in optimization")
except ImportError as e:
    error_msg = (
        f"[CRITICAL ERROR] could not start due to missing or problematic modules.\n"
        f"Details: {e}\n"
        f"Please ensure all required Python modules (config_loader.py, calibration.py, "
        f"image_processing.py, analysis.py, reporting.py, and their dependencies like OpenCV, Pandas, Numpy) "
        f"are correctly installed and accessible in your Python environment (PYTHONPATH).\n"
        f"Refer to the installation documentation for troubleshooting."
    )
    print(error_msg, file=sys.stderr)
    sys.exit(1)

def setup_logging(log_level_str: str, log_to_console: bool, output_dir: Path) -> None:
    """
    Configures the logging system for the application.

    Args:
        log_level_str: The desired logging level as a string (e.g., "INFO", "DEBUG").
        log_to_console: Boolean indicating whether to log to the console.
        output_dir: The base directory where log files will be saved.
    """
    numeric_log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    log_format = '[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    handlers: List[logging.Handler] = []

    # File Handler
    log_file_name = f"inspection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = output_dir / "logs" / log_file_name
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    handlers.append(file_handler)

    # Console Handler (Optional)
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers.append(console_handler)

    logging.basicConfig(level=numeric_log_level, handlers=handlers, force=True)
    logging.info(f"Logging configured. Level: {log_level_str}. Log file: {log_file_path}")

@performance_timer
def process_single_image_optimized(
    image_path: Path,
    output_dir_image: Path,
    profile_config: Dict[str, Any],
    global_config: Dict[str, Any],
    calibration_um_per_px: Optional[float],
    user_core_dia_um: Optional[float],
    user_clad_dia_um: Optional[float],
    fiber_type_key: str
) -> Dict[str, Any]:
    """
    Optimized version of process_single_image with performance improvements.
    """
    image_start_time = time.perf_counter()
    logging.info(f"--- Processing image: {image_path.name} ---")
    output_dir_image.mkdir(parents=True, exist_ok=True)

    # Initialize optimizer
    processor = ImageProcessor() if not PERFORMANCE_OPTIMIZER_AVAILABLE else OptimizedImageProcessor()

    # Load and Preprocess Image
    logging.info("Step 1: Loading and Preprocessing...")
    preprocess_results = load_and_preprocess_image(str(image_path), profile_config)
    if preprocess_results is None:
        logging.error(f"Failed to load/preprocess image {image_path.name}. Skipping.")
        return {
            "image_filename": image_path.name,
            "pass_fail_status": "ERROR_LOAD_PREPROCESS",
            "processing_time_s": round(time.perf_counter() - image_start_time, 2),
            "total_defect_count": 0,
            "core_defect_count": 0,
            "cladding_defect_count": 0,
            "failure_reason_summary": "Load/preprocess failed"
        }
    
    original_bgr, original_gray, processed_image = preprocess_results

    # Resize for faster processing if image is large
    processed_resized, scale_factor = processor.resize_for_processing(processed_image)
    gray_resized, _ = processor.resize_for_processing(original_gray) if scale_factor != 1.0 else (original_gray, 1.0)

    # Locate Fiber Structure (on resized image)
    logging.info("Step 2: Locating Fiber Structure...")
    localization_data = locate_fiber_structure(processed_resized, profile_config, original_gray_image=gray_resized)
    
    if localization_data is None or "cladding_center_xy" not in localization_data:
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
    
    # Scale results back to original image coordinates if resized
    if scale_factor != 1.0:
        # Scale center coordinates
        localization_data['cladding_center_xy'] = tuple(
            int(coord / scale_factor) for coord in localization_data['cladding_center_xy']
        )
        # Scale radii
        if 'cladding_radius_px' in localization_data:
            localization_data['cladding_radius_px'] = localization_data['cladding_radius_px'] / scale_factor
        if 'core_center_xy' in localization_data:
            localization_data['core_center_xy'] = tuple(
                int(coord / scale_factor) for coord in localization_data['core_center_xy']
            )
        if 'core_radius_px' in localization_data:
            localization_data['core_radius_px'] = localization_data['core_radius_px'] / scale_factor
        # Scale ellipse parameters if present
        if 'cladding_ellipse_params' in localization_data:
            center, axes, angle = localization_data['cladding_ellipse_params']
            localization_data['cladding_ellipse_params'] = (
                tuple(int(c / scale_factor) for c in center),
                tuple(a / scale_factor for a in axes),
                angle
            )

    # Calculate scaling factor
    current_image_um_per_px = calibration_um_per_px
    if user_clad_dia_um is not None:
        detected_cladding_radius_px = localization_data.get("cladding_radius_px")
        if detected_cladding_radius_px and detected_cladding_radius_px > 0:
            detected_cladding_diameter_px = detected_cladding_radius_px * 2.0
            current_image_um_per_px = user_clad_dia_um / detected_cladding_diameter_px
            logging.info(f"Using image-specific scale for {image_path.name}: {current_image_um_per_px:.4f} µm/px")
        elif localization_data.get("cladding_ellipse_params"):
            ellipse_axes = localization_data["cladding_ellipse_params"][1]
            avg_radius_px = sum(ellipse_axes) / 2.0
            avg_diameter_px = avg_radius_px * 2.0
            current_image_um_per_px = user_clad_dia_um / avg_diameter_px
            logging.info(f"Using ellipse-based scale for {image_path.name}: {current_image_um_per_px:.4f} µm/px")

    # Generate Zone Masks (on original size)
    logging.info("Step 3: Generating Zone Masks...")
    try:
        zone_definitions_list = get_zone_definitions(fiber_type_key)  # Fixed: Only pass fiber_type_key
    except ValueError as e:
        logging.error(f"Zone definitions for fiber type '{fiber_type_key}' not found: {e}")
        return {
            "image_filename": image_path.name,
            "pass_fail_status": "ERROR_CONFIG",
            "processing_time_s": round(time.perf_counter() - image_start_time, 2),
            "total_defect_count": 0,
            "core_defect_count": 0,
            "cladding_defect_count": 0,
            "failure_reason_summary": f"Zone definitions not found for {fiber_type_key}: {e}"
        }

    zone_masks = generate_zone_masks(
        processed_image.shape, localization_data, zone_definitions_list,
        um_per_px=current_image_um_per_px,
        user_core_diameter_um=user_core_dia_um,
        user_cladding_diameter_um=user_clad_dia_um
    )

    if not zone_masks:
        logging.error(f"Failed to generate zone masks for {image_path.name}. Skipping.")
        return {
            "image_filename": image_path.name,
            "pass_fail_status": "ERROR_ZONE_MASKS",
            "processing_time_s": round(time.perf_counter() - image_start_time, 2),
            "total_defect_count": 0,
            "core_defect_count": 0,
            "cladding_defect_count": 0,
            "failure_reason_summary": "Zone mask generation failed"
        }

    # Detect Defects per Zone
    logging.info("Step 4: Detecting Defects...")
    algo_params = global_config.get("algorithm_parameters", {})
    all_detected_defects = []
    confidence_maps_all_zones = {}

    for zone_name, zone_mask in zone_masks.items():
        if np.sum(zone_mask) == 0:
            logging.debug(f"Zone '{zone_name}' is empty. Skipping defect detection.")
            continue
        
        defects_mask, confidence_map = detect_defects(
            processed_image, zone_mask, zone_name, profile_config, algo_params,
            localization_data=localization_data
        )
        confidence_maps_all_zones[zone_name] = confidence_map if confidence_map is not None else np.zeros_like(zone_mask, dtype=np.float32)
        
        # Label connected components
        num_labels, labeled_defects, stats, centroids = cv2.connectedComponentsWithStats(defects_mask.astype(np.uint8), connectivity=8)
        
        for label_id in range(1, num_labels):
            defect_mask_single = (labeled_defects == label_id).astype(np.uint8) * 255
            defect_info = {
                "zone": zone_name,
                "defect_mask": defect_mask_single,
                "area_px": stats[label_id, cv2.CC_STAT_AREA],
                "bounding_box": {
                    "x": stats[label_id, cv2.CC_STAT_LEFT],
                    "y": stats[label_id, cv2.CC_STAT_TOP],
                    "width": stats[label_id, cv2.CC_STAT_WIDTH],
                    "height": stats[label_id, cv2.CC_STAT_HEIGHT]
                },
                "centroid": (centroids[label_id][0], centroids[label_id][1])
            }
            all_detected_defects.append(defect_info)

    logging.info(f"Total defects detected across all zones: {len(all_detected_defects)}")

    # Characterize and Classify Defects
    logging.info("Step 5: Characterizing and Classifying Defects...")
    characterized_defects = characterize_and_classify_defects(
        all_detected_defects, processed_image, 
        localization_data, current_image_um_per_px, global_config
    )

    # Apply Pass/Fail Rules
    logging.info("Step 6: Applying Pass/Fail Rules...")
    overall_status, failure_reasons = apply_pass_fail_rules(
        characterized_defects, fiber_type_key
    )

    # Generate Reports
    logging.info("Step 7: Generating Reports...")

    # Create analysis_results dictionary for reporting functions
    analysis_results = {
        "characterized_defects": characterized_defects,
        "overall_status": overall_status,
        "failure_reasons": failure_reasons,
        "image_filename": image_path.name,
        "total_defect_count": len(characterized_defects),
        "core_defect_count": sum(1 for d in characterized_defects if d["zone"] == "Core"),
        "cladding_defect_count": sum(1 for d in characterized_defects if d["zone"] == "Cladding")
    }

    annotated_image_path = output_dir_image / f"{image_path.stem}_annotated.png"
    generate_annotated_image(
        original_bgr, 
        analysis_results,
        localization_data, 
        zone_masks,
        fiber_type_key,
        annotated_image_path
    )

    csv_report_path = output_dir_image / f"{image_path.stem}_report.csv"
    generate_defect_csv_report(
        analysis_results,
        csv_report_path
    )

    polar_histogram_path = output_dir_image / f"{image_path.stem}_histogram.png"
    generate_polar_defect_histogram(
        analysis_results,
        localization_data,
        zone_masks,
        fiber_type_key,
        polar_histogram_path
    )

    processing_time_s = time.perf_counter() - image_start_time
    logging.info(f"Image processing complete. Status: {overall_status}. Time: {processing_time_s:.2f}s")

    # Prepare summary
    total_defect_count = len(characterized_defects)
    summary_for_batch = {
        "image_filename": image_path.name,
        "pass_fail_status": overall_status,
        "processing_time_s": round(processing_time_s, 2),
        "total_defect_count": total_defect_count,
        "core_defect_count": sum(1 for d in characterized_defects if d["zone"] == "Core"),
        "cladding_defect_count": sum(1 for d in characterized_defects if d["zone"] == "Cladding"),
        "failure_reason_summary": "; ".join(failure_reasons) if failure_reasons else "N/A"
    }
    return summary_for_batch

def process_image_wrapper(args):
    """Wrapper for multiprocessing with performance optimization"""
    image_path, output_dir, profile_config, global_config, um_per_px, core_dia, clad_dia, fiber_type = args
    try:
        return process_single_image_optimized(
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

def process_with_progress(images_to_process, current_run_output_dir, active_profile_config, 
                         global_config, loaded_um_per_px, args_namespace):
    """Process images with progress monitoring"""
    all_image_summaries = []
    
    # Determine number of processes
    num_processes = min(cpu_count() - 1, len(images_to_process))
    num_processes = max(1, num_processes)

    if num_processes > 1 and len(images_to_process) > 1:
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
            for image_path in images_to_process
        ]
        
        # Process in parallel with progress bar
        with Pool(processes=num_processes) as pool:
            with tqdm(total=len(images_to_process), desc="Processing images") as pbar:
                for i, result in enumerate(pool.imap(process_image_wrapper, process_args)):
                    all_image_summaries.append(result)
                    pbar.set_postfix({
                        'file': result['image_filename'],
                        'status': result['pass_fail_status'],
                        'time': f'{result["processing_time_s"]:.1f}s'
                    })
                    pbar.update(1)
    else:
        # Sequential processing with progress bar
        logging.info("Using sequential processing")
        
        with tqdm(total=len(images_to_process), desc="Processing images") as pbar:
            for i, image_file_path in enumerate(images_to_process):
                start_time = time.time()
                logging.info(f"--- Starting image {i+1}/{len(images_to_process)}: {image_file_path.name} ---")
                
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
                
                elapsed = time.time() - start_time
                pbar.set_postfix({
                    'file': image_file_path.name,
                    'status': summary['pass_fail_status'],
                    'time': f'{elapsed:.1f}s'
                })
                pbar.update(1)
    
    return all_image_summaries

def execute_inspection_run(args_namespace: Any) -> None:
    """
    Core inspection logic that takes an args-like namespace object.
    This function contains the main processing flow.
    """
    # Declare intent to use global variables
    global PADIM_SPECIFIC_AVAILABLE, SEGDECNET_AVAILABLE

    # Output Directory Setup
    base_output_dir = Path(args_namespace.output_dir)
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    current_run_output_dir = base_output_dir / f"run_{run_timestamp}"
    current_run_output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration and Logging Setup
    try:
        config_file_path = str(args_namespace.config_file)
        global_config = load_config(config_file_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[CRITICAL] Failed to load configuration: {e}. Exiting.", file=sys.stderr)
        try:
            fallback_log_dir = Path(".") / "error_logs"
            fallback_log_dir.mkdir(parents=True, exist_ok=True)
            setup_logging("ERROR", True, fallback_log_dir)
            logging.critical(f"Failed to load configuration: {e}. Exiting.")
        except Exception as log_e:
            print(f"[CRITICAL] Logging setup failed during config error: {log_e}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        logging.error(f"Critical error in inspection run: {e}", exc_info=True)
        raise

    general_settings = global_config.get("general_settings", {})
    setup_logging(
        general_settings.get("log_level", "INFO"),
        general_settings.get("log_to_console", True),
        current_run_output_dir
    )

    logging.info("Inspection System Started.")
    logging.info(f"Configuration loaded from: {config_file_path}")
    logging.info(f"Results will be saved to: {current_run_output_dir}")

    # Processing Profile
    profile_name = args_namespace.profile
    active_profile_config = get_processing_profile(profile_name)  # Fixed: Only pass profile_name
    if active_profile_config is None:
        logging.critical(f"Processing profile '{profile_name}' not found in configuration. Exiting.")
        sys.exit(1)
    logging.info(f"Using processing profile: '{profile_name}'")

    # Calibration Data
    calibration_file_path = str(args_namespace.calibration_file)
    calibration_data = load_calibration_data(calibration_file_path)
    loaded_um_per_px = calibration_data.get("um_per_px") if isinstance(calibration_data, dict) else calibration_data

    if loaded_um_per_px is not None:
        logging.info(f"Calibration loaded: {loaded_um_per_px:.4f} µm/px")
    else:
        logging.info("No calibration data; using default or user-provided scale factors.")

    # Validate fiber type
    fiber_type_key = args_namespace.fiber_type
    try:
        zone_defs = get_zone_definitions(fiber_type_key)  # Fixed: Only pass fiber_type_key
        logging.info(f"Using fiber type rules: '{fiber_type_key}'")
    except ValueError as e:
        logging.critical(f"Fiber type '{fiber_type_key}' not found in zone definitions: {e}")
        sys.exit(1)

    # Model availability check
    if active_profile_config.get("defect_detection", {}).get("use_padim", False) and not PADIM_SPECIFIC_AVAILABLE:
        logging.warning("PaDiM model requested but not available. Will proceed without it.")
        active_profile_config["defect_detection"]["use_padim"] = False

    if active_profile_config.get("defect_detection", {}).get("use_segdecnet", False) and not SEGDECNET_AVAILABLE:
        logging.warning("SegDecNet model requested but not available. Will proceed without it.")
        active_profile_config["defect_detection"]["use_segdecnet"] = False

    # Input Directory and Image Discovery
    input_dir = Path(args_namespace.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logging.critical(f"Input directory does not exist or is not a directory: {input_dir}. Exiting.")
        sys.exit(1)

    supported_image_extensions = general_settings.get("supported_image_extensions", [".png", ".jpg", ".jpeg", ".bmp", ".tiff"])
    image_paths_to_process = [
        img_path for img_path in input_dir.iterdir()
        if img_path.is_file() and img_path.suffix.lower() in supported_image_extensions
    ]

    if not image_paths_to_process:
        logging.warning(f"No images found in the input directory: {input_dir}. Exiting.")
        sys.exit(0)

    logging.info(f"Found {len(image_paths_to_process)} image(s) to process.")

    # Batch Processing with Progress Monitoring
    batch_start_time = time.perf_counter()
    logging.info("--- Starting Batch Processing ---")

    all_image_summaries = process_with_progress(
        image_paths_to_process, current_run_output_dir, active_profile_config,
        global_config, loaded_um_per_px, args_namespace
    )

    # Final Summary Report
    if all_image_summaries:
        summary_df = pd.DataFrame(all_image_summaries)
        summary_report_path = current_run_output_dir / "batch_summary_report.csv"
        try:
            summary_df.to_csv(summary_report_path, index=False, encoding='utf-8')
            logging.info(f"Batch summary report saved to: {summary_report_path}")
        except Exception as e:
            logging.error(f"Failed to save batch summary report: {e}")
    else:
        logging.warning("No image summaries were generated for the batch report.")

    batch_duration = time.perf_counter() - batch_start_time
    logging.info(f"Batch Processing Complete ---")
    logging.info(f"Total images processed: {len(image_paths_to_process)}")
    logging.info(f"Total batch duration: {batch_duration:.2f} seconds.")
    logging.info(f"All reports for this run saved in: {current_run_output_dir}")

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
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Automated Fiber Optic End Face Inspection System.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing images to inspect.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where results will be saved.")
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the JSON configuration file (default: config.json).")
    parser.add_argument("--calibration_file", type=str, default="calibration.json", help="Path to the JSON calibration file (default: calibration.json).")
    parser.add_argument("--profile", type=str, default="deep_inspection", choices=["fast_scan", "deep_inspection"], help="Processing profile to use (default: deep_inspection).")
    parser.add_argument("--fiber_type", type=str, default="single_mode_pc", help="Key for fiber type specific rules, e.g., 'single_mode_pc', 'multi_mode_pc' (must match config.json).")
    parser.add_argument("--core_dia_um", type=float, default=None, help="Optional: Known core diameter in microns for this batch.")
    parser.add_argument("--clad_dia_um", type=float, default=None, help="Optional: Known cladding diameter in microns for this batch.")
    
    args = parser.parse_args()
    execute_inspection_run(args)

if __name__ == "__main__":
    main()