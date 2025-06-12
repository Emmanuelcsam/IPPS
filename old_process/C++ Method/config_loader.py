#!/usr/bin/env python3
# config_loader.py

"""
Configuration Loader Module
==========================================
This module is responsible for loading, validating, and providing access to
all operational settings from the 'config.json' file. It ensures that
the application has a centralized and validated source for its parameters.
"""

import json # Standard library for JSON parsing.
from pathlib import Path # Standard library for object-oriented path manipulation.
from typing import Dict, Any, Optional, List # Standard library for type hinting.
import logging # Standard library for logging events.

# --- Global Configuration Variable ---
# This variable will hold the loaded configuration dictionary.
# It's intended to be read-only after initial loading.
_config: Optional[Dict[str, Any]] = None # Initialize as None.

# --- Default Configuration Structure (for reference and initial setup) ---
# This provides a baseline structure if a config.json is missing or for new setups.
DEFAULT_CONFIG_STRUCTURE = {
    "general_settings": { # General operational parameters.
        "output_dir_name": "output", # Default directory for saving results.
        "log_level": "INFO", # Default logging level (e.g., DEBUG, INFO, WARNING, ERROR).
        "log_to_console": True # Whether to output logs to the console.
    },
    "processing_profiles": { # Different sets of parameters for different inspection needs.
        "fast_scan": { # Profile for quick, less intensive scans.
            "description": "Uses a minimal set of computationally inexpensive algorithms for quick checks.",
            "preprocessing": { # Parameters for image preprocessing.
                "clahe_clip_limit": 1.0, # CLAHE contrast limit.
                "clahe_tile_grid_size": [4, 4], # CLAHE grid size.
                "gaussian_blur_kernel_size": [3, 3] # Kernel size for Gaussian blur.
            },
            "localization": { # Parameters for fiber localization.
                "hough_dp": 1.5, # Inverse ratio of accumulator resolution for HoughCircles.
                "hough_min_dist_factor": 0.2, # Min distance between circle centers (factor of image smaller dim).
                "hough_param1": 50, # Upper Canny threshold for HoughCircles.
                "hough_param2": 25, # Accumulator threshold for HoughCircles.
                "hough_min_radius_factor": 0.1, # Min circle radius (factor of image smaller dim).
                "hough_max_radius_factor": 0.4 # Max circle radius (factor of image smaller dim).
            },
            "defect_detection": { # Parameters for defect detection.
                "region_algorithms": ["black_hat"], # Algorithms for region-based defects.
                "linear_algorithms": ["lei_simple"], # Algorithms for linear defects (scratches).
                "confidence_threshold": 0.6, # Min confidence score for a defect to be confirmed.
                "min_defect_area_px": 15 # Min area in pixels for a defect.
            }
        },
        "deep_inspection": { # Profile for thorough, high-accuracy scans.
            "description": "Uses the full suite of algorithms and fusion for maximum accuracy.",
            "preprocessing": {
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": [8, 8],
                "gaussian_blur_kernel_size": [5, 5]
            },
            "localization": {
                "hough_dp": 1.2,
                "hough_min_dist_factor": 0.15,
                "hough_param1": 70,
                "hough_param2": 35,
                "hough_min_radius_factor": 0.08,
                "hough_max_radius_factor": 0.45
            },
            "defect_detection": {
                "region_algorithms": ["morph_gradient", "black_hat"],
                "linear_algorithms": ["lei_advanced", "skeletonization"],
                "confidence_threshold": 0.9,
                "min_defect_area_px": 5,
                "scratch_aspect_ratio_threshold": 3.0, # Aspect ratio to classify scratches.
                "algorithm_weights": { # Weights for combining results from different algorithms.
                    "morph_gradient": 0.4,
                    "black_hat": 0.6,
                    "lei_advanced": 0.7,
                    "skeletonization": 0.3
                }
            }
        }
    },
    "algorithm_parameters": { # Specific parameters for individual algorithms.
        "flat_field_image_path": None, # Path to flat-field image for illumination correction (null if not used).
        "morph_gradient_kernel_size": [5, 5], # Kernel for morphological gradient.
        "black_hat_kernel_size": [11, 11], # Kernel for black-hat transform.
        "lei_kernel_lengths": [11, 17], # List of kernel lengths for LEI.
        "lei_angle_step_deg": 15, # Angular step for LEI.
        "sobel_scharr_ksize": 3, # Kernel size for Sobel/Scharr gradient.
        "skeletonization_dilation_kernel_size": [3, 3] # Dilation kernel after skeletonization.
    },
    "zone_definitions_iec61300_3_35": { # Zone definitions based on IEC standard.
        "single_mode_pc": [ # Rules for single-mode PC connectors.
            {"name": "Core", "r_min_factor": 0.0, "r_max_factor_core_relative": 1.0, "color_bgr": [255,0,0], "pass_fail_rules": {"max_scratches": 0, "max_defects": 0, "max_defect_size_um": 3}},
            {"name": "Cladding", "r_min_factor_cladding_relative": 0.0, "r_max_factor_cladding_relative": 1.0, "color_bgr": [0,255,0], "pass_fail_rules": {"max_scratches": 5, "max_scratches_gt_5um": 0, "max_defects": 5, "max_defect_size_um": 10}},
            {"name": "Adhesive", "r_min_factor_cladding_relative": 1.0, "r_max_factor_cladding_relative": 1.15, "color_bgr": [0,255,255], "pass_fail_rules": {"max_defects": "unlimited", "max_defect_size_um": 50}},
            {"name": "Contact", "r_min_factor_cladding_relative": 1.15, "r_max_factor_cladding_relative": 2.0, "color_bgr": [255,0,255], "pass_fail_rules": {"max_defects": "unlimited", "max_defect_size_um": 100}}
        ],
        "multi_mode_pc": [ # Rules for multi-mode PC connectors.
            {"name": "Core", "r_min_factor": 0.0, "r_max_factor_core_relative": 1.0, "color_bgr": [255,100,100], "pass_fail_rules": {"max_scratches": 1, "max_scratch_length_um": 10, "max_defects": 3, "max_defect_size_um": 5}},
            {"name": "Cladding", "r_min_factor_cladding_relative": 0.0, "r_max_factor_cladding_relative": 1.0, "color_bgr": [100,255,100], "pass_fail_rules": {"max_scratches": "unlimited", "max_defects": "unlimited", "max_defect_size_um": 20}},
            {"name": "Adhesive", "r_min_factor_cladding_relative": 1.0, "r_max_factor_cladding_relative": 1.15, "color_bgr": [100,255,255], "pass_fail_rules": {"max_defects": "unlimited"}},
            {"name": "Contact", "r_min_factor_cladding_relative": 1.15, "r_max_factor_cladding_relative": 2.0, "color_bgr": [255,100,255], "pass_fail_rules": {"max_defects": "unlimited"}}
        ]
        # Add other fiber types (e.g., APC) as needed.
    },
    "reporting": { # Parameters for generating reports.
        "annotated_image_dpi": 150, # DPI for saved annotated images.
        "defect_label_font_scale": 0.4, # Font scale for defect labels.
        "defect_label_thickness": 1, # Thickness for defect labels.
        "pass_fail_stamp_font_scale": 1.5, # Font scale for PASS/FAIL stamp.
        "pass_fail_stamp_thickness": 2 # Thickness for PASS/FAIL stamp.
    }
}

def load_config(config_path_str: str = "config.json") -> Dict[str, Any]:
    """
    Loads the configuration from a JSON file.
    If the file doesn't exist, it creates one with a default structure.
    It also performs basic validation on the loaded configuration.

    Args:
        config_path_str: The path to the configuration JSON file.

    Returns:
        A dictionary containing the loaded configuration.

    Raises:
        FileNotFoundError: If the config file cannot be created.
        ValueError: If the config file contains invalid JSON or fails validation.
    """
    global _config # Allow modification of the global _config variable.
    config_path = Path(config_path_str) # Convert string path to Path object.

    if not config_path.exists(): # Check if the config file exists.
        logging.warning(f"Configuration file '{config_path}' not found. Creating with default structure.")
        try:
            with open(config_path, "w", encoding="utf-8") as f: # Open file for writing.
                json.dump(DEFAULT_CONFIG_STRUCTURE, f, indent=2) # Write default config with indentation.
            logging.info(f"Default configuration file created at '{config_path}'. Please review and customize.")
            _config = DEFAULT_CONFIG_STRUCTURE # Use default config.
        except IOError as e: # Handle potential I/O errors during file creation.
            logging.error(f"Could not create default configuration file: {e}")
            raise FileNotFoundError(f"Failed to create config file at {config_path}") from e # Raise error.
    else: # If config file exists.
        try:
            with open(config_path, "r", encoding="utf-8") as f: # Open file for reading.
                loaded_data = json.load(f) # Load JSON data.
            _config = loaded_data # Store loaded data.
            logging.info(f"Configuration loaded successfully from '{config_path}'.")
        except json.JSONDecodeError as e: # Handle JSON decoding errors.
            logging.error(f"Invalid JSON in configuration file '{config_path}': {e}")
            raise ValueError(f"Configuration file '{config_path}' contains invalid JSON.") from e # Raise error.
        except IOError as e: # Handle I/O errors during file reading.
            logging.error(f"Could not read configuration file '{config_path}': {e}")
            raise FileNotFoundError(f"Failed to read config file at {config_path}") from e # Raise error.

    _validate_config(_config) # Validate the loaded or default configuration.
    return _config # Return the configuration.

def _validate_config(config_data: Dict[str, Any]):
    """
    Performs basic validation of the loaded configuration structure.
    Logs warnings or raises errors for critical missing parts.

    Args:
        config_data: The configuration dictionary to validate.

    Raises:
        ValueError: If critical configuration sections are missing.
    """
    # Example basic validation: Check for presence of top-level keys.
    required_top_level_keys = [
        "general_settings",
        "processing_profiles",
        "algorithm_parameters",
        "zone_definitions_iec61300_3_35",
        "reporting"
    ]
    for key in required_top_level_keys: # Iterate through required keys.
        if key not in config_data: # Check if key is missing.
            logging.error(f"Critical configuration section '{key}' is missing.")
            raise ValueError(f"Missing critical configuration section: '{key}'") # Raise error.

    # Validate processing profiles.
    if "processing_profiles" in config_data: # Check if profiles exist.
        if not config_data["processing_profiles"]: # Check if profiles are empty.
            logging.warning("No processing profiles defined in configuration.")
        for profile_name, profile_data in config_data["processing_profiles"].items(): # Iterate profiles.
            if "preprocessing" not in profile_data or "localization" not in profile_data or "defect_detection" not in profile_data:
                logging.error(f"Profile '{profile_name}' is missing one or more required sections (preprocessing, localization, defect_detection).")
                raise ValueError(f"Incomplete profile definition for '{profile_name}'.") # Raise error.
    
    # Validate zone definitions
    if "zone_definitions_iec61300_3_35" in config_data: # Check if zone definitions exist.
        if not config_data["zone_definitions_iec61300_3_35"]: # Check if zone definitions are empty.
            logging.warning("No zone definitions (IEC61300-3-35) found in configuration.")
        for fiber_type, zones in config_data["zone_definitions_iec61300_3_35"].items(): # Iterate fiber types.
            if not isinstance(zones, list): # Check if zones is a list.
                logging.error(f"Zone definitions for fiber type '{fiber_type}' must be a list.")
                raise ValueError(f"Invalid format for zone definitions of '{fiber_type}'.") # Raise error.
            for zone_def in zones: # Iterate zone definitions.
                if not all(k in zone_def for k in ["name", "color_bgr", "pass_fail_rules"]):
                    logging.error(f"Zone definition under '{fiber_type}' is missing required keys (name, color_bgr, pass_fail_rules). Found: {zone_def}")
                    raise ValueError(f"Incomplete zone definition for '{fiber_type}'.") # Raise error.

    logging.info("Configuration validation passed.") # Log successful validation.

def get_config() -> Dict[str, Any]:
    """
    Returns the loaded configuration.
    If the configuration hasn't been loaded yet, it attempts to load it first.

    Returns:
        The global configuration dictionary.

    Raises:
        RuntimeError: If the configuration has not been loaded and cannot be loaded.
    """
    if _config is None: # Check if config is not loaded.
        logging.info("Configuration not yet loaded. Attempting to load now...")
        try:
            load_config() # Attempt to load config.
        except Exception as e: # Handle any errors during loading.
            logging.critical(f"Failed to load configuration during get_config(): {e}")
            raise RuntimeError("Configuration could not be loaded.") from e # Raise runtime error.
    return _config # Return the (now loaded) config.

def get_processing_profile(profile_name: str) -> Dict[str, Any]:
    """
    Retrieves a specific processing profile from the configuration.

    Args:
        profile_name: The name of the processing profile to retrieve.

    Returns:
        A dictionary containing the parameters for the specified profile.

    Raises:
        ValueError: If the profile_name is not found in the configuration.
    """
    config = get_config() # Get the global configuration.
    profiles = config.get("processing_profiles", {}) # Get processing profiles.
    if profile_name not in profiles: # Check if requested profile exists.
        logging.error(f"Processing profile '{profile_name}' not found in configuration.")
        raise ValueError(f"Unknown processing profile: '{profile_name}'") # Raise error.
    logging.debug(f"Retrieved processing profile: '{profile_name}'.")
    return profiles[profile_name] # Return the profile.

def get_zone_definitions(fiber_type_key: str = "single_mode_pc") -> List[Dict[str, Any]]:
    """
    Retrieves the zone definitions for a specific fiber type.

    Args:
        fiber_type_key: The key for the fiber type (e.g., "single_mode_pc")
                        as defined in config.json.

    Returns:
        A list of zone definition dictionaries.

    Raises:
        ValueError: If the fiber_type_key is not found.
    """
    config = get_config() # Get the global configuration.
    all_zone_defs = config.get("zone_definitions_iec61300_3_35", {}) # Get all zone definitions.
    if fiber_type_key not in all_zone_defs: # Check if requested fiber type exists.
        logging.error(f"Zone definitions for fiber type '{fiber_type_key}' not found.")
        # Fallback to the first available definition or an empty list if none exist.
        if all_zone_defs: # If other definitions exist.
            fallback_key = list(all_zone_defs.keys())[0] # Get the first available key.
            logging.warning(f"Falling back to first available zone definition: '{fallback_key}'.")
            return all_zone_defs[fallback_key] # Return fallback.
        else: # If no definitions exist.
            raise ValueError(f"No zone definitions found in config for '{fiber_type_key}' or any other type.")
    logging.debug(f"Retrieved zone definitions for fiber type: '{fiber_type_key}'.")
    return all_zone_defs[fiber_type_key] # Return definitions for specified fiber type.


if __name__ == "__main__":
    # This block is for testing the config_loader module independently.
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s') # Basic logging config.

    # --- Test Case 1: Load existing or create default config.json ---
    print("\n--- Test Case 1: Load or Create Config ---")
    try:
        config_data = load_config() # Load/create config.
        # print("Loaded Configuration:")
        # print(json.dumps(config_data, indent=2)) # Print loaded config.
    except Exception as e: # Handle errors.
        print(f"Error in Test Case 1: {e}")

    # --- Test Case 2: Get the loaded configuration ---
    print("\n--- Test Case 2: Get Config (should be loaded now) ---")
    try:
        current_config = get_config() # Get config.
        if current_config: # Check if config is loaded.
            print(f"Successfully retrieved configuration. Output directory name: {current_config['general_settings']['output_dir_name']}")
    except Exception as e: # Handle errors.
        print(f"Error in Test Case 2: {e}")

    # --- Test Case 3: Get a specific processing profile ---
    print("\n--- Test Case 3: Get Processing Profile ---")
    try:
        deep_inspection_profile = get_processing_profile("deep_inspection") # Get deep inspection profile.
        print("Deep Inspection Profile Preprocessing Settings:")
        print(json.dumps(deep_inspection_profile.get("preprocessing"), indent=2)) # Print preprocessing settings.
    except ValueError as e: # Handle value errors.
        print(f"Error in Test Case 3 (deep_inspection): {e}")
    
    try:
        fast_scan_profile = get_processing_profile("fast_scan") # Get fast scan profile.
        print("\nFast Scan Profile Defect Detection Settings:")
        print(json.dumps(fast_scan_profile.get("defect_detection"), indent=2)) # Print defect detection settings.
    except ValueError as e: # Handle value errors.
        print(f"Error in Test Case 3 (fast_scan): {e}")

    try:
        # Test getting a non-existent profile
        get_processing_profile("non_existent_profile") # Attempt to get non-existent profile.
    except ValueError as e: # Handle value errors.
        print(f"\nSuccessfully caught error for non-existent profile: {e}")

    # --- Test Case 4: Get zone definitions ---
    print("\n--- Test Case 4: Get Zone Definitions ---")
    try:
        sm_zones = get_zone_definitions("single_mode_pc") # Get single mode zones.
        print(f"\nSingle-Mode PC Zone Definitions (First zone name): {sm_zones[0]['name']}")
        # print(json.dumps(sm_zones, indent=2))
        
        mm_zones = get_zone_definitions("multi_mode_pc") # Get multi mode zones.
        print(f"\nMulti-Mode PC Zone Definitions (First zone name): {mm_zones[0]['name']}")
    except ValueError as e: # Handle value errors.
        print(f"Error in Test Case 4: {e}")

    try:
        # Test getting non-existent zone definitions
        get_zone_definitions("non_existent_fiber_type") # Attempt to get non-existent type.
    except ValueError as e: # Handle value errors.
        print(f"\nSuccessfully caught error for non-existent fiber type for zones: {e}")

    # --- Test Case 5: Simulate missing config file and then corrupted file ---
    print("\n--- Test Case 5: Simulate Missing/Corrupt Config File ---")
    test_config_path = Path("temp_config_test.json") # Define temporary config path.
    if test_config_path.exists(): test_config_path.unlink() # Delete if exists.

    print("\nAttempting to load with missing file (should create default):")
    try:
        load_config(str(test_config_path)) # Load with missing file.
        print(f"'{test_config_path}' created with default structure.")
    except Exception as e: # Handle errors.
        print(f"Error: {e}")

    print("\nSimulating corrupt JSON file:")
    try:
        with open(test_config_path, "w", encoding="utf-8") as f: # Open for writing.
            f.write("{invalid_json: ") # Write invalid JSON.
        load_config(str(test_config_path)) # Attempt to load corrupt file.
    except ValueError as e: # Handle value errors.
        print(f"Successfully caught error for corrupt JSON: {e}")
    finally: # Cleanup.
        if test_config_path.exists(): test_config_path.unlink() # Delete temp file.
