#!/usr/bin/env python3
# config_loader.py


"""
Configuration Loader Module
==========================================
This module is responsible for loading, validating, and providing access to
all operational settings from the 'config.json' file. It ensures that
the application has a centralized and validated source for its parameters.
"""

# This is a docstring for the module, providing a high-level overview of its purpose.

import json  # Imports the standard JSON library, essential for parsing and creating JSON files.
from pathlib import Path  # Imports the Path object from the pathlib module for modern, object-oriented filesystem path manipulation.
from typing import Dict, Any, Optional, List  # Imports type hinting classes from the 'typing' module to improve code clarity and allow for static analysis.
import logging  # Imports the standard logging module to record events, warnings, and errors during the script's execution.

# --- Global Configuration Variable ---
# This section defines a global variable to hold the configuration once it's loaded.
# It's intended to be read-only after initial loading to ensure consistency across the application.
_config: Optional[Dict[str, Any]] = None  # Declares a global variable '_config' which will store the configuration dictionary. It's initialized to None.

# --- Default Configuration Structure (for reference and initial setup) ---
# This dictionary provides a complete, well-structured baseline configuration.
# It's used to create a 'config.json' file if one doesn't exist, ensuring the program can run out-of-the-box.
DEFAULT_CONFIG_STRUCTURE = {
    "general_settings": {  # Contains general operational parameters for the application.
        "output_dir_name": "output",  # Specifies the default directory name where all results will be saved.
        "log_level": "INFO",  # Sets the default logging level (e.g., DEBUG, INFO, WARNING, ERROR) for the application's logger.
        "log_to_console": True  # A boolean flag to determine whether log messages should be printed to the console.
    },
    "processing_profiles": {  # Defines different sets of parameters for various inspection scenarios, allowing for flexibility.
        "fast_scan": {  # A profile optimized for speed, using less computationally intensive algorithms.
            "description": "Uses a minimal set of computationally inexpensive algorithms for quick checks.",  # Describes the purpose of this profile.
            "preprocessing": {  # Contains parameters for the initial image preprocessing steps.
                "clahe_clip_limit": 1.0,  # Sets the contrast limit for CLAHE (Contrast Limited Adaptive Histogram Equalization).
                "clahe_tile_grid_size": [4, 4],  # Defines the grid size for CLAHE, affecting local contrast normalization.
                "gaussian_blur_kernel_size": [3, 3]  # Specifies the kernel size for the Gaussian blur filter to reduce noise.
            },
            "localization": {  # Contains parameters for locating the fiber structure in the image.
                "hough_dp": 1.5,  # Sets the inverse ratio of the accumulator resolution for the HoughCircles function.
                "hough_min_dist_factor": 0.2,  # Defines the minimum distance between circle centers as a factor of the image's smaller dimension.
                "hough_param1": 50,  # Sets the upper threshold for the internal Canny edge detector in HoughCircles.
                "hough_param2": 25,  # Sets the accumulator threshold for circle detection; lower values detect more (potentially false) circles.
                "hough_min_radius_factor": 0.1,  # Sets the minimum circle radius as a factor of the image's smaller dimension.
                "hough_max_radius_factor": 0.4  # Sets the maximum circle radius as a factor of the image's smaller dimension.
            },
            "defect_detection": {  # Contains parameters for the defect detection algorithms.
                "region_algorithms": ["black_hat"],  # A list of algorithms to be used for detecting region-based defects (like pits).
                "linear_algorithms": ["lei_simple"],  # A list of algorithms to be used for detecting linear defects (scratches).
                "confidence_threshold": 0.6,  # The minimum confidence score required to confirm a potential defect.
                "min_defect_area_px": 15  # The minimum area in pixels for a detected anomaly to be considered a defect.
            }
        },
        "deep_inspection": {  # A profile optimized for accuracy, using a full suite of algorithms.
            "description": "Uses the full suite of algorithms and fusion for maximum accuracy.",  # Describes the purpose of this profile.
            "preprocessing": {  # Preprocessing settings for the deep inspection profile.
                "clahe_clip_limit": 2.0,  # A higher clip limit for more aggressive contrast enhancement.
                "clahe_tile_grid_size": [8, 8],  # A larger grid size for more localized contrast enhancement.
                "gaussian_blur_kernel_size": [5, 5]  # A larger kernel for more aggressive noise smoothing.
            },
            "localization": {  # Localization settings for the deep inspection profile.
                "hough_dp": 1.2,  # A different accumulator resolution for potentially more accurate circle detection.
                "hough_min_dist_factor": 0.15,  # A smaller minimum distance factor for detecting closely spaced circles.
                "hough_param1": 70,  # A higher Canny threshold to focus on stronger edges.
                "hough_param2": 35,  # A higher accumulator threshold for higher confidence in detected circles.
                "hough_min_radius_factor": 0.08,  # A smaller minimum radius factor.
                "hough_max_radius_factor": 0.45  # A larger maximum radius factor.
            },
            "defect_detection": {  # Defect detection settings for the deep inspection profile.
                "region_algorithms": ["morph_gradient", "black_hat"],  # Uses multiple algorithms for region defects to improve accuracy through fusion.
                "linear_algorithms": ["lei_advanced", "skeletonization"],  # Uses multiple advanced algorithms for scratch detection.
                "confidence_threshold": 0.9,  # A high confidence threshold, requiring strong evidence from multiple algorithms.
                "min_defect_area_px": 5,  # A smaller minimum defect area to detect finer imperfections.
                "scratch_aspect_ratio_threshold": 3.0,  # The aspect ratio (length/width) above which a defect is classified as a scratch.
                "algorithm_weights": {  # Defines weights for each algorithm's contribution to the final confidence score.
                    "morph_gradient": 0.4,  # Weight for the morphological gradient algorithm.
                    "black_hat": 0.6,  # Weight for the black-hat transform algorithm.
                    "lei_advanced": 0.7,  # Weight for the advanced LEI scratch detection algorithm.
                    "skeletonization": 0.3  # Weight for the skeletonization-based scratch detection algorithm.
                }
            }
        }
    },
    "algorithm_parameters": {  # Defines specific, shared parameters for individual algorithms across all profiles.
        "flat_field_image_path": None,  # Path to an optional flat-field image for correcting uneven illumination (null if not used).
        "morph_gradient_kernel_size": [5, 5],  # The kernel size for the morphological gradient operation.
        "black_hat_kernel_size": [11, 11],  # The kernel size for the black-hat transform, should be larger than the defects of interest.
        "lei_kernel_lengths": [11, 17],  # A list of different kernel lengths to be used by the LEI algorithm to find scratches of varying lengths.
        "lei_angle_step_deg": 15,  # The angular step (in degrees) for rotating the LEI kernel to search for scratches at all orientations.
        "sobel_scharr_ksize": 3,  # The kernel size for the Sobel or Scharr gradient filters used in edge detection.
        "skeletonization_dilation_kernel_size": [3, 3]  # The kernel size for dilating the result after skeletonization to make faint lines more robust.
    },
    "zone_definitions_iec61300_3_35": {  # Defines inspection zones and pass/fail criteria based on the IEC 61300-3-35 standard.
        "single_mode_pc": [  # A set of rules specifically for single-mode PC (Physical Contact) connectors.
            {"name": "Core", "r_min_factor": 0.0, "r_max_factor_core_relative": 1.0, "color_bgr": [255,0,0], "pass_fail_rules": {"max_scratches": 0, "max_defects": 0, "max_defect_size_um": 3}}, # Defines the Core zone, its color for annotation, and its strict pass/fail rules.
            {"name": "Cladding", "r_min_factor_cladding_relative": 0.0, "r_max_factor_cladding_relative": 1.0, "color_bgr": [0,255,0], "pass_fail_rules": {"max_scratches": 5, "max_scratches_gt_5um": 0, "max_defects": 5, "max_defect_size_um": 10}}, # Defines the Cladding zone and its rules.
            {"name": "Adhesive", "r_min_factor_cladding_relative": 1.0, "r_max_factor_cladding_relative": 1.15, "color_bgr": [0,255,255], "pass_fail_rules": {"max_defects": "unlimited", "max_defect_size_um": 50}}, # Defines the Adhesive zone with more lenient rules.
            {"name": "Contact", "r_min_factor_cladding_relative": 1.15, "r_max_factor_cladding_relative": 2.0, "color_bgr": [255,0,255], "pass_fail_rules": {"max_defects": "unlimited", "max_defect_size_um": 100}} # Defines the outer Contact zone with the most lenient rules.
        ],
        "multi_mode_pc": [  # A set of rules specifically for multi-mode PC connectors, which typically have less strict requirements.
            {"name": "Core", "r_min_factor": 0.0, "r_max_factor_core_relative": 1.0, "color_bgr": [255,100,100], "pass_fail_rules": {"max_scratches": 1, "max_scratch_length_um": 10, "max_defects": 3, "max_defect_size_um": 5}}, # Defines the Core zone for multi-mode fibers.
            {"name": "Cladding", "r_min_factor_cladding_relative": 0.0, "r_max_factor_cladding_relative": 1.0, "color_bgr": [100,255,100], "pass_fail_rules": {"max_scratches": "unlimited", "max_defects": "unlimited", "max_defect_size_um": 20}}, # Defines the Cladding zone for multi-mode fibers.
            {"name": "Adhesive", "r_min_factor_cladding_relative": 1.0, "r_max_factor_cladding_relative": 1.15, "color_bgr": [100,255,255], "pass_fail_rules": {"max_defects": "unlimited"}}, # Defines the Adhesive zone for multi-mode fibers.
            {"name": "Contact", "r_min_factor_cladding_relative": 1.15, "r_max_factor_cladding_relative": 2.0, "color_bgr": [255,100,255], "pass_fail_rules": {"max_defects": "unlimited"}} # Defines the Contact zone for multi-mode fibers.
        ]
        # This structure allows adding other fiber types (e.g., APC) in the future without changing the code.
    },
    "reporting": {  # Contains parameters that control the generation of output reports and visualizations.
        "annotated_image_dpi": 150,  # Sets the DPI (dots per inch) for the saved annotated output images.
        "defect_label_font_scale": 0.4,  # Controls the font size for labels placed on detected defects in the annotated image.
        "defect_label_thickness": 1,  # Controls the thickness of the text for defect labels.
        "pass_fail_stamp_font_scale": 1.5,  # Controls the font size for the large "PASS" or "FAIL" stamp on the annotated image.
        "pass_fail_stamp_thickness": 2  # Controls the thickness of the text for the "PASS" or "FAIL" stamp.
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
    global _config  # Declares that this function intends to modify the global '_config' variable.
    config_path = Path(config_path_str)  # Converts the input string path into a more robust Path object.

    if not config_path.exists():  # Checks if the configuration file does not exist at the specified path.
        logging.warning(f"Configuration file '{config_path}' not found. Creating with default structure.")  # Logs a warning that a default file will be created.
        try:  # Begins a try block to handle potential errors during file creation.
            with open(config_path, "w", encoding="utf-8") as f:  # Opens the file in write mode ('w') with UTF-8 encoding.
                json.dump(DEFAULT_CONFIG_STRUCTURE, f, indent=2)  # Writes the default config dictionary to the file, formatted with an indent of 2 spaces for readability.
            logging.info(f"Default configuration file created at '{config_path}'. Please review and customize.")  # Informs the user that the file has been created.
            _config = DEFAULT_CONFIG_STRUCTURE  # Assigns the default structure to the global config variable for the current run.
        except IOError as e:  # Catches input/output errors that might occur if the file cannot be written (e.g., permissions issue).
            logging.error(f"Could not create default configuration file: {e}")  # Logs the specific I/O error.
            raise FileNotFoundError(f"Failed to create config file at {config_path}") from e  # Raises a FileNotFoundError to halt the program, as it cannot proceed without a config.
    else:  # This block executes if the configuration file already exists.
        try:  # Begins a try block to handle potential errors during file reading and parsing.
            with open(config_path, "r", encoding="utf-8") as f:  # Opens the existing file in read mode ('r').
                loaded_data = json.load(f)  # Parses the JSON content of the file into a Python dictionary.
            _config = loaded_data  # Assigns the loaded data to the global config variable.
            logging.info(f"Configuration loaded successfully from '{config_path}'.")  # Logs that the configuration was loaded successfully.
        except json.JSONDecodeError as e:  # Catches errors if the file content is not valid JSON.
            logging.error(f"Invalid JSON in configuration file '{config_path}': {e}")  # Logs an error indicating the file is corrupt or malformed.
            raise ValueError(f"Configuration file '{config_path}' contains invalid JSON.") from e  # Raises a ValueError, as the program cannot interpret the corrupt config.
        except IOError as e:  # Catches input/output errors that might occur during reading.
            logging.error(f"Could not read configuration file '{config_path}': {e}")  # Logs the specific read error.
            raise FileNotFoundError(f"Failed to read config file at {config_path}") from e  # Raises a FileNotFoundError if the file can't be read.

    _validate_config(_config)  # Calls a helper function to validate the structure and content of the loaded configuration.
    return _config  # Returns the final, validated configuration dictionary.

def _validate_config(config_data: Dict[str, Any]):
    """
    Performs basic validation of the loaded configuration structure.
    Logs warnings or raises errors for critical missing parts.

    Args:
        config_data: The configuration dictionary to validate.

    Raises:
        ValueError: If critical configuration sections are missing.
    """
    # This is a "private" helper function (by convention, starting with an underscore) to ensure the config is usable.
    required_top_level_keys = [  # Defines a list of essential top-level keys that must exist in the config file.
        "general_settings",
        "processing_profiles",
        "algorithm_parameters",
        "zone_definitions_iec61300_3_35",
        "reporting"
    ]
    for key in required_top_level_keys:  # Iterates through each of the required keys.
        if key not in config_data:  # Checks if a required key is missing from the configuration dictionary.
            logging.error(f"Critical configuration section '{key}' is missing.")  # Logs a critical error.
            raise ValueError(f"Missing critical configuration section: '{key}'")  # Raises a ValueError to stop the program, as it cannot function correctly.

    # This block validates the structure of the 'processing_profiles' section.
    if "processing_profiles" in config_data:  # Checks if the 'processing_profiles' key exists.
        if not config_data["processing_profiles"]:  # Checks if the dictionary for profiles is empty.
            logging.warning("No processing profiles defined in configuration.")  # Issues a warning if no profiles are defined.
        for profile_name, profile_data in config_data["processing_profiles"].items():  # Iterates through each profile (e.g., 'fast_scan', 'deep_inspection').
            if "preprocessing" not in profile_data or "localization" not in profile_data or "defect_detection" not in profile_data: # Ensures each profile has the necessary sub-sections.
                logging.error(f"Profile '{profile_name}' is missing one or more required sections (preprocessing, localization, defect_detection).") # Logs an error if a section is missing.
                raise ValueError(f"Incomplete profile definition for '{profile_name}'.")  # Raises a ValueError for an incomplete profile.
    
    # This block validates the structure of the zone definitions.
    if "zone_definitions_iec61300_3_35" in config_data:  # Checks if the 'zone_definitions' key exists.
        if not config_data["zone_definitions_iec61300_3_35"]:  # Checks if the dictionary for zone definitions is empty.
            logging.warning("No zone definitions (IEC61300-3-35) found in configuration.")  # Issues a warning if no definitions are found.
        for fiber_type, zones in config_data["zone_definitions_iec61300_3_35"].items():  # Iterates through each fiber type (e.g., 'single_mode_pc').
            if not isinstance(zones, list):  # Verifies that the zones for each fiber type are defined in a list.
                logging.error(f"Zone definitions for fiber type '{fiber_type}' must be a list.")  # Logs an error for incorrect formatting.
                raise ValueError(f"Invalid format for zone definitions of '{fiber_type}'.")  # Raises a ValueError for incorrect data type.
            for zone_def in zones:  # Iterates through each individual zone definition within the list.
                if not all(k in zone_def for k in ["name", "color_bgr", "pass_fail_rules"]): # Checks if each zone dictionary has the mandatory keys.
                    logging.error(f"Zone definition under '{fiber_type}' is missing required keys (name, color_bgr, pass_fail_rules). Found: {zone_def}") # Logs an error with the problematic definition.
                    raise ValueError(f"Incomplete zone definition for '{fiber_type}'.")  # Raises a ValueError for an incomplete zone definition.

    logging.info("Configuration validation passed.")  # Logs a message indicating that the configuration has been successfully validated.

def get_config() -> Dict[str, Any]:
    """
    Returns the loaded configuration.
    If the configuration hasn't been loaded yet, it attempts to load it first.

    Returns:
        The global configuration dictionary.

    Raises:
        RuntimeError: If the configuration has not been loaded and cannot be loaded.
    """
    if _config is None:  # Checks if the global '_config' variable has not been populated yet.
        logging.info("Configuration not yet loaded. Attempting to load now...")  # Logs that a lazy load is being triggered.
        try:  # Begins a try block to handle any exceptions during the on-demand loading.
            load_config()  # Calls the main loading function to populate the global '_config' variable.
        except Exception as e:  # Catches any exception that might occur during loading.
            logging.critical(f"Failed to load configuration during get_config(): {e}")  # Logs a critical error, as the program cannot proceed.
            raise RuntimeError("Configuration could not be loaded.") from e  # Raises a RuntimeError, wrapping the original exception.
    return _config  # Returns the global configuration dictionary, which is now guaranteed to be loaded.

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
    config = get_config()  # Ensures the configuration is loaded and gets the entire dictionary.
    profiles = config.get("processing_profiles", {})  # Safely retrieves the 'processing_profiles' dictionary, returning an empty dict if it doesn't exist.
    if profile_name not in profiles:  # Checks if the requested profile name is not a key in the profiles dictionary.
        logging.error(f"Processing profile '{profile_name}' not found in configuration.")  # Logs an error indicating the profile is missing.
        raise ValueError(f"Unknown processing profile: '{profile_name}'")  # Raises a ValueError because the requested profile cannot be found.
    logging.debug(f"Retrieved processing profile: '{profile_name}'.")  # Logs a debug message confirming which profile was retrieved.
    return profiles[profile_name]  # Returns the dictionary associated with the requested profile name.

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
    config = get_config()  # Ensures the configuration is loaded and gets the entire dictionary.
    all_zone_defs = config.get("zone_definitions_iec61300_3_35", {})  # Safely retrieves the dictionary of all zone definitions.
    if fiber_type_key not in all_zone_defs:  # Checks if the specified fiber type is not defined in the configuration.
        logging.error(f"Zone definitions for fiber type '{fiber_type_key}' not found.")  # Logs an error about the missing definition.
        # This block implements a fallback mechanism to prevent the program from crashing if a specific definition is missing.
        if all_zone_defs:  # Checks if there are any other zone definitions available at all.
            fallback_key = list(all_zone_defs.keys())[0]  # Gets the name of the first available fiber type definition to use as a fallback.
            logging.warning(f"Falling back to first available zone definition: '{fallback_key}'.")  # Informs the user that a fallback is being used.
            return all_zone_defs[fallback_key]  # Returns the list of zones for the fallback fiber type.
        else:  # This block executes if there are no zone definitions at all.
            raise ValueError(f"No zone definitions found in config for '{fiber_type_key}' or any other type.")  # Raises a ValueError as there's no data to return.
    logging.debug(f"Retrieved zone definitions for fiber type: '{fiber_type_key}'.")  # Logs a debug message indicating successful retrieval.
    return all_zone_defs[fiber_type_key]  # Returns the list of zone definitions for the specified fiber type.


if __name__ == "__main__":
    # This block executes only when the script is run directly (e.g., 'python config_loader.py'), not when it's imported.
    # It's used for testing the module's functionality in isolation.
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')  # Configures basic logging for the test run.

    # --- Test Case 1: Load existing or create default config.json ---
    print("\n--- Test Case 1: Load or Create Config ---")  # Prints a header for the test case.
    try:  # Starts a try block to catch any errors during the test.
        config_data = load_config()  # Calls the main function to either load an existing config or create a new one.
        # print("Loaded Configuration:")
        # print(json.dumps(config_data, indent=2)) # This commented line would print the entire loaded config for inspection.
    except Exception as e:  # Catches any exceptions that occur.
        print(f"Error in Test Case 1: {e}")  # Prints the error message.

    # --- Test Case 2: Get the loaded configuration ---
    print("\n--- Test Case 2: Get Config (should be loaded now) ---")  # Prints a header for the test case.
    try:  # Starts a try block.
        current_config = get_config()  # Calls the getter function, which should return the already-loaded config.
        if current_config:  # Checks if the returned config is not None.
            print(f"Successfully retrieved configuration. Output directory name: {current_config['general_settings']['output_dir_name']}") # Prints a value from the config to confirm it was retrieved correctly.
    except Exception as e:  # Catches any exceptions.
        print(f"Error in Test Case 2: {e}")  # Prints the error message.

    # --- Test Case 3: Get a specific processing profile ---
    print("\n--- Test Case 3: Get Processing Profile ---")  # Prints a header for the test case.
    try:  # Starts a try block.
        deep_inspection_profile = get_processing_profile("deep_inspection")  # Attempts to retrieve the 'deep_inspection' profile.
        print("Deep Inspection Profile Preprocessing Settings:")  # Prints a descriptive header.
        print(json.dumps(deep_inspection_profile.get("preprocessing"), indent=2))  # Prints a part of the profile to verify correctness.
    except ValueError as e:  # Catches a ValueError if the profile doesn't exist.
        print(f"Error in Test Case 3 (deep_inspection): {e}")  # Prints the error.
    
    try:  # Starts another try block for a different profile.
        fast_scan_profile = get_processing_profile("fast_scan")  # Attempts to retrieve the 'fast_scan' profile.
        print("\nFast Scan Profile Defect Detection Settings:")  # Prints a descriptive header.
        print(json.dumps(fast_scan_profile.get("defect_detection"), indent=2))  # Prints a part of the profile.
    except ValueError as e:  # Catches a ValueError if the profile doesn't exist.
        print(f"Error in Test Case 3 (fast_scan): {e}")  # Prints the error.

    try:  # This block tests the error handling for a non-existent profile.
        # Test getting a non-existent profile
        get_processing_profile("non_existent_profile")  # This call is expected to raise a ValueError.
    except ValueError as e:  # Catches the expected ValueError.
        print(f"\nSuccessfully caught error for non-existent profile: {e}")  # Prints a success message confirming the error was handled correctly.

    # --- Test Case 4: Get zone definitions ---
    print("\n--- Test Case 4: Get Zone Definitions ---")  # Prints a header for the test case.
    try:  # Starts a try block.
        sm_zones = get_zone_definitions("single_mode_pc")  # Retrieves the zone definitions for single-mode fibers.
        print(f"\nSingle-Mode PC Zone Definitions (First zone name): {sm_zones[0]['name']}")  # Prints the name of the first zone to confirm retrieval.
        # print(json.dumps(sm_zones, indent=2))
        
        mm_zones = get_zone_definitions("multi_mode_pc")  # Retrieves the zone definitions for multi-mode fibers.
        print(f"\nMulti-Mode PC Zone Definitions (First zone name): {mm_zones[0]['name']}")  # Prints the name of the first zone.
    except ValueError as e:  # Catches a ValueError if the definition doesn't exist.
        print(f"Error in Test Case 4: {e}")  # Prints the error.

    try:  # This block tests the error handling for non-existent zone definitions.
        # Test getting non-existent zone definitions
        get_zone_definitions("non_existent_fiber_type")  # This call is expected to trigger the fallback logic or raise an error.
    except ValueError as e:  # Catches the expected error.
        print(f"\nSuccessfully caught error for non-existent fiber type for zones: {e}")  # Prints a success message.

    # --- Test Case 5: Simulate missing config file and then corrupted file ---
    print("\n--- Test Case 5: Simulate Missing/Corrupt Config File ---")  # Prints a header for the test case.
    test_config_path = Path("temp_config_test.json")  # Defines a path for a temporary test config file.
    if test_config_path.exists(): test_config_path.unlink()  # Deletes the temporary file if it exists from a previous failed run.

    print("\nAttempting to load with missing file (should create default):")  # Explains the next action.
    try:  # Starts a try block.
        load_config(str(test_config_path))  # Calls load_config with the path to a non-existent file.
        print(f"'{test_config_path}' created with default structure.")  # Confirms that the file was created as expected.
    except Exception as e:  # Catches any errors.
        print(f"Error: {e}")  # Prints the error message.

    print("\nSimulating corrupt JSON file:")  # Explains the next action.
    try:  # Starts a try block.
        with open(test_config_path, "w", encoding="utf-8") as f:  # Opens the temporary file for writing.
            f.write("{invalid_json: ")  # Writes an invalid string that is not proper JSON.
        load_config(str(test_config_path))  # Attempts to load the now-corrupted file, expecting an error.
    except ValueError as e:  # Catches the expected ValueError from the JSON parser.
        print(f"Successfully caught error for corrupt JSON: {e}")  # Prints a success message confirming the error was handled.
    finally:  # This block will always execute, whether an error occurred or not.
        if test_config_path.exists(): test_config_path.unlink()  # Cleans up by deleting the temporary test file.