#!/usr/bin/env python3
# run_inspector_interactive.py

"""
D-Scope Blink: Interactive Runner
=================================
Enhanced interactive interface with support for all inspection features.
"""
import argparse
import sys

from pathlib import Path
import logging
from typing import Optional

# Ensure all modules are importable
sys.path.insert(0, str(Path(__file__).parent))

try:
    import main as d_scope_main_module
    from config_loader import load_config
    import cv2
    import numpy as np
except ImportError as e:
    print(f"[CRITICAL] Failed to import required modules: {e}.", file=sys.stderr)
    print("Please ensure all dependencies are installed: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

def get_validated_path(prompt_message: str, is_dir: bool = True, check_exists: bool = True, create_if_not_exist_for_output: bool = False) -> Path:
    """
    Prompts the user for a path and validates it.

    Args:
        prompt_message: The message to display to the user.
        is_dir: True if the path should be a directory, False if a file.
        check_exists: True if the path must exist.
        create_if_not_exist_for_output: If True and is_dir is True, creates the directory if it doesn't exist.

    Returns:
        A Path object for the validated path.
    """
    while True: # Loop until a valid path is provided.
        path_str = input(prompt_message).strip() # Get path string from user.
        if not path_str: # Check if input is empty.
            print("Path cannot be empty. Please try again.")
            continue # Ask for input again.
        
        path_obj = Path(path_str) # Convert string to Path object.

        if check_exists and not path_obj.exists(): # Check if path exists.
            if is_dir and create_if_not_exist_for_output: # If it's an output directory that can be created.
                try:
                    path_obj.mkdir(parents=True, exist_ok=True) # Create directory.
                    print(f"Output directory '{path_obj}' created.")
                    return path_obj # Return created path.
                except OSError as e: # Handle errors during directory creation.
                    print(f"Error: Could not create directory '{path_obj}': {e}. Please check permissions and path.")
                    continue # Ask for input again.
            else: # If path must exist but doesn't.
                print(f"Error: Path '{path_obj}' does not exist. Please try again.")
                continue # Ask for input again.
        
        if is_dir and path_obj.exists() and not path_obj.is_dir(): # Check if path is a directory.
            print(f"Error: Path '{path_obj}' is not a directory. Please try again.")
            continue # Ask for input again.
        
        if not is_dir and path_obj.exists() and not path_obj.is_file(): # Check if path is a file.
            print(f"Error: Path '{path_obj}' is not a file. Please try again.")
            continue # Ask for input again.
            
        return path_obj # Return validated path.

def get_float_input(prompt_message: str, default_val: Optional[float] = None, allow_empty: bool = True) -> Optional[float]:
    """
    Prompts the user for a float input, with optional default.

    Args:
        prompt_message: The message to display.
        default_val: The default value if user enters nothing (only if allow_empty is True).
        allow_empty: If True, pressing Enter uses default_val or returns None.

    Returns:
        The float value or None.
    """
    while True: # Loop until valid input.
        val_str = input(prompt_message).strip() # Get string input.
        if not val_str: # If input is empty.
            if allow_empty: # If empty input is allowed.
                return default_val # Return default value.
            else: # If empty input is not allowed.
                print("Input cannot be empty. Please enter a value.")
                continue # Ask for input again.
        try:
            return float(val_str) # Convert to float and return.
        except ValueError: # Handle conversion errors.
            print("Invalid input. Please enter a valid number (e.g., 9.0, 125, 50.5).")

def main_interactive():
    """
    Runs the D-Scope Blink system through an interactive questionnaire.
    """
    print("=" * 70) # Print header.
    print(" D-Scope Blink: Automated Fiber Optic Inspection System (Interactive Runner)")
    print("=" * 70)
    print("\nWelcome! This script will guide you through the inspection setup.")

    # --- Get Input and Output Directories ---
    print("\n--- Directory Setup ---")
    input_dir = get_validated_path("Enter the FULL path to the directory containing images to inspect: ", is_dir=True, check_exists=True)
    output_dir = get_validated_path("Enter the FULL path for the output directory (will be created if it doesn't exist): ", is_dir=True, check_exists=False, create_if_not_exist_for_output=True)

    # --- Get Fiber Specifications ---
    print("\n--- Fiber Specifications (Optional) ---")
    core_dia_um: Optional[float] = None # Initialize core diameter.
    clad_dia_um: Optional[float] = None # Initialize cladding diameter.
    
    provide_specs_choice = input("Do you want to provide known fiber dimensions (microns)? (y/n, default: n): ").strip().lower()
    if provide_specs_choice == 'y': # If user wants to provide specs.
        core_dia_um = get_float_input("  Enter CORE diameter in microns (e.g., 9, 50.0, 62.5) (press Enter to skip): ", default_val=None, allow_empty=True)
        # For cladding, if user provides specs, it's good to have a value. Let's suggest a common default.
        clad_dia_um_prompt = "  Enter CLADDING diameter in microns (e.g., 125.0) (press Enter for default 125.0 if core was given, else skip): "
        default_clad = 125.0 if core_dia_um is not None else None # Suggest 125 if core was entered.
        clad_dia_um = get_float_input(clad_dia_um_prompt, default_val=default_clad, allow_empty=True)
        if clad_dia_um is None and default_clad is not None and core_dia_um is not None: # If user skipped but a default was suggested.
            clad_dia_um = default_clad
            print(f"  Using default cladding diameter: {clad_dia_um} µm")


    # --- Get Processing Profile ---
    print("\n--- Processing Profile ---")
    profile_choices = {"1": "deep_inspection", "2": "fast_scan"} # Define profile choices.
    profile_prompt = "Select processing profile (1: deep_inspection, 2: fast_scan) (default: 1): "
    profile_choice_num = input(profile_prompt).strip()
    profile_name = profile_choices.get(profile_choice_num, "deep_inspection") # Default to deep_inspection.
    print(f"  Using profile: {profile_name}")

    # --- Get Fiber Type Key ---
    print("\n--- Fiber Type for Rules ---")
    fiber_type_prompt = "Enter fiber type key for pass/fail rules (e.g., single_mode_pc, multi_mode_pc) (default: single_mode_pc): "
    fiber_type_key = input(fiber_type_prompt).strip()
    if not fiber_type_key: # If input is empty.
        fiber_type_key = "single_mode_pc" # Default to single_mode_pc.
    print(f"  Using fiber type key: {fiber_type_key}")

    # --- Default Config and Calibration File Paths ---
    # These are typically in the same directory as the scripts or a known location.
    config_file_path = "config.json" # Default config file name.
    calibration_file_path = "calibration.json" # Default calibration file name.
    print(f"\n--- Configuration Files ---")
    print(f"  Using configuration file: '{config_file_path}' (must exist or be creatable by the system).")
    print(f"  Using calibration file: '{calibration_file_path}' (if it exists).")

    print("\n--- Starting Inspection Process ---")
    print("Please wait, this may take some time depending on the number of images and profile...")

    # --- Prepare to call the main logic ---
    # The main.py script's main() function uses argparse.
    # We need to call the core logic within main.py.
    # Let's assume main.py is modified to have a function like:
    # def execute_full_inspection(input_dir_str, output_dir_str, config_file_str,
    #                             calibration_file_str, profile_name_str, fiber_type_str,
    #                             core_dia_opt_float, clad_dia_opt_float)
    #
    # If main.py is not modified, we might need to simulate argparse arguments,
    # or directly call the sequence of operations from main.py here.
    # For now, we'll assume we can call a function in d_scope_main_module.
    # The 'main()' function in the provided 'main.py' already orchestrates everything.
    # We will call that, but we need to simulate the 'args' object it expects.

    class ArgsSimulator: # Class to simulate argparse Namespace object.
        """Simulates the argparse.Namespace object for main.py's main function."""
        def __init__(self, input_dir, output_dir, config_file, calibration_file,
                     profile, fiber_type, core_dia_um, clad_dia_um):
            self.input_dir = str(input_dir) # Convert Path to string.
            self.output_dir = str(output_dir) # Convert Path to string.
            self.config_file = str(config_file) # Convert Path to string.
            self.calibration_file = str(calibration_file) # Convert Path to string.
            self.profile = profile
            self.fiber_type = fiber_type
            self.core_dia_um = core_dia_um
            self.clad_dia_um = clad_dia_um

    simulated_args = ArgsSimulator( # Create simulated args object.
        input_dir=input_dir,
        output_dir=output_dir,
        config_file=Path(config_file_path), # Pass as Path, main.py converts to str if needed or uses Path.
        calibration_file=Path(calibration_file_path),
        profile=profile_name,
        fiber_type=fiber_type_key,
        core_dia_um=core_dia_um,
        clad_dia_um=clad_dia_um
    )

    try:
        # Call a modified main function from main.py that accepts these args
        # For now, let's assume d_scope_main_module.main() can be called,
        # and it will internally use these simulated_args if we modify it slightly
        # or we call a new entry point.
        
        # The simplest adaptation is to modify main.py's main() to accept an 'args_override'
        # If not, this runner would have to replicate the entire orchestration logic of main.py's main().
        
        # Let's assume we add a function to main.py:
        # def run_inspection_with_params(params_obj):
        #     # ... uses params_obj instead of parser.parse_args() ...
        #     # ... then continues with the rest of main() logic ...
        
        # For now, to avoid modifying main.py immediately, this runner will print the collected params
        # and instruct how main.py could be called.
        # A true direct call requires main.py to be import-friendly for its core logic.

        print("\nCollected Parameters for D-Scope Blink:") # Print collected parameters.
        print(f"  Input Directory: {simulated_args.input_dir}")
        print(f"  Output Directory: {simulated_args.output_dir}")
        print(f"  Config File: {simulated_args.config_file}")
        print(f"  Calibration File: {simulated_args.calibration_file}")
        print(f"  Processing Profile: {simulated_args.profile}")
        print(f"  Fiber Type Key: {simulated_args.fiber_type}")
        print(f"  Core Diameter (µm): {simulated_args.core_dia_um if simulated_args.core_dia_um is not None else 'Not Provided'}")
        print(f"  Cladding Diameter (µm): {simulated_args.clad_dia_um if simulated_args.clad_dia_um is not None else 'Not Provided'}")

        # This is where you would call the refactored main logic.
        # For example, if main.py had:
        # def execute_inspection_run(args_namespace):
        #    # ... all of main.py's logic after parsing ...
        # then you would call:
        # d_scope_main_module.execute_inspection_run(simulated_args)

        # Since the original main.py uses argparse directly in its main(),
        # the most direct way without refactoring main.py is to
        # essentially replicate the main() logic here, or have main.py's main()
        # optionally accept an args object.
        
        # For this iteration, I will call the existing main() from main.py.
        # This requires main.py's main() to be slightly adjusted to accept an optional 'args_override'.
        # If main.py's main() is not changed, this call will fail or argparse will try to parse CLI args.

        # Let's assume main.py's main() is modified like this:
        # def main(args_override=None):
        #     if args_override:
        #         args = args_override
        #     else:
        #         parser = argparse.ArgumentParser(...)
        #         args = parser.parse_args()
        #     # ... rest of the logic ...

        # Then we can call it:
        if hasattr(d_scope_main_module, 'main_with_args'): # Check if a compatible main function exists.
            d_scope_main_module.main_with_args(simulated_args) # Call the compatible main function.
        else: # If not, print instructions.
            print("\n[INFO] To fully integrate this interactive runner, 'main.py' should be modified")
            print("       to accept parameters programmatically (e.g., its main() function taking an 'args' object).")
            print("       For now, the parameters have been collected. You would typically pass these to the core inspection logic.")
            print("       Consider refactoring main.py to expose a function like 'execute_full_inspection(params_dict)'.")


    except Exception as e: # Handle any errors during the main process.
        logging.error(f"An error occurred during the inspection process: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")
        print("Please check the log file in the output directory for more details.")
    finally: # Final message.
        print("\n--- Interactive Inspection Run Finished ---")
        print(f"Check the output directory '{output_dir}' for results and logs.")

# Note: The following functions (main, cli_main, main_with_args) were part of the original
# uploaded content but seem to be intended for a main.py, not run.py.
# They are kept here based on the initial file content but would typically
# reside in the module that d_scope_main_module refers to.
# If d_scope_main_module.main_with_args is not found, the script will print an info message.

def main(args_override=None): # Add args_override=None
    """
    Main function to drive the D-Scope Blink inspection system.
    Can accept an args_override object for programmatic execution.
    """
    global_args = None # Initialize global_args.
    if args_override: # If args_override is provided.
        global_args = args_override # Use provided args.
        print("[INFO] main.py's main() called with overridden arguments.") # For debugging.
    else: # If no override, parse from command line.
        # --- Argument Parsing ---
        parser = argparse.ArgumentParser(description="D-Scope Blink: Automated Fiber Optic End Face Inspection System.")
        # ... (all your existing argparse.add_argument calls) ...
        # Example: parser.add_argument("input_dir", type=str, help="Path to the directory containing images to inspect.")
        # ...
        global_args = parser.parse_args() # Parse CLI args.

    # --- Output Directory Setup ---
    # Use global_args.output_dir, global_args.input_dir etc. from here onwards
    base_output_dir = Path(global_args.output_dir)
    # ... (rest of your existing main() logic, using 'global_args' instead of 'args') ...
    
    # Example of using a value from global_args:
    # setup_logging(
    #     global_config.get("general_settings", {}).get("log_level", "INFO"), # Assuming global_config is loaded based on global_args.config_file
    #     global_config.get("general_settings", {}).get("log_to_console", True),
    #     current_run_output_dir
    # )
    # ... and so on for all uses of 'args' ...

# This allows main.py to still be run from CLI
# if __name__ == "__main__":
#     main() 

# OR, to make it cleaner for the runner, you might rename the original main to cli_main
# and have a new main that decides:
def cli_main(): # Original main function, now for CLI.
    parser = argparse.ArgumentParser(description="D-Scope Blink: Automated Fiber Optic End Face Inspection System.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing images to inspect.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where results will be saved.")
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the JSON configuration file (default: config.json).")
    parser.add_argument("--calibration_file", type=str, default="calibration.json", help="Path to the JSON calibration file (default: calibration.json).")
    parser.add_argument("--profile", type=str, default="deep_inspection", choices=["fast_scan", "deep_inspection"], help="Processing profile to use (default: deep_inspection).")
    parser.add_argument("--fiber_type", type=str, default="single_mode_pc", help="Key for fiber type specific rules, e.g., 'single_mode_pc', 'multi_mode_pc' (must match config.json).")
    parser.add_argument("--core_dia_um", type=float, default=None, help="Optional: Known core diameter in microns for this batch.")
    parser.add_argument("--clad_dia_um", type=float, default=None, help="Optional: Known cladding diameter in microns for this batch.")
    args = parser.parse_args()
    main_with_args(args) # Call the core logic.

def main_with_args(args_namespace): # Renamed from 'main' or new function.
    """
    Core inspection logic that takes an args-like namespace object.
    """
    # --- Output Directory Setup ---
    # This section would require datetime and time imports, and other dependencies
    # like pandas (pd) and custom functions (setup_logging, load_config, etc.)
    # which are not fully defined within this run.py script but are assumed
    # to be in d_scope_main_module or its dependencies.
    
    # Example (needs 'import datetime', 'import time', 'import pandas as pd'):
    # import datetime
    # import time
    # import pandas as pd # Assuming pandas is used for DataFrame
    # from main import setup_logging, get_processing_profile, load_calibration_data, process_single_image # Hypothetical imports
    
    base_output_dir = Path(args_namespace.output_dir)
    # run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') # Requires datetime import
    # current_run_output_dir = base_output_dir / f"run_{run_timestamp}"
    # current_run_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Configuration and Logging Setup ---
    # try:
    #     global_config = load_config(args_namespace.config_file) # Assumes load_config is available
    # except (FileNotFoundError, ValueError) as e:
    #     print(f"[CRITICAL] Failed to load configuration: {e}. Exiting.", file=sys.stderr)
    #     logging.critical(f"Failed to load configuration: {e}. Exiting.")
    #     sys.exit(1)

    # general_settings = global_config.get("general_settings", {})
    # setup_logging( # Assumes setup_logging is available
    #     general_settings.get("log_level", "INFO"),
    #     general_settings.get("log_to_console", True),
    #     current_run_output_dir
    # )

    logging.info("D-Scope Blink: Inspection System Started (called via main_with_args).") # Modified log
    logging.info(f"Input Directory: {args_namespace.input_dir}")
    # logging.info(f"Output Directory (this run): {current_run_output_dir}") # Depends on current_run_output_dir
    logging.info(f"Using Profile: {args_namespace.profile}")
    logging.info(f"Fiber Type Key for Rules: {args_namespace.fiber_type}")
    if args_namespace.core_dia_um: logging.info(f"User Provided Core Diameter: {args_namespace.core_dia_um} µm")
    if args_namespace.clad_dia_um: logging.info(f"User Provided Cladding Diameter: {args_namespace.clad_dia_um} µm")
    
    # ... The rest of the logic from the uploaded main_with_args would go here ...
    # This includes image discovery, batch processing, and summary report generation.
    # For brevity and because these depend on unimported modules and functions within
    # the context of run.py itself (they belong in main.py), they are omitted here.
    # The call d_scope_main_module.main_with_args(simulated_args) is the key part
    # that executes this logic if main.py is structured accordingly.
    print(f"[INFO] Placeholder for main_with_args logic from main.py, if it were in run.py")


    try:
        # Direct integration with main.py
        from main import execute_inspection_run
        
        print("\n[INFO] Starting D-Scope Blink inspection with collected parameters...")
        
        # Call the main inspection logic directly
        execute_inspection_run(simulated_args)
        
        print("\n--- Inspection Complete ---")
        print(f"Results saved to: {output_dir}")
        
    except ImportError as e:
        print(f"\n[ERROR] Could not import inspection module: {e}")
        print("Please ensure main.py is in the same directory as run.py")
    except Exception as e:
        logging.error(f"Inspection failed: {e}", exc_info=True)
        print(f"\n[ERROR] Inspection failed: {e}")
        print("Check the log files for detailed error information")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    try:
        main_interactive()
    except KeyboardInterrupt:
        print("\n\n[INFO] Inspection cancelled by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Interactive runner failed: {e}")
        print(f"\n[CRITICAL] Interactive runner failed: {e}")
        sys.exit(1)