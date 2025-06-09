#!/usr/bin/env python3
# run_inspector_interactive.py

"""
Interactive Runner
=================================
Enhanced interactive interface for the D-Scope Blink system.
This script gathers parameters from the user and then calls the main inspection
logic, which is expected to be in 'main.py'.
"""
import sys
from pathlib import Path
import logging
from typing import Optional, cast, Any # Added cast and Any

# Ensure all modules are importable from the script's directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    # Attempt to import the main module from main.py
    # This module should contain the core inspection logic.
    import main as d_scope_main_module
    # The following imports might be specific to main.py or shared,
    # but run.py itself doesn't directly use them beyond parameter collection.
    # If config_loader, cv2, or numpy are ONLY used within main.py,
    # they don't strictly need to be imported here, but it can help catch
    # missing dependencies early.
    from config_loader import load_config # Assuming config_loader.py exists
    import cv2
    import numpy as np
except ImportError as e:
    print(f"[CRITICAL] Failed to import required modules: {e}.", file=sys.stderr)
    print("Please ensure 'main.py', 'config_loader.py', and all their dependencies ", file=sys.stderr)
    print("(e.g., OpenCV, NumPy) are available in the same directory or Python path.", file=sys.stderr)
    print("You might need to install dependencies: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

def get_validated_path(prompt_message: str, is_dir: bool = True, check_exists: bool = True, create_if_not_exist_for_output: bool = False) -> Path:
    """
    Prompts the user for a path and validates it.

    Args:
        prompt_message: The message to display to the user.
        is_dir: True if the path should be a directory, False if a file.
        check_exists: True if the path must exist (unless create_if_not_exist_for_output is True).
        create_if_not_exist_for_output: If True and is_dir is True, creates the directory if it doesn't exist.

    Returns:
        A Path object for the validated path.
    """
    while True: # Loop until a valid path is provided.
        path_str = input(prompt_message).strip() # Get path string from user.
        if not path_str: # Check if input is empty.
            print("Path cannot be empty. Please try again.")
            continue # Ask for input again.
        
        path_obj = Path(path_str).resolve() # Convert string to Path object and resolve to absolute path.

        if check_exists and not path_obj.exists(): # Check if path exists.
            if is_dir and create_if_not_exist_for_output: # If it's an output directory that can be created.
                try:
                    path_obj.mkdir(parents=True, exist_ok=True) # Create directory.
                    print(f"Output directory '{path_obj}' created.")
                    return path_obj # Return created path.
                except OSError as e: # Handle errors during directory creation.
                    print(f"Error: Could not create directory '{path_obj}': {e}. Please check permissions and path.")
                    continue # Ask for input again.
            else: # If path must exist but doesn't (and not creating).
                print(f"Error: Path '{path_obj}' does not exist. Please try again.")
                continue # Ask for input again.
        
        # If path_obj.exists() is true at this point, or if check_exists was false:
        if path_obj.exists(): # Perform type checks only if path exists
            if is_dir and not path_obj.is_dir(): # Check if path is a directory.
                print(f"Error: Path '{path_obj}' exists but is not a directory. Please try again.")
                continue # Ask for input again.
            
            if not is_dir and not path_obj.is_file(): # Check if path is a file.
                print(f"Error: Path '{path_obj}' exists but is not a file. Please try again.")
                continue # Ask for input again.
        elif not create_if_not_exist_for_output and not check_exists: # Path doesn't exist, not creating, not checking
             pass # Allow non-existent paths if check_exists is False (e.g. for output file names)
            
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

class ArgsSimulator:
    """
    Simulates the argparse.Namespace object that the main inspection script (main.py)
    is expected to use for its parameters.
    """
    def __init__(self, input_dir: Path, output_dir: Path, config_file: Path,
                 calibration_file: Path, profile: str, fiber_type: str,
                 core_dia_um: Optional[float], clad_dia_um: Optional[float]):
        # Store paths as strings, as argparse typically provides them,
        # and it's safer for cross-module compatibility if main.py expects strings.
        self.input_dir = str(input_dir)
        self.output_dir = str(output_dir)
        self.config_file = str(config_file)
        self.calibration_file = str(calibration_file)
        self.profile = profile
        self.fiber_type = fiber_type
        self.core_dia_um = core_dia_um
        self.clad_dia_um = clad_dia_um
        
    def __repr__(self):
        return (f"ArgsSimulator(input_dir='{self.input_dir}', output_dir='{self.output_dir}', "
                f"config_file='{self.config_file}', calibration_file='{self.calibration_file}', "
                f"profile='{self.profile}', fiber_type='{self.fiber_type}', "
                f"core_dia_um={self.core_dia_um}, clad_dia_um={self.clad_dia_um})")

def main_interactive():
    """
    Runs the D-Scope Blink system through an interactive questionnaire to gather
    parameters, then calls the core inspection logic from main.py.
    """
    print("=" * 70)
    print(" D-Scope Blink: Automated Fiber Optic Inspection System (Interactive Runner)")
    print("=" * 70)
    print("\nWelcome! This script will guide you through the inspection setup.")

    # --- Get Input and Output Directories ---
    print("\n--- Directory Setup ---")
    input_dir = get_validated_path("Enter the FULL path to the directory containing images to inspect: ", is_dir=True, check_exists=True)
    # For output_dir, we check_exists=False because get_validated_path will create it if create_if_not_exist_for_output is True.
    output_dir = get_validated_path("Enter the FULL path for the output directory (will be created if it doesn't exist): ", is_dir=True, check_exists=False, create_if_not_exist_for_output=True)

    # --- Get Fiber Specifications ---
    print("\n--- Fiber Specifications (Optional) ---")
    core_dia_um: Optional[float] = None
    clad_dia_um: Optional[float] = None
    
    provide_specs_choice = input("Do you want to provide known fiber dimensions (microns)? (y/n, default: n): ").strip().lower()
    if provide_specs_choice == 'y':
        core_dia_um = get_float_input("  Enter CORE diameter in microns (e.g., 9, 50.0, 62.5) (press Enter to skip): ", default_val=None, allow_empty=True)
        
        default_clad_suggestion = 125.0 if core_dia_um is not None else None
        clad_dia_um_prompt = "  Enter CLADDING diameter in microns (e.g., 125.0) "
        if default_clad_suggestion:
            clad_dia_um_prompt += f"(press Enter for default {default_clad_suggestion} if core was given, else skip): "
        else:
            clad_dia_um_prompt += "(press Enter to skip): "
            
        clad_dia_um = get_float_input(clad_dia_um_prompt, default_val=default_clad_suggestion, allow_empty=True)
        
        # If user skipped cladding but a default was suggested (because core was entered), use the default.
        if clad_dia_um is None and default_clad_suggestion is not None and core_dia_um is not None:
            clad_dia_um = default_clad_suggestion
            print(f"  Using default cladding diameter: {clad_dia_um} µm")

    # --- Get Processing Profile ---
    print("\n--- Processing Profile ---")
    profile_choices = {"1": "deep_inspection", "2": "fast_scan"}
    profile_prompt = "Select processing profile (1: deep_inspection, 2: fast_scan) (default: 1): "
    profile_choice_num = input(profile_prompt).strip()
    profile_name = profile_choices.get(profile_choice_num, "deep_inspection") # Default to deep_inspection
    print(f"  Using profile: {profile_name}")

    # --- Get Fiber Type Key ---
    print("\n--- Fiber Type for Rules ---")
    fiber_type_prompt = "Enter fiber type key for pass/fail rules (e.g., single_mode_pc, multi_mode_pc) (default: single_mode_pc): "
    fiber_type_key = input(fiber_type_prompt).strip()
    if not fiber_type_key:
        fiber_type_key = "single_mode_pc" # Default to single_mode_pc
    print(f"  Using fiber type key: {fiber_type_key}")

    # --- Default Config and Calibration File Paths ---
    # These are assumed to be relative to the script's location or a known path.
    # main.py will resolve these relative to its own location or CWD if not absolute.
    script_dir = Path(__file__).resolve().parent
    default_config_file_path = script_dir / "config.json"
    default_calibration_file_path = script_dir / "calibration.json"
    
    print(f"\n--- Configuration Files (Expected Location) ---")
    config_file_interactive_prompt = f"Enter path to config file (default: '{default_config_file_path}'): "
    config_file_str = input(config_file_interactive_prompt).strip()
    config_file = Path(config_file_str) if config_file_str else default_config_file_path
    # Validate config file existence (it should ideally exist)
    config_file = get_validated_path(f"Re-enter path to config file (must exist): ", is_dir=False, check_exists=True) if not config_file.is_file() and config_file_str else config_file


    calibration_file_interactive_prompt = f"Enter path to calibration file (default: '{default_calibration_file_path}', press Enter to skip if not used/exists): "
    calibration_file_str = input(calibration_file_interactive_prompt).strip()
    calibration_file = Path(calibration_file_str) if calibration_file_str else default_calibration_file_path
    # Calibration file is optional, so we don't strictly enforce its existence here. main.py should handle it.
    if calibration_file_str or default_calibration_file_path.exists(): # If user provided one or default exists
         print(f"  Using calibration file: '{calibration_file.resolve()}' (existence will be checked by main process).")
    else:
        print(f"  No specific calibration file provided, or default '{default_calibration_file_path}' not found. Main process may use internal defaults or skip.")


    print("\n--- Preparing for Inspection ---")
    print("Parameters collected. Attempting to start the inspection process via main.py...")

    simulated_args = ArgsSimulator(
        input_dir=input_dir,
        output_dir=output_dir,
        config_file=config_file.resolve(), # Pass resolved path
        calibration_file=calibration_file.resolve(), # Pass resolved path
        profile=profile_name,
        fiber_type=fiber_type_key,
        core_dia_um=core_dia_um,
        clad_dia_um=clad_dia_um
    )

    print("\nCollected Parameters for D-Scope Blink:")
    print(f"  Input Directory: {simulated_args.input_dir}")
    print(f"  Output Directory: {simulated_args.output_dir}")
    print(f"  Config File: {simulated_args.config_file}")
    print(f"  Calibration File: {simulated_args.calibration_file}")
    print(f"  Processing Profile: {simulated_args.profile}")
    print(f"  Fiber Type Key: {simulated_args.fiber_type}")
    print(f"  Core Diameter (µm): {simulated_args.core_dia_um if simulated_args.core_dia_um is not None else 'Not Provided'}")
    print(f"  Cladding Diameter (µm): {simulated_args.clad_dia_um if simulated_args.clad_dia_um is not None else 'Not Provided'}")

    # --- Execute Main Inspection Logic from d_scope_main_module (main.py) ---
    inspection_successful = False
    try:
        # Option 1: Prefer 'execute_inspection_run' if available in d_scope_main_module (your main.py)
        if hasattr(d_scope_main_module, 'execute_inspection_run'):
            print("\n[INFO] Calling 'execute_inspection_run' from the imported main module (main.py)...")
            # Corrected based on Problems.txt [cite: 130]
            d_scope_main_module.execute_inspection_run(cast(Any, simulated_args))
            inspection_successful = True
        # Option 2: Fallback to 'main_with_args' if 'execute_inspection_run' is not available
        elif hasattr(d_scope_main_module, 'main_with_args'):
            print("\n[INFO] 'execute_inspection_run' not found in main.py.")
            print("       Attempting to call 'main_with_args' from the imported main module as an alternative...")
            # Corrected based on Problems.txt [cite: 130, 131]
            d_scope_main_module.main_with_args(cast(Any, simulated_args))
            inspection_successful = True
        # If neither suitable function is available in main.py
        else:
            print("\n❌ [ERROR] Could not find a suitable function ('execute_inspection_run' or 'main_with_args') in the main module (main.py).")
            print("       To fully integrate this interactive runner, your 'main.py' file needs to provide")
            print("       one of these functions designed to accept parameters programmatically.")
            print("       For example, in main.py, you could define:")
            print("         def execute_inspection_run(args_namespace):")
            print("             # ... your main inspection logic using args_namespace ...")
            print("       OR, if you have an existing main() function that parses arguments:")
            print("         def main_with_args(args_namespace):")
            print("             # ... your main inspection logic using args_namespace ...")
            print("         def main(): # Your original main for CLI")
            print("             import argparse")
            print("             parser = argparse.ArgumentParser(...)")
            print("             # ... add arguments ...")
            print("             cli_args = parser.parse_args()")
            print("             main_with_args(cli_args)")
            print("\n       Please ensure your main.py is structured accordingly.")
            print("       Parameters were collected, but the inspection could not be started.")

        if inspection_successful:
            print("\n✅ --- Inspection Process Reported as Complete by Main Module ---")

    except ImportError as ie: # Specifically catch if d_scope_main_module itself couldn't be imported initially
        # This case should ideally be caught by the top-level import, but kept for robustness.
        logging.error(f"Failed to import the main inspection module (main.py): {ie}", exc_info=True)
        print(f"\n❌ [CRITICAL ERROR] Failed to import the main inspection module (main.py): {ie}")
        print("       Ensure 'main.py' exists in the same directory as 'run.py' and has no import errors itself.")
    except Exception as e: # Catch other errors from the called inspection function in main.py
        logging.error(f"An error occurred during the call to the main module's inspection function: {e}", exc_info=True)
        print(f"\n❌ An unexpected error occurred during the inspection process: {e}")
        print("       Please check the log file in the output directory (if created) and the console for more details.")
    finally:
        print("\n--- Interactive Inspection Run Concluded ---")
        if inspection_successful:
            print(f"Check the output directory '{simulated_args.output_dir}' for results and logs.")
        else:
            print(f"The inspection process may not have started or completed successfully.")
            print(f"       If an output directory '{simulated_args.output_dir}' was created by this script, check it for any partial results or logs.")
            print(f"       Review console messages for specific errors, especially regarding main.py integration.")

if __name__ == "__main__":
    # Basic logging setup for the runner itself.
    # The main inspection logic in main.py should set up its own more detailed logging.
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    try:
        main_interactive()
    except KeyboardInterrupt:
        print("\n\n[INFO] Interactive inspection setup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        # Catch any unexpected errors within main_interactive itself (not from main.py calls, which are handled inside)
        logging.critical(f"Interactive runner encountered a critical failure: {e}", exc_info=True)
        print(f"\n❌ [CRITICAL ERROR] The interactive runner failed unexpectedly: {e}")
        sys.exit(1)