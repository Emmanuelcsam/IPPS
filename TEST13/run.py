#!/usr/bin/env python3
# run_inspector_interactive.py

"""
Interactive Runner
=================================
Enhanced interactive interface for the D-Scope Blink system.
This script gathers parameters from the user and then calls the main inspection
logic, which is expected to be in 'main.py'.
"""
# Import the sys module to interact with the Python runtime environment, used here for path manipulation and exiting the script.
import sys
# Import the Path class from the pathlib module for object-oriented filesystem paths, making path operations more readable and cross-platform compatible.
from pathlib import Path
# Import the logging module to record events, errors, and informational messages for debugging and monitoring the script's execution.
import logging
# Import specific types from the typing module for type hinting, which improves code clarity and allows for static analysis.
from typing import Optional, cast, Any # Added cast and Any

# Ensure all modules are importable from the script's directory by adding the parent directory of this script file to the Python path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Start a try-except block to gracefully handle potential errors if required modules are not found.
try:

    import main as main_module
    # The following imports are declared to catch missing dependencies early, even if they are primarily used by main.py.
    # Import the load_config function from the config_loader module, which is responsible for loading operational parameters.
    from config_loader import load_config # Assuming config_loader.py exists
    # Import the OpenCV library, essential for the image processing tasks performed in the main module.
    import cv2
    # Import the NumPy library, which is the fundamental package for numerical operations and is heavily used with OpenCV.
    import numpy as np
# Catch the ImportError that occurs if any of the above modules cannot be found.
except ImportError as e:
    # Print a critical error message to the standard error stream indicating which module failed to import.
    print(f"[CRITICAL] Failed to import required modules: {e}.", file=sys.stderr)
    # Provide guidance to the user on how to resolve the missing dependency issue.
    print("Please ensure 'main.py', 'config_loader.py', and all their dependencies ", file=sys.stderr)
    # Continue the guidance message.
    print("(e.g., OpenCV, NumPy) are available in the same directory or Python path.", file=sys.stderr)
    # Suggest installing dependencies from a requirements file, which is a standard Python practice.
    print("You might need to install dependencies: pip install -r requirements.txt", file=sys.stderr)
    # Exit the script with a status code of 1 to indicate that an error occurred.
    sys.exit(1)

# Define a function to get and validate a filesystem path from the user, with clear type hints for its arguments and return value.
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
    # Start an infinite loop that continues until a valid path is provided by the user.
    while True: # Loop until a valid path is provided.
        # Prompt the user for input using the provided message and remove any leading/trailing whitespace.
        path_str = input(prompt_message).strip() # Get path string from user.
        # Check if the user provided an empty string.
        if not path_str: # Check if input is empty.
            # Inform the user that the path cannot be empty.
            print("Path cannot be empty. Please try again.")
            # Skip the rest of the loop and re-prompt the user for input.
            continue # Ask for input again.
        
        # Create a Path object from the user's string and resolve it to an absolute path for consistency.
        path_obj = Path(path_str).resolve() # Convert string to Path object and resolve to absolute path.

        # Check if the path is required to exist but doesn't.
        if check_exists and not path_obj.exists(): # Check if path exists.
            # If the path is a directory and the option to create it is enabled.
            if is_dir and create_if_not_exist_for_output: # If it's an output directory that can be created.
                # Start a try block to handle potential OS errors during directory creation.
                try:
                    # Create the directory, including any necessary parent directories, without raising an error if it already exists.
                    path_obj.mkdir(parents=True, exist_ok=True) # Create directory.
                    # Inform the user that the directory was successfully created.
                    print(f"Output directory '{path_obj}' created.")
                    # Return the newly created and validated Path object, exiting the loop.
                    return path_obj # Return created path.
                # Catch any OSError that might occur, for example, due to lack of permissions.
                except OSError as e: # Handle errors during directory creation.
                    # Print an informative error message explaining why the directory could not be created.
                    print(f"Error: Could not create directory '{path_obj}': {e}. Please check permissions and path.")
                    # Skip the rest of the loop and re-prompt the user.
                    continue # Ask for input again.
            # If the path was required to exist but doesn't, and it's not an output directory to be created.
            else: # If path must exist but doesn't (and not creating).
                # Print an error message indicating the path does not exist.
                print(f"Error: Path '{path_obj}' does not exist. Please try again.")
                # Skip the rest of the loop and re-prompt the user.
                continue # Ask for input again.
        
        # If the path exists, further validation is needed to check if it's a file or a directory as expected.
        if path_obj.exists(): # Perform type checks only if path exists
            # If the path is expected to be a directory but it's not.
            if is_dir and not path_obj.is_dir(): # Check if path is a directory.
                # Print an error message stating the validation failure.
                print(f"Error: Path '{path_obj}' exists but is not a directory. Please try again.")
                # Skip the rest of the loop and re-prompt the user.
                continue # Ask for input again.
            
            # If the path is expected to be a file but it's not.
            if not is_dir and not path_obj.is_file(): # Check if path is a file.
                # Print an error message stating the validation failure.
                print(f"Error: Path '{path_obj}' exists but is not a file. Please try again.")
                # Skip the rest of the loop and re-prompt the user.
                continue # Ask for input again.
        # This condition handles cases for output files that don't exist yet, where existence check is off.
        elif not create_if_not_exist_for_output and not check_exists: # Path doesn't exist, not creating, not checking
             # Do nothing and allow the function to proceed, as this path is valid under these conditions.
             pass # Allow non-existent paths if check_exists is False (e.g. for output file names)
            
        # If all checks have passed, return the validated Path object.
        return path_obj # Return validated path.

# Define a function to get a floating-point number from the user, with support for defaults.
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
    # Start an infinite loop that continues until a valid float or allowed empty input is received.
    while True: # Loop until valid input.
        # Prompt the user for input and remove any leading/trailing whitespace.
        val_str = input(prompt_message).strip() # Get string input.
        # Check if the user provided an empty string.
        if not val_str: # If input is empty.
            # If empty input is allowed for this prompt.
            if allow_empty: # If empty input is allowed.
                # Return the specified default value (which could be None).
                return default_val # Return default value.
            # If empty input is not allowed.
            else: # If empty input is not allowed.
                # Inform the user that input is required.
                print("Input cannot be empty. Please enter a value.")
                # Skip the rest of the loop and re-prompt the user.
                continue # Ask for input again.
        # Start a try block to handle potential errors when converting the string to a float.
        try:
            # Attempt to convert the user's string to a float and return it.
            return float(val_str) # Convert to float and return.
        # Catch the ValueError that occurs if the string is not a valid number.
        except ValueError: # Handle conversion errors.
            # Print an informative error message with examples of valid input.
            print("Invalid input. Please enter a valid number (e.g., 9.0, 125, 50.5).")

# Define a class to simulate the argparse.Namespace object.
class ArgsSimulator:
    """
    Simulates the argparse.Namespace object that the main inspection script (main.py)
    is expected to use for its parameters.
    """
    # The constructor for the class, which takes all the parameters gathered interactively.
    def __init__(self, input_dir: Path, output_dir: Path, config_file: Path,
                 calibration_file: Path, profile: str, fiber_type: str,
                 core_dia_um: Optional[float], clad_dia_um: Optional[float]):
        # Store the input directory path as a string to mimic argparse behavior.
        self.input_dir = str(input_dir)
        # Store the output directory path as a string.
        self.output_dir = str(output_dir)
        # Store the configuration file path as a string.
        self.config_file = str(config_file)
        # Store the calibration file path as a string.
        self.calibration_file = str(calibration_file)
        # Store the selected processing profile name.
        self.profile = profile
        # Store the fiber type key for rule application.
        self.fiber_type = fiber_type
        # Store the user-provided core diameter in microns.
        self.core_dia_um = core_dia_um
        # Store the user-provided cladding diameter in microns.
        self.clad_dia_um = clad_dia_um
        
    # Define the official string representation of the object, useful for debugging.
    def __repr__(self):
        # Return a formatted string that shows how to recreate the object with its current state.
        return (f"ArgsSimulator(input_dir='{self.input_dir}', output_dir='{self.output_dir}', "
                f"config_file='{self.config_file}', calibration_file='{self.calibration_file}', "
                f"profile='{self.profile}', fiber_type='{self.fiber_type}', "
                f"core_dia_um={self.core_dia_um}, clad_dia_um={self.clad_dia_um})")

# Define the main function that orchestrates the interactive session.
def main_interactive():
    """
    Runs the D-Scope Blink system through an interactive questionnaire to gather
    parameters, then calls the core inspection logic from main.py.
    """
    # Print a decorative top border for the welcome message.
    print("=" * 70)
    # Print the main title of the application.
    print(" D-Scope Blink: Automated Fiber Optic Inspection System (Interactive Runner)")
    # Print a decorative bottom border for the welcome message.
    print("=" * 70)
    # Print a welcome message to the user.
    print("\nWelcome! This script will guide you through the inspection setup.")

    # --- Get Input and Output Directories ---
    # Print a section header for directory setup.
    print("\n--- Directory Setup ---")
    # Call the validation function to get the input directory path from the user, ensuring it exists.
    input_dir = get_validated_path("Enter the FULL path to the directory containing images to inspect: ", is_dir=True, check_exists=True)
    # Call the validation function to get the output directory path, allowing it to be created if it doesn't exist.
    output_dir = get_validated_path("Enter the FULL path for the output directory (will be created if it doesn't exist): ", is_dir=True, check_exists=False, create_if_not_exist_for_output=True)

    # --- Get Fiber Specifications ---
    # Print a section header for fiber specifications, noting they are optional.
    print("\n--- Fiber Specifications (Optional) ---")
    # Initialize core diameter variable to None, indicating no value yet.
    core_dia_um: Optional[float] = None
    # Initialize cladding diameter variable to None.
    clad_dia_um: Optional[float] = None
    
    # Ask the user if they want to provide known fiber dimensions.
    provide_specs_choice = input("Do you want to provide known fiber dimensions (microns)? (y/n, default: n): ").strip().lower()
    # Check if the user's choice is 'y' (yes).
    if provide_specs_choice == 'y':
        # If yes, prompt for the core diameter, allowing the user to skip by pressing Enter.
        core_dia_um = get_float_input("  Enter CORE diameter in microns (e.g., 9, 50.0, 62.5) (press Enter to skip): ", default_val=None, allow_empty=True)
        
        # Suggest a default cladding diameter of 125.0 µm if a core diameter was provided, otherwise suggest nothing.
        default_clad_suggestion = 125.0 if core_dia_um is not None else None
        # Start building the prompt message for the cladding diameter.
        clad_dia_um_prompt = "  Enter CLADDING diameter in microns (e.g., 125.0) "
        # If a default was suggested, add it to the prompt.
        if default_clad_suggestion:
            # Append the default value information to the prompt string.
            clad_dia_um_prompt += f"(press Enter for default {default_clad_suggestion} if core was given, else skip): "
        # Otherwise, if no default was suggested.
        else:
            # Append a generic "press Enter to skip" message.
            clad_dia_um_prompt += "(press Enter to skip): "
            
        # Prompt for the cladding diameter using the constructed prompt and suggested default.
        clad_dia_um = get_float_input(clad_dia_um_prompt, default_val=default_clad_suggestion, allow_empty=True)
        
        # This logic ensures that if the user entered a core diameter but skipped the cladding prompt, the suggested default is used.
        if clad_dia_um is None and default_clad_suggestion is not None and core_dia_um is not None:
            # Assign the default suggestion to the cladding diameter variable.
            clad_dia_um = default_clad_suggestion
            # Inform the user that the default value is being used.
            print(f"  Using default cladding diameter: {clad_dia_um} µm")

    # --- Get Processing Profile ---
    # Print a section header for the processing profile selection.
    print("\n--- Processing Profile ---")
    # Define a dictionary to map user's numerical choice to profile names.
    profile_choices = {"1": "deep_inspection", "2": "fast_scan"}
    # Define the prompt message for the user.
    profile_prompt = "Select processing profile (1: deep_inspection, 2: fast_scan) (default: 1): "
    # Get the user's choice.
    profile_choice_num = input(profile_prompt).strip()
    # Look up the choice in the dictionary; if not found or empty, use "deep_inspection" as the default.
    profile_name = profile_choices.get(profile_choice_num, "deep_inspection") # Default to deep_inspection
    # Inform the user which profile is being used.
    print(f"  Using profile: {profile_name}")

    # --- Get Fiber Type Key ---
    # Print a section header for selecting the fiber type.
    print("\n--- Fiber Type for Rules ---")
    # Define the prompt asking for the fiber type key, which is used to load the correct pass/fail rules.
    fiber_type_prompt = "Enter fiber type key for pass/fail rules (e.g., single_mode_pc, multi_mode_pc) (default: single_mode_pc): "
    # Get the user's input.
    fiber_type_key = input(fiber_type_prompt).strip()
    # Check if the user entered anything.
    if not fiber_type_key:
        # If the input was empty, assign the default value "single_mode_pc".
        fiber_type_key = "single_mode_pc" # Default to single_mode_pc
    # Inform the user which fiber type key is being used.
    print(f"  Using fiber type key: {fiber_type_key}")

    # --- Default Config and Calibration File Paths ---
    # These paths are determined relative to this script's location, providing a sensible default.
    # Get the directory where the current script is located.
    script_dir = Path(__file__).resolve().parent
    # Construct the default path for the configuration file.
    default_config_file_path = script_dir / "config.json"
    # Construct the default path for the calibration file.
    default_calibration_file_path = script_dir / "calibration.json"
    
    # Print a section header for configuration files.
    print(f"\n--- Configuration Files (Expected Location) ---")
    # Create the prompt for the config file, showing the user the default path.
    config_file_interactive_prompt = f"Enter path to config file (default: '{default_config_file_path}'): "
    # Get the user's input for the config file path.
    config_file_str = input(config_file_interactive_prompt).strip()
    # If the user entered a path, use it; otherwise, use the default path.
    config_file = Path(config_file_str) if config_file_str else default_config_file_path
    # This re-validates the config file path, forcing the user to re-enter if the file doesn't exist.
    config_file = get_validated_path(f"Re-enter path to config file (must exist): ", is_dir=False, check_exists=True) if not config_file.is_file() and config_file_str else config_file


    # Create the prompt for the calibration file, showing the default and noting it's skippable.
    calibration_file_interactive_prompt = f"Enter path to calibration file (default: '{default_calibration_file_path}', press Enter to skip if not used/exists): "
    # Get the user's input for the calibration file path.
    calibration_file_str = input(calibration_file_interactive_prompt).strip()
    # If the user entered a path, use it; otherwise, use the default path.
    calibration_file = Path(calibration_file_str) if calibration_file_str else default_calibration_file_path
    # The existence of the calibration file is optional, as the main script may have fallbacks.
    # Check if the user provided a path or if the default file exists.
    if calibration_file_str or default_calibration_file_path.exists(): # If user provided one or default exists
         # Inform the user which calibration file will be used and that the main process will handle checking it.
         print(f"  Using calibration file: '{calibration_file.resolve()}' (existence will be checked by main process).")
    # If no file was specified and the default doesn't exist.
    else:
        # Inform the user that no specific calibration file was found and the main process will use its own defaults.
        print(f"  No specific calibration file provided, or default '{default_calibration_file_path}' not found. Main process may use internal defaults or skip.")


    # Print a separator to indicate the setup phase is complete.
    print("\n--- Preparing for Inspection ---")
    # Inform the user that the parameters are collected and the main process is about to start.
    print("Parameters collected. Attempting to start the inspection process via main.py...")

    # Create an instance of the ArgsSimulator class, populating it with all the collected parameters.
    simulated_args = ArgsSimulator(
        input_dir=input_dir,
        output_dir=output_dir,
        config_file=config_file.resolve(), # Pass the fully resolved, absolute path for clarity.
        calibration_file=calibration_file.resolve(), # Pass the fully resolved, absolute path.
        profile=profile_name,
        fiber_type=fiber_type_key,
        core_dia_um=core_dia_um,
        clad_dia_um=clad_dia_um
    )

    # Print a header for the summary of collected parameters.
    print("\nCollected Parameters for D-Scope Blink:")
    # Print the input directory that will be used.
    print(f"  Input Directory: {simulated_args.input_dir}")
    # Print the output directory.
    print(f"  Output Directory: {simulated_args.output_dir}")
    # Print the configuration file path.
    print(f"  Config File: {simulated_args.config_file}")
    # Print the calibration file path.
    print(f"  Calibration File: {simulated_args.calibration_file}")
    # Print the chosen processing profile.
    print(f"  Processing Profile: {simulated_args.profile}")
    # Print the chosen fiber type key.
    print(f"  Fiber Type Key: {simulated_args.fiber_type}")
    # Print the core diameter, or "Not Provided" if it was skipped.
    print(f"  Core Diameter (µm): {simulated_args.core_dia_um if simulated_args.core_dia_um is not None else 'Not Provided'}")
    # Print the cladding diameter, or "Not Provided" if it was skipped.
    print(f"  Cladding Diameter (µm): {simulated_args.clad_dia_um if simulated_args.clad_dia_um is not None else 'Not Provided'}")


    # Initialize a flag to track if the inspection process completes successfully.
    inspection_successful = False
    # Start a try block to catch any exceptions that occur during the execution of the main logic.
    try:
        # Check if the imported main module has a function named 'execute_inspection_run', which is the preferred entry point.
        if hasattr(main_module, 'execute_inspection_run'):
            # Log that the preferred function is being called.
            print("\n[INFO] Calling 'execute_inspection_run' from the imported main module (main.py)...")
            # Call the function, passing the simulated arguments object. 'cast' is used to satisfy the type checker.
            # This line shows a correction was made based on a problem description file. 
            main_module.execute_inspection_run(cast(Any, simulated_args))
            # If the function call completes without an error, set the success flag to True.
            inspection_successful = True
        # If the preferred function is not found, check for a fallback function named 'main_with_args'.
        elif hasattr(main_module, 'main_with_args'):
            # Inform the user that the preferred function was not found and the fallback is being attempted.
            print("\n[INFO] 'execute_inspection_run' not found in main.py.")
            # Continue the informational message.
            print("       Attempting to call 'main_with_args' from the imported main module as an alternative...")
            # Call the fallback function, passing the simulated arguments.
            # This line shows corrections were made based on a problem description file. 
            main_module.main_with_args(cast(Any, simulated_args))
            # If the function call completes without an error, set the success flag to True.
            inspection_successful = True
        # If neither of the suitable functions is found in main.py.
        else:
            # Print a clear error message that a suitable entry point could not be found.
            print("\n❌ [ERROR] Could not find a suitable function ('execute_inspection_run' or 'main_with_args') in the main module (main.py).")
            # Provide detailed instructions on how the developer should modify main.py to integrate with this runner.
            print("       To fully integrate this interactive runner, your 'main.py' file needs to provide")
            # Continue the instructions.
            print("       one of these functions designed to accept parameters programmatically.")
            # Show an example of the preferred function signature.
            print("       For example, in main.py, you could define:")
            # Display the example code for the developer.
            print("         def execute_inspection_run(args_namespace):")
            # Explain what the function should do.
            print("             # ... your main inspection logic using args_namespace ...")
            # Show an example of how to adapt an existing argparse-based main function.
            print("       OR, if you have an existing main() function that parses arguments:")
            # Display the example code for the refactored main function.
            print("         def main_with_args(args_namespace):")
            # Explain its purpose.
            print("             # ... your main inspection logic using args_namespace ...")
            # Show how the original command-line main function would now call the new function.
            print("         def main(): # Your original main for CLI")
            # Example of standard argparse setup.
            print("             import argparse")
            # Example of creating the parser.
            print("             parser = argparse.ArgumentParser(...)")
            # Example of adding arguments.
            print("             # ... add arguments ...")
            # Example of parsing command-line arguments.
            print("             cli_args = parser.parse_args()")
            # Example of calling the logic function with the parsed arguments.
            print("             main_with_args(cli_args)")
            # Final plea for the developer to structure their code correctly.
            print("\n       Please ensure your main.py is structured accordingly.")
            # Conclude the error message.
            print("       Parameters were collected, but the inspection could not be started.")

        # After the main logic has been called (if a function was found).
        if inspection_successful:
            # Print a success message confirming that the main module reported completion.
            print("\n✅ --- Inspection Process Reported as Complete by Main Module ---")

    # This except block is for the initial import at the top of the file, kept here for robustness.
    except ImportError as ie: 
        # Log the error with full traceback information for debugging.
        logging.error(f"Failed to import the main inspection module (main.py): {ie}", exc_info=True)
        # Print a critical error message to the user.
        print(f"\n❌ [CRITICAL ERROR] Failed to import the main inspection module (main.py): {ie}")
        # Provide clear instructions on how to fix the problem.
        print("       Ensure 'main.py' exists in the same directory as 'run.py' and has no import errors itself.")
    # This except block catches any other runtime errors that might come from the main.py function call.
    except Exception as e: # Catch other errors from the called inspection function in main.py
        # Log the unexpected error with full traceback information.
        logging.error(f"An error occurred during the call to the main module's inspection function: {e}", exc_info=True)
        # Print a user-friendly error message about the unexpected failure.
        print(f"\n❌ An unexpected error occurred during the inspection process: {e}")
        # Advise the user on where to look for more detailed error information.
        print("       Please check the log file in the output directory (if created) and the console for more details.")
    # The finally block executes regardless of whether an exception occurred or not.
    finally:
        # Print a concluding message for the interactive session.
        print("\n--- Interactive Inspection Run Concluded ---")
        # Check if the inspection was successful.
        if inspection_successful:
            # If successful, tell the user where to find the results.
            print(f"Check the output directory '{simulated_args.output_dir}' for results and logs.")
        # If the inspection was not successful.
        else:
            # Inform the user that the process may have failed.
            print(f"The inspection process may not have started or completed successfully.")
            # Advise them to check the output directory for any partial logs or results.
            print(f"       If an output directory '{simulated_args.output_dir}' was created by this script, check it for any partial results or logs.")
            # Urge them to review the console output for specific errors, especially about the main.py integration.
            print(f"       Review console messages for specific errors, especially regarding main.py integration.")

# This standard Python construct ensures that the code inside it only runs when the script is executed directly.
if __name__ == "__main__":
    # Configure basic logging for the runner script itself. The main module is expected to have its own detailed logging.
    # Set the logging level to INFO, define a clear format for messages, and include timestamps.
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Start a try block to catch exceptions that might occur during the interactive setup process.
    try:
        # Call the main function to start the interactive session.
        main_interactive()
    # Catch a KeyboardInterrupt, which occurs when the user presses Ctrl+C.
    except KeyboardInterrupt:
        # Print a clean message indicating that the user cancelled the operation.
        print("\n\n[INFO] Interactive inspection setup cancelled by user.")
        # Exit the script with a status code of 0, indicating a clean exit.
        sys.exit(0)
    # Catch any other unexpected exceptions that were not handled within main_interactive.
    except Exception as e:
        # Log the critical failure with the full traceback for debugging.
        logging.critical(f"Interactive runner encountered a critical failure: {e}", exc_info=True)
        # Print a user-facing critical error message.
        print(f"\n❌ [CRITICAL ERROR] The interactive runner failed unexpectedly: {e}")
        # Exit the script with a status code of 1 to indicate a crash.
        sys.exit(1)