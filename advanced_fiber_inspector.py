#!/usr/bin/env python3
"""
Advanced Fiber Optic End Face Defect Detection System
=====================================================
This script implements a highly accurate, multi-method approach to detecting defects
on fiber optic connector end faces. It combines DO2MR (Difference of Min-Max Ranking)
for region-based defects and LEI (Linear Enhancement Inspector) for scratch detection,
along with other CV techniques, and provides detailed reporting.

Author: Gemini AI
Date: June 4, 2025
Version: 1.1
"""

# Import all necessary libraries for image processing, numerical operations, and visualization
import cv2  # OpenCV for image processing operations
import numpy as np  # NumPy for efficient numerical computations
import matplotlib.pyplot as plt  # Matplotlib for generating plots and visualizations
import os  # Operating system interface for file operations
import csv  # CSV file handling for report generation
import json  # JSON handling for configuration and calibration data
from datetime import datetime  # Datetime for timestamping operations
from pathlib import Path  # Path handling for cross-platform compatibility
import warnings  # Warning handling to suppress non-critical warnings
from typing import Dict, List, Tuple, Optional, Union, Any  # Type hints for better code clarity
import time  # Time tracking for performance monitoring
from dataclasses import dataclass, field  # Dataclasses for structured data
import pandas as pd # Pandas for easy CSV writing of batch summary

# Suppress warnings that might clutter the output
warnings.filterwarnings('ignore') # Ignores runtime warnings, e.g., from division by zero in calculations.

# --- Data Structures (from Part 1) ---

@dataclass
class FiberSpecifications:
    """Data structure to hold user-provided or default fiber optic specifications."""
    core_diameter_um: Optional[float] = None  # Diameter of the fiber core in micrometers.
    cladding_diameter_um: Optional[float] = 125.0  # Diameter of the fiber cladding in micrometers (default for many fibers).
    ferrule_diameter_um: Optional[float] = 250.0 # Outer diameter of the ferrule in micrometers (approximate).
    fiber_type: str = "unknown"  # Type of fiber, e.g., "single-mode", "multi-mode".

@dataclass
class ZoneDefinition:
    """Data structure to define parameters for a fiber zone."""
    name: str  # Name of the zone (e.g., "core", "cladding").
    r_min_factor_or_um: float # Minimum radius factor (relative to main radius) or absolute radius in um.
    r_max_factor_or_um: float # Maximum radius factor (relative to main radius) or absolute radius in um.
    color_bgr: Tuple[int, int, int]  # BGR color for visualizing this zone.
    max_defect_size_um: Optional[float] = None # Maximum allowable defect size in this zone in micrometers (for pass/fail).
    defects_allowed: bool = True # Whether defects are generally allowed in this zone.

@dataclass
class DetectedZoneInfo:
    """Data structure to hold information about a detected zone in an image."""
    name: str # Name of the zone.
    center_px: Tuple[int, int]  # Center coordinates (x, y) in pixels.
    radius_px: float  # Radius in pixels (typically r_max_px for the zone).
    radius_um: Optional[float] = None  # Radius in micrometers (if conversion is available).
    mask: Optional[np.ndarray] = None # Binary mask for the zone.

@dataclass
class DefectMeasurement:
    """Data structure for defect measurements."""
    value_px: Optional[float] = None # Measurement in pixels.
    value_um: Optional[float] = None # Measurement in micrometers.

@dataclass
class DefectInfo:
    """Data structure to hold detailed information about a detected defect."""
    # Non-default fields first
    defect_id: int  # Unique identifier for the defect within an image.
    zone_name: str  # Name of the zone where the defect is primarily located.
    defect_type: str  # Type of defect (e.g., "Region", "Scratch").
    centroid_px: Tuple[int, int]  # Centroid coordinates (x, y) in pixels.
    bounding_box_px: Tuple[int, int, int, int]  # Bounding box (x, y, width, height) in pixels.
    # Fields with default values
    area: DefectMeasurement = field(default_factory=DefectMeasurement) # Area of the defect.
    perimeter: DefectMeasurement = field(default_factory=DefectMeasurement) # Perimeter of the defect.
    major_dimension: DefectMeasurement = field(default_factory=DefectMeasurement) # Primary dimension.
    minor_dimension: DefectMeasurement = field(default_factory=DefectMeasurement) # Secondary dimension.
    confidence_score: float = 0.0  # Confidence score for the detection (0.0 to 1.0).
    detection_methods: List[str] = field(default_factory=list)  # List of methods that identified this defect.
    contour: Optional[np.ndarray] = None # The contour of the defect in pixels.

@dataclass
class ImageAnalysisStats:
    """Statistics for a single image analysis."""
    total_defects: int = 0 # Total number of defects found.
    core_defects: int = 0 # Number of defects in the core.
    cladding_defects: int = 0 # Number of defects in the cladding.
    ferrule_defects: int = 0 # Number of defects in the ferrule_contact zone.
    adhesive_defects: int = 0 # Number of defects in the adhesive zone.
    processing_time_s: float = 0.0 # Time taken to process the image in seconds.
    status: str = "Pending" # Pass/Fail/Review status.
    microns_per_pixel: Optional[float] = None # Calculated conversion ratio for this image (µm/px).

@dataclass
class ImageResult:
    """Data structure to store all results for a single processed image."""
    filename: str # Original filename of the image.
    timestamp: datetime # Timestamp of when the analysis was performed.
    fiber_specs_used: FiberSpecifications # Fiber specifications used for this image.
    operating_mode: str # "PIXEL_ONLY" or "MICRON_CALCULATED" or "MICRON_INFERRED".
    detected_zones: Dict[str, DetectedZoneInfo] = field(default_factory=dict) # Information about detected zones.
    defects: List[DefectInfo] = field(default_factory=list) # List of detected defects.
    stats: ImageAnalysisStats = field(default_factory=ImageAnalysisStats) # Summary statistics for the image.
    annotated_image_path: Optional[Path] = None # Path to the saved annotated image.
    report_csv_path: Optional[Path] = None # Path to the saved CSV report for this image.
    histogram_path: Optional[Path] = None # Path to the saved defect distribution histogram.
    error_message: Optional[str] = None # Error message if processing failed.
    intermediate_defect_maps: Dict[str, np.ndarray] = field(default_factory=dict) # For debugging.

# --- Configuration Class (from Part 1) ---
@dataclass
class InspectorConfig:
    """Class to hold all configuration parameters for the fiber inspection process."""
    OUTPUT_DIR_NAME: str = "fiber_inspection_output" # Name of the directory to save results.
    MIN_DEFECT_AREA_PX: int = 10  # Minimum area in pixels for a contour to be considered a defect.
    PERFORM_CALIBRATION: bool = False # Whether to attempt system calibration with a target.
    CALIBRATION_IMAGE_PATH: Optional[str] = None # Path to the calibration target image.
    CALIBRATION_DOT_SPACING_UM: float = 10.0 # Known spacing of dots on calibration target in microns.
    CALIBRATION_FILE_JSON: str = "calibration_data.json" # File to save/load calibration data.
    DEFAULT_ZONES: List[ZoneDefinition] = field(default_factory=lambda: [ # Default zone definitions.
        ZoneDefinition(name="core", r_min_factor_or_um=0.0, r_max_factor_or_um=0.4, # e.g. 50um core for 125um cladding
                       color_bgr=(255, 0, 0), max_defect_size_um=5.0), # Blue for core
        ZoneDefinition(name="cladding", r_min_factor_or_um=0.4, r_max_factor_or_um=1.0,
                       color_bgr=(0, 255, 0), max_defect_size_um=10.0), # Green for cladding
        ZoneDefinition(name="ferrule_contact", r_min_factor_or_um=1.0, r_max_factor_or_um=2.0, # e.g. up to 250um for ferrule
                       color_bgr=(0, 0, 255), max_defect_size_um=25.0), # Red for ferrule
        ZoneDefinition(name="adhesive", r_min_factor_or_um=2.0, r_max_factor_or_um=2.2, # Area just beyond ferrule
                       color_bgr=(0, 255, 255), max_defect_size_um=50.0, defects_allowed=False) # Yellow for adhesive
    ])
    GAUSSIAN_BLUR_KERNEL_SIZE: Tuple[int, int] = (7, 7)  # Kernel size for Gaussian blur.
    GAUSSIAN_BLUR_SIGMA: int = 2  # Sigma for Gaussian blur.
    BILATERAL_FILTER_D: int = 9 # Diameter of each pixel neighborhood for bilateral filter.
    BILATERAL_FILTER_SIGMA_COLOR: int = 75 # Filter sigma in the color space.
    BILATERAL_FILTER_SIGMA_SPACE: int = 75 # Filter sigma in the coordinate space.
    CLAHE_CLIP_LIMIT: float = 2.0 # Clip limit for CLAHE.
    CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8) # Tile grid size for CLAHE.
    HOUGH_DP_VALUES: List[float] = field(default_factory=lambda: [1.0, 1.2, 1.5]) # Inverse ratio of accumulator resolution.
    HOUGH_MIN_DIST_FACTOR: float = 0.1 # Minimum distance between centers of detected circles, as a factor of image smaller dimension.
    HOUGH_PARAM1_VALUES: List[int] = field(default_factory=lambda: [50, 70, 100])  # Upper threshold for Canny edge detector in Hough.
    HOUGH_PARAM2_VALUES: List[int] = field(default_factory=lambda: [25, 30, 40])  # Accumulator threshold for circle detection.
    HOUGH_MIN_RADIUS_FACTOR: float = 0.05 # Minimum circle radius as a factor of image smaller dimension.
    HOUGH_MAX_RADIUS_FACTOR: float = 0.6 # Maximum circle radius as a factor of image smaller dimension.
    CIRCLE_CONFIDENCE_THRESHOLD: float = 0.3 # Minimum confidence for a detected circle to be considered valid.
    DO2MR_KERNEL_SIZES: List[Tuple[int, int]] = field(default_factory=lambda: [(5, 5), (9, 9), (13, 13)]) # Structuring element sizes.
    DO2MR_GAMMA_VALUES: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0]) # Sensitivity parameter for thresholding residual.
    DO2MR_MEDIAN_BLUR_KERNEL_SIZE: int = 5 # Kernel size for median blur on residual image.
    DO2MR_MORPH_OPEN_KERNEL_SIZE: Tuple[int, int] = (3, 3) # Kernel for morphological opening post-threshold.
    LEI_KERNEL_LENGTHS: List[int] = field(default_factory=lambda: [11, 17, 23]) # Lengths of the linear detector.
    LEI_ANGLE_STEP: int = 15  # Angular resolution for scratch detection (degrees).
    LEI_THRESHOLD_FACTOR: float = 2.0 # Factor for Otsu or adaptive thresholding on response map.
    LEI_MORPH_CLOSE_KERNEL_SIZE: Tuple[int, int] = (5, 1) # Kernel for morphological closing (elongated for scratches). (length, thickness)
    LEI_MIN_SCRATCH_AREA_PX: int = 15 # Minimum area for a scratch.
    CANNY_LOW_THRESHOLD: int = 50 # Low threshold for Canny edge detection.
    CANNY_HIGH_THRESHOLD: int = 150 # High threshold for Canny edge detection.
    ADAPTIVE_THRESH_BLOCK_SIZE: int = 11 # Block size for adaptive thresholding.
    ADAPTIVE_THRESH_C: int = 2 # Constant subtracted from the mean in adaptive thresholding.
    MIN_METHODS_FOR_CONFIRMED_DEFECT: int = 2 # Min number of methods that must detect a defect.
    CONFIDENCE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: { # Weights for different detection methods.
        "do2mr": 1.0, "lei": 1.0, "canny": 0.6, "adaptive_thresh": 0.7, "otsu_global": 0.5,
    })
    DEFECT_COLORS: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: { # BGR colors for defect types.
        "Region": (0, 255, 255), "Scratch": (255, 0, 255), "Contamination": (255, 165, 0),
        "Pit": (0, 128, 255), "Chip": (128, 0, 128), "Linear Region": (255,105,180) # Hot Pink for Linear Region
    })
    FONT_SCALE: float = 0.5 # Font scale for annotations.
    LINE_THICKNESS: int = 1 # Line thickness for drawing.
    BATCH_SUMMARY_FILENAME: str = "batch_inspection_summary.csv" # Filename for the overall batch summary.
    DETAILED_REPORT_PER_IMAGE: bool = True # Whether to generate a detailed CSV for each image.
    SAVE_ANNOTATED_IMAGE: bool = True # Whether to save the annotated image.
    SAVE_DEFECT_MAPS: bool = False # Whether to save intermediate defect maps (for debugging).
    SAVE_HISTOGRAM: bool = True # Whether to save the polar defect distribution histogram.

# --- Utility Functions (from Part 1) ---
def _log_message(message: str, level: str = "INFO"):
    """Prints a timestamped log message to the console."""
    # Get current time in a specific format.
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    # Print the formatted log message.
    print(f"[{current_time}] [{level.upper()}] {message}")

def _start_timer() -> float:
    """Returns the current time to start a timer."""
    # Get high-resolution performance counter time.
    return time.perf_counter()

def _log_duration(operation_name: str, start_time: float, image_result: Optional[ImageResult] = None):
    """Logs the duration of an operation."""
    # Calculate the elapsed time.
    duration = time.perf_counter() - start_time
    # Log the duration message.
    _log_message(f"Operation '{operation_name}' completed in {duration:.4f} seconds.")
    # Placeholder for storing timing in ImageResult if needed later
    # if image_result and hasattr(image_result, 'timing_log'):
    #     image_result.timing_log[operation_name] = duration
    # Return the duration.
    return duration

# --- Main Inspector Class (Combines Part 1, 2, and new methods for Part 3) ---
class FiberInspector:
    """
    Main class to orchestrate the fiber optic end face inspection process.
    """
    def __init__(self, config: Optional[InspectorConfig] = None):
        """Initializes the FiberInspector instance."""
        self.config = config if config else InspectorConfig() # Store or create default config.
        self.fiber_specs = FiberSpecifications() # Initialize fiber specifications.
        self.pixels_per_micron: Optional[float] = None # To be set by calibration or inference.
        self.operating_mode: str = "PIXEL_ONLY" # Default operating mode.
        self.current_image_result: Optional[ImageResult] = None # Holds results for the image being processed.
        self.batch_results_summary_list: List[Dict[str, Any]] = [] # Stores summary dicts for batch report.
        self.output_dir_path: Path = Path(self.config.OUTPUT_DIR_NAME) # Path to the main output directory.
        self.output_dir_path.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist.
        self.active_zone_definitions: List[ZoneDefinition] = [] # Active zone definitions.
        _log_message("FiberInspector initialized.", level="DEBUG") # Log initialization.
        self._initialize_zone_parameters() # Initialize zone parameters based on config.

    def _get_user_specifications(self):
        """Prompts for fiber specs and updates internal state."""
        start_time = _start_timer() # Start timer.
        _log_message("Starting user specification input...") # Log start.
        print("\n--- Fiber Optic Specifications ---") # Print section header.
        provide_specs_input = input("Provide known fiber specifications (microns)? (y/n, default: n): ").strip().lower() # Get user choice.

        if provide_specs_input == 'y': # If user wants to provide specs.
            _log_message("User chose to provide fiber specifications.") # Log choice.
            try:
                # Prompt for core diameter.
                core_dia_str = input(f"Enter CORE diameter in microns (e.g., 9, 50, 62.5) (optional, press Enter to skip): ").strip()
                if core_dia_str: self.fiber_specs.core_diameter_um = float(core_dia_str) # Convert and store if provided.
                # Prompt for cladding diameter.
                clad_dia_str = input(f"Enter CLADDING diameter in microns (e.g., 125) (default: {self.fiber_specs.cladding_diameter_um}): ").strip()
                if clad_dia_str: self.fiber_specs.cladding_diameter_um = float(clad_dia_str) # Convert and store if provided.
                # Prompt for ferrule diameter.
                ferrule_dia_str = input(f"Enter FERRULE outer diameter in microns (e.g., 250) (default: {self.fiber_specs.ferrule_diameter_um}): ").strip()
                if ferrule_dia_str: self.fiber_specs.ferrule_diameter_um = float(ferrule_dia_str) # Convert and store if provided.
                # Prompt for fiber type.
                self.fiber_specs.fiber_type = input("Enter fiber type (e.g., single-mode, multi-mode) (optional): ").strip()

                if self.fiber_specs.cladding_diameter_um is not None: # If cladding diameter provided.
                    self.operating_mode = "MICRON_CALCULATED" # Set mode.
                    _log_message(f"Operating mode set to MICRON_CALCULATED. Specs: Core={self.fiber_specs.core_diameter_um}, Clad={self.fiber_specs.cladding_diameter_um}, Ferrule={self.fiber_specs.ferrule_diameter_um}, Type='{self.fiber_specs.fiber_type}'.")
                else: # If cladding diameter not provided.
                    self.operating_mode = "PIXEL_ONLY" # Fallback mode.
                    _log_message("Cladding diameter not provided, falling back to PIXEL_ONLY mode.", level="WARNING")
            except ValueError: # Handle invalid input.
                _log_message("Invalid input for diameter. Proceeding in PIXEL_ONLY mode.", level="ERROR")
                self.operating_mode = "PIXEL_ONLY" # Revert to pixel mode.
                self.fiber_specs = FiberSpecifications() # Reset specs.
        else: # If user skips specs.
            self.operating_mode = "PIXEL_ONLY" # Default to pixel mode.
            _log_message("User chose to skip fiber specifications. Operating mode set to PIXEL_ONLY.")
        _log_duration("User Specification Input", start_time) # Log duration.
        self._initialize_zone_parameters() # Re-initialize zone parameters with updated mode/specs.

    def _initialize_zone_parameters(self):
        """Initializes active_zone_definitions based on operating mode and specs."""
        _log_message("Initializing zone parameters...") # Log start.
        self.active_zone_definitions = [] # Clear previous definitions.
        if self.operating_mode == "MICRON_CALCULATED" and self.fiber_specs.cladding_diameter_um is not None: # If micron mode with specs.
            # Calculate radii in microns from diameters.
            core_r_um = self.fiber_specs.core_diameter_um / 2.0 if self.fiber_specs.core_diameter_um else 0.0
            cladding_r_um = self.fiber_specs.cladding_diameter_um / 2.0
            ferrule_r_um = self.fiber_specs.ferrule_diameter_um / 2.0 if self.fiber_specs.ferrule_diameter_um else cladding_r_um * 2.0 # Approx if not given
            adhesive_r_um = ferrule_r_um * 1.1 # Example: adhesive zone 10% larger than ferrule

            # Find corresponding default zone definitions for color and max_defect_size_um
            default_core = next((z for z in self.config.DEFAULT_ZONES if z.name == "core"), None)
            default_cladding = next((z for z in self.config.DEFAULT_ZONES if z.name == "cladding"), None)
            default_ferrule = next((z for z in self.config.DEFAULT_ZONES if z.name == "ferrule_contact"), None)
            default_adhesive = next((z for z in self.config.DEFAULT_ZONES if z.name == "adhesive"), None)

            # Create zone definitions with absolute micron values.
            self.active_zone_definitions = [
                ZoneDefinition(name="core", r_min_factor_or_um=0.0, r_max_factor_or_um=core_r_um,
                               color_bgr=default_core.color_bgr if default_core else (255,0,0), # Default color if not in config.
                               max_defect_size_um=default_core.max_defect_size_um if default_core else 5.0), # Default max size.
                ZoneDefinition(name="cladding", r_min_factor_or_um=core_r_um, r_max_factor_or_um=cladding_r_um,
                               color_bgr=default_cladding.color_bgr if default_cladding else (0,255,0),
                               max_defect_size_um=default_cladding.max_defect_size_um if default_cladding else 10.0),
                ZoneDefinition(name="ferrule_contact", r_min_factor_or_um=cladding_r_um, r_max_factor_or_um=ferrule_r_um,
                               color_bgr=default_ferrule.color_bgr if default_ferrule else (0,0,255),
                               max_defect_size_um=default_ferrule.max_defect_size_um if default_ferrule else 25.0),
                ZoneDefinition(name="adhesive", r_min_factor_or_um=ferrule_r_um, r_max_factor_or_um=adhesive_r_um,
                               color_bgr=default_adhesive.color_bgr if default_adhesive else (0,255,255),
                               max_defect_size_um=default_adhesive.max_defect_size_um if default_adhesive else 50.0,
                               defects_allowed=default_adhesive.defects_allowed if default_adhesive else False)
            ]
            _log_message(f"Zone parameters set for MICRON_CALCULATED: Core R={core_r_um}µm, Clad R={cladding_r_um}µm.")
        else: # PIXEL_ONLY or MICRON_INFERRED (initially uses factors from config)
            self.active_zone_definitions = self.config.DEFAULT_ZONES # Use default factors from config.
            _log_message(f"Zone parameters set to default relative factors for {self.operating_mode} mode.")

    def _get_image_paths_from_user(self) -> List[Path]:
        """Prompts for image directory and returns list of image Paths."""
        start_time = _start_timer() # Start timer.
        _log_message("Starting image path collection...") # Log start.
        image_paths: List[Path] = [] # Initialize list for image paths.
        while True: # Loop until a valid directory with images is provided.
            dir_path_str = input("Enter the path to the directory containing fiber images: ").strip() # Get directory path from user.
            image_dir = Path(dir_path_str) # Convert string path to Path object.
            if not image_dir.is_dir(): # Check if the path is a valid directory.
                _log_message(f"Error: The path '{image_dir}' is not a valid directory. Please try again.", level="ERROR")
                continue # Ask for input again.
            supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'] # Define supported image extensions.
            for item in image_dir.iterdir(): # Iterate through items in the directory.
                if item.is_file() and item.suffix.lower() in supported_extensions: # Check if item is a file and has a supported extension.
                    image_paths.append(item) # Add valid image path to the list.
            if not image_paths: # If no images were found in the directory.
                _log_message(f"No images found in directory: {image_dir}. Please check the path or directory content.", level="WARNING")
                # Optionally, allow user to exit or try a different path. Here, we'll just loop.
            else: # If images are found.
                _log_message(f"Found {len(image_paths)} images in '{image_dir}'.")
                break # Exit the loop.
        _log_duration("Image Path Collection", start_time) # Log the duration of the operation.
        return image_paths # Return the list of image paths.

    def _load_single_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Loads a single image from the given path."""
        _log_message(f"Loading image: {image_path.name}") # Log the attempt to load an image.
        try:
            image = cv2.imread(str(image_path)) # Read the image using OpenCV.
            if image is None: # Check if the image was loaded successfully.
                _log_message(f"Failed to load image: {image_path}", level="ERROR") # Log error if loading failed.
                return None # Return None on failure.
            # Handle image format (grayscale, BGRA).
            if len(image.shape) == 2: # If image is grayscale.
                _log_message(f"Image '{image_path.name}' is grayscale. Will be used as is or converted if necessary by specific functions.")
            elif image.shape[2] == 4: # If image has an alpha channel (BGRA).
                 _log_message(f"Image '{image_path.name}' has an alpha channel. Converting to BGR.")
                 image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) # Convert BGRA to BGR.
            _log_message(f"Successfully loaded image: {image_path.name} with shape {image.shape}")
            return image # Return the loaded image.
        except Exception as e: # Catch any other exceptions during image loading.
            _log_message(f"An error occurred while loading image {image_path}: {e}", level="ERROR")
            return None # Return None on error.

    # --- Preprocessing Methods (from Part 2) ---
    def _preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Applies various preprocessing techniques to the input image."""
        preprocess_start_time = _start_timer() # Start timer for preprocessing.
        _log_message("Starting image preprocessing...") # Log the start of preprocessing.
        if image is None: _log_message("Input image for preprocessing is None.", level="ERROR"); return {} # Handle None input.
        # Convert to grayscale if color, else work on a copy.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 and image.shape[2] == 3 else image.copy()
        processed_images: Dict[str, np.ndarray] = {'original_gray': gray.copy()} # Store original grayscale.
        # Apply Gaussian Blur.
        try: processed_images['gaussian_blurred'] = cv2.GaussianBlur(gray, self.config.GAUSSIAN_BLUR_KERNEL_SIZE, self.config.GAUSSIAN_BLUR_SIGMA)
        except Exception as e: _log_message(f"Error during Gaussian Blur: {e}", level="WARNING"); processed_images['gaussian_blurred'] = gray.copy()
        # Apply Bilateral Filter.
        try: processed_images['bilateral_filtered'] = cv2.bilateralFilter(gray, self.config.BILATERAL_FILTER_D, self.config.BILATERAL_FILTER_SIGMA_COLOR, self.config.BILATERAL_FILTER_SIGMA_SPACE)
        except Exception as e: _log_message(f"Error during Bilateral Filter: {e}", level="WARNING"); processed_images['bilateral_filtered'] = gray.copy()
        # Apply CLAHE.
        try:
            clahe = cv2.createCLAHE(clipLimit=self.config.CLAHE_CLIP_LIMIT, tileGridSize=self.config.CLAHE_TILE_GRID_SIZE) # Create CLAHE object.
            processed_images['clahe_enhanced'] = clahe.apply(processed_images.get('bilateral_filtered', gray)) # Apply CLAHE.
        except Exception as e: _log_message(f"Error during CLAHE: {e}", level="WARNING"); processed_images['clahe_enhanced'] = gray.copy()
        # Apply Histogram Equalization.
        try: processed_images['hist_equalized'] = cv2.equalizeHist(gray)
        except Exception as e: _log_message(f"Error during Histogram Equalization: {e}", level="WARNING"); processed_images['hist_equalized'] = gray.copy()
        _log_duration("Image Preprocessing", preprocess_start_time, self.current_image_result) # Log duration.
        return processed_images # Return dictionary of processed images.

    # --- Fiber Center and Zone Detection Methods (from Part 2) ---
    def _find_fiber_center_and_radius(self, processed_images: Dict[str, np.ndarray]) -> Optional[Tuple[Tuple[int, int], float]]:
        """Robustly finds the primary circular feature (assumed cladding) center and radius."""
        detection_start_time = _start_timer() # Start timer.
        _log_message("Starting fiber center and radius detection...") # Log start.
        all_detected_circles: List[Tuple[int, int, int, float, str]] = [] # List for candidate circles.
        h, w = processed_images['original_gray'].shape[:2] # Get image dimensions.
        min_dist_circles = int(min(h, w) * self.config.HOUGH_MIN_DIST_FACTOR) # Min distance between circle centers.
        min_r_hough = int(min(h, w) * self.config.HOUGH_MIN_RADIUS_FACTOR) # Min radius for Hough.
        max_r_hough = int(min(h, w) * self.config.HOUGH_MAX_RADIUS_FACTOR) # Max radius for Hough.

        # Iterate over suitable preprocessed images for Hough transform.
        for image_key in ['gaussian_blurred', 'bilateral_filtered', 'clahe_enhanced']:
            img_to_process = processed_images.get(image_key) # Get the image.
            if img_to_process is None: continue # Skip if image not available.
            # Iterate over different Hough Circle parameters.
            for dp in self.config.HOUGH_DP_VALUES:
                for param1 in self.config.HOUGH_PARAM1_VALUES:
                    for param2 in self.config.HOUGH_PARAM2_VALUES:
                        try:
                            # Detect circles.
                            circles = cv2.HoughCircles(img_to_process, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist_circles, param1=param1, param2=param2, minRadius=min_r_hough, maxRadius=max_r_hough)
                            if circles is not None: # If circles are found.
                                circles = np.uint16(np.around(circles)) # Convert parameters to int.
                                for i in circles[0, :]: # Process each detected circle.
                                    cx, cy, r = int(i[0]), int(i[1]), int(i[2]) # Extract parameters.
                                    # Calculate a simple confidence score.
                                    dist_to_img_center = np.sqrt((cx - w//2)**2 + (cy - h//2)**2)
                                    norm_r = r / max_r_hough if max_r_hough > 0 else 0
                                    confidence = (param2 / max(self.config.HOUGH_PARAM2_VALUES)) * 0.5 + norm_r * 0.5 - (dist_to_img_center / (min(h,w)/2)) * 0.2
                                    all_detected_circles.append((cx, cy, r, max(0, min(1, confidence)), image_key)) # Add to list.
                        except Exception as e: _log_message(f"Error in HoughCircles on {image_key}: {e}", level="WARNING") # Log error.

        if not all_detected_circles: # If no circles were detected.
            _log_message("No circles detected by Hough Transform.", level="WARNING")
            _log_duration("Fiber Center Detection (No Circles)", detection_start_time, self.current_image_result)
            return None # Return None.

        all_detected_circles.sort(key=lambda x: x[3], reverse=True) # Sort circles by confidence.
        best_cx, best_cy, best_r, best_conf, src = all_detected_circles[0] # Get the best circle.

        if best_conf < self.config.CIRCLE_CONFIDENCE_THRESHOLD: # Check if confidence meets threshold.
            _log_message(f"Best detected circle confidence ({best_conf:.2f}) from {src} is below threshold ({self.config.CIRCLE_CONFIDENCE_THRESHOLD}).", level="WARNING")
            _log_duration("Fiber Center Detection (Low Confidence)", detection_start_time, self.current_image_result)
            return None # Return None if confidence is too low.

        _log_message(f"Best fiber center detected at ({best_cx}, {best_cy}) with radius {best_r}px. Confidence: {best_conf:.2f} (from {src}).")
        _log_duration("Fiber Center Detection", detection_start_time, self.current_image_result) # Log duration.
        return (best_cx, best_cy), float(best_r) # Return center and radius.

    def _calculate_pixels_per_micron(self, detected_cladding_radius_px: float) -> Optional[float]:
        """Calculates the pixels_per_micron ratio based on specs or inference."""
        calc_start_time = _start_timer() # Start timer.
        _log_message("Calculating pixels per micron...") # Log start.
        calculated_ppm: Optional[float] = None # Initialize ratio.
        if self.operating_mode in ["MICRON_CALCULATED", "MICRON_INFERRED"]: # Check operating mode.
            if self.fiber_specs.cladding_diameter_um and self.fiber_specs.cladding_diameter_um > 0: # Check if cladding spec available.
                if detected_cladding_radius_px > 0: # Check if detected radius is valid.
                    calculated_ppm = (2 * detected_cladding_radius_px) / self.fiber_specs.cladding_diameter_um # Calculate ratio.
                    self.pixels_per_micron = calculated_ppm # Store in instance.
                    if self.current_image_result: self.current_image_result.stats.microns_per_pixel = 1.0 / calculated_ppm if calculated_ppm > 0 else None # Store µm/px in result.
                    _log_message(f"Calculated pixels_per_micron: {calculated_ppm:.4f} px/µm (µm/px: {1/calculated_ppm:.4f}).")
                else: _log_message("Detected cladding radius is zero or negative, cannot calculate px/µm.", level="WARNING")
            else: _log_message("Cladding diameter in microns not specified, cannot calculate px/µm.", level="WARNING")
        else: _log_message("Not in MICRON_CALCULATED or MICRON_INFERRED mode, skipping px/µm calculation.", level="DEBUG")
        _log_duration("Pixels per Micron Calculation", calc_start_time, self.current_image_result) # Log duration.
        return calculated_ppm # Return calculated ratio.

    def _create_zone_masks(self, image_shape: Tuple[int, int], fiber_center_px: Tuple[int, int], detected_cladding_radius_px: float) -> Dict[str, DetectedZoneInfo]:
        """Creates binary masks for each defined fiber zone (core, cladding, ferrule, etc.)."""
        mask_start_time = _start_timer() # Start timer.
        _log_message("Creating zone masks...") # Log start.
        detected_zones_info: Dict[str, DetectedZoneInfo] = {} # Initialize dictionary for zone info.
        h, w = image_shape[:2]; cx, cy = fiber_center_px # Get image dimensions and center.
        y_coords, x_coords = np.ogrid[:h, :w] # Create Y and X coordinate grids.
        dist_sq_map = (x_coords - cx)**2 + (y_coords - cy)**2 # Calculate squared distance from center for each pixel.

        for zone_def in self.active_zone_definitions: # Iterate through active zone definitions.
            r_min_px, r_max_px = 0.0, 0.0 # Initialize radii in pixels.
            r_min_um, r_max_um = None, None # Initialize radii in microns.

            # Determine radii based on operating mode.
            if self.operating_mode == "PIXEL_ONLY" or (self.operating_mode == "MICRON_INFERRED" and not self.pixels_per_micron):
                # Radii are factors of the detected cladding radius.
                r_min_px = zone_def.r_min_factor_or_um * detected_cladding_radius_px
                r_max_px = zone_def.r_max_factor_or_um * detected_cladding_radius_px
                # If in MICRON_INFERRED mode and conversion is available, calculate micron radii.
                if self.operating_mode == "MICRON_INFERRED" and self.pixels_per_micron and self.pixels_per_micron > 0 :
                    r_min_um = r_min_px / self.pixels_per_micron
                    r_max_um = r_max_px / self.pixels_per_micron
            elif self.operating_mode == "MICRON_CALCULATED" or (self.operating_mode == "MICRON_INFERRED" and self.pixels_per_micron):
                # Radii are given directly in microns (from zone_def.r_min/max_factor_or_um), convert to pixels.
                if self.pixels_per_micron and self.pixels_per_micron > 0:
                    r_min_px = zone_def.r_min_factor_or_um * self.pixels_per_micron
                    r_max_px = zone_def.r_max_factor_or_um * self.pixels_per_micron
                    r_min_um = zone_def.r_min_factor_or_um # This is already in um.
                    r_max_um = zone_def.r_max_factor_or_um # This is already in um.
                else: # Fallback if conversion ratio is missing.
                    _log_message(f"Pixels_per_micron not available for zone '{zone_def.name}' in {self.operating_mode}. Mask creation might be inaccurate.", level="WARNING")
                    r_min_px = zone_def.r_min_factor_or_um # Treat as pixels if conversion fails.
                    r_max_px = zone_def.r_max_factor_or_um # Treat as pixels if conversion fails.

            # Create the binary mask for the current zone (annulus).
            zone_mask_np = ((dist_sq_map >= r_min_px**2) & (dist_sq_map < r_max_px**2)).astype(np.uint8) * 255
            # Store the zone information.
            detected_zones_info[zone_def.name] = DetectedZoneInfo(name=zone_def.name, center_px=fiber_center_px, radius_px=r_max_px, radius_um=r_max_um, mask=zone_mask_np)
            _log_message(f"Created mask for zone '{zone_def.name}': r_min={r_min_px:.2f}px, r_max={r_max_px:.2f}px.") # Log details.
        _log_duration("Zone Mask Creation", mask_start_time, self.current_image_result) # Log duration.
        return detected_zones_info # Return dictionary of zone information.

    # --- Defect Detection Algorithms (from Part 2) ---
    def _detect_region_defects_do2mr(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """Detects region-based defects (dirt, pits, contamination) using a DO2MR-inspired method."""
        do2mr_start_time = _start_timer() # Start timer for DO2MR detection.
        _log_message(f"Starting DO2MR region defect detection for zone '{zone_name}'...") # Log start.
        if image_gray is None or zone_mask is None: _log_message("Input image or mask is None for DO2MR.", level="ERROR"); return None # Handle None input.
        masked_image = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask) # Apply the zone mask to the image.
        vote_map = np.zeros_like(image_gray, dtype=np.float32) # Initialize a vote map for combining results.

        # Iterate over configured kernel sizes for DO2MR.
        for kernel_size in self.config.DO2MR_KERNEL_SIZES:
            struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size) # Create structuring element.
            min_filtered = cv2.erode(masked_image, struct_element) # Apply minimum filter (erosion).
            max_filtered = cv2.dilate(masked_image, struct_element) # Apply maximum filter (dilation).
            residual = cv2.subtract(max_filtered, min_filtered) # Calculate residual image.
            blur_ksize = self.config.DO2MR_MEDIAN_BLUR_KERNEL_SIZE # Get median blur kernel size from config.
            res_blurred = cv2.medianBlur(residual, blur_ksize) if blur_ksize > 0 else residual # Apply median blur if kernel size > 0.

            # Iterate over configured gamma values for thresholding.
            for gamma in self.config.DO2MR_GAMMA_VALUES:
                masked_res_vals = res_blurred[zone_mask > 0] # Get residual values within the current zone.
                if masked_res_vals.size == 0: continue # Skip if no values in zone (e.g., empty mask).
                mean_val, std_val = np.mean(masked_res_vals), np.std(masked_res_vals) # Calculate mean and std deviation.
                thresh_val = np.clip(mean_val + gamma * std_val, 0, 255) # Calculate dynamic threshold and clip to 0-255.
                _, defect_mask_pass = cv2.threshold(res_blurred, thresh_val, 255, cv2.THRESH_BINARY) # Apply threshold.
                defect_mask_pass = cv2.bitwise_and(defect_mask_pass, defect_mask_pass, mask=zone_mask) # Ensure defects are within zone.
                open_k = self.config.DO2MR_MORPH_OPEN_KERNEL_SIZE # Get morph open kernel size.
                if open_k[0] > 0 and open_k[1] > 0: # If kernel size is valid.
                    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, open_k) # Create open kernel.
                    defect_mask_pass = cv2.morphologyEx(defect_mask_pass, cv2.MORPH_OPEN, open_kernel) # Apply morphological opening.
                vote_map += (defect_mask_pass / 255.0) # Add to the vote map (normalized).

        num_param_sets = len(self.config.DO2MR_KERNEL_SIZES) * len(self.config.DO2MR_GAMMA_VALUES) # Total number of parameter sets.
        min_votes = max(1, int(num_param_sets * 0.3)) # Minimum votes required (e.g., 30% of sets).
        combined_map = np.where(vote_map >= min_votes, 255, 0).astype(np.uint8) # Create final combined map based on votes.
        _log_duration(f"DO2MR Detection for {zone_name}", do2mr_start_time, self.current_image_result) # Log duration.
        return combined_map # Return the combined defect map.

    def _detect_scratches_lei(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """Detects linear scratches using an LEI-inspired method."""
        lei_start_time = _start_timer() # Start timer for LEI detection.
        _log_message(f"Starting LEI scratch detection for zone '{zone_name}'...") # Log start.
        if image_gray is None or zone_mask is None: _log_message("Input image or mask is None for LEI.", level="ERROR"); return None # Handle None input.
        masked_image = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask) # Apply the zone mask.
        enhanced_image = cv2.equalizeHist(masked_image) # Enhance image contrast using histogram equalization.
        enhanced_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=zone_mask) # Re-apply mask after equalization.
        max_response_map = np.zeros_like(enhanced_image, dtype=np.float32) # Initialize map for max responses.

        # Iterate over different kernel lengths for multi-scale scratch detection.
        for kernel_length in self.config.LEI_KERNEL_LENGTHS:
            # Iterate through angles from 0 to 180 degrees.
            for angle_deg in range(0, 180, self.config.LEI_ANGLE_STEP):
                line_kernel_base = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1)) # Create base linear kernel.
                rot_matrix = cv2.getRotationMatrix2D((kernel_length // 2, 0), angle_deg, 1.0) # Get rotation matrix.
                bbox_size = int(np.ceil(kernel_length * 1.5)) # Calculate bounding box size for rotated kernel.
                rotated_kernel = cv2.warpAffine(line_kernel_base, rot_matrix, (bbox_size, bbox_size)) # Rotate the kernel.
                if np.sum(rotated_kernel) > 0: rotated_kernel = rotated_kernel.astype(np.float32) / np.sum(rotated_kernel) # Normalize kernel.
                else: continue # Skip if kernel sum is zero.
                response = cv2.filter2D(enhanced_image.astype(np.float32), -1, rotated_kernel) # Apply filter.
                max_response_map = np.maximum(max_response_map, response) # Update max response map.

        if np.max(max_response_map) > 0: cv2.normalize(max_response_map, max_response_map, 0, 255, cv2.NORM_MINMAX) # Normalize map to 0-255.
        response_8u = max_response_map.astype(np.uint8) # Convert to 8-bit unsigned integer.
        _, scratch_mask = cv2.threshold(response_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Threshold response map.
        close_k_shape = self.config.LEI_MORPH_CLOSE_KERNEL_SIZE # Get morph close kernel shape from config.
        if close_k_shape[0] > 0 and close_k_shape[1] > 0: # If kernel shape is valid.
            # Using a general elliptical kernel for closing as scratch orientations vary.
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) # Example: 5x5 ellipse.
            scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, close_kernel) # Apply morphological closing.
        scratch_mask = cv2.bitwise_and(scratch_mask, scratch_mask, mask=zone_mask) # Ensure scratches are within zone.
        _log_duration(f"LEI Scratch Detection for {zone_name}", lei_start_time, self.current_image_result) # Log duration.
        return scratch_mask # Return the binary scratch mask.

    def _detect_defects_canny(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """Detects defects using Canny edge detection followed by morphological operations."""
        _log_message(f"Starting Canny defect detection for zone '{zone_name}'...") # Log start.
        edges = cv2.Canny(image_gray, self.config.CANNY_LOW_THRESHOLD, self.config.CANNY_HIGH_THRESHOLD) # Apply Canny edge detection.
        edges_masked = cv2.bitwise_and(edges, edges, mask=zone_mask) # Apply zone mask.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # Define kernel for morphological closing.
        closed_edges = cv2.morphologyEx(edges_masked, cv2.MORPH_CLOSE, kernel) # Apply closing to connect edges.
        _log_message(f"Canny defect detection for zone '{zone_name}' complete.") # Log completion.
        return closed_edges # Return the resulting defect mask.

    def _detect_defects_adaptive_thresh(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """Detects defects using adaptive thresholding."""
        _log_message(f"Starting Adaptive Threshold defect detection for zone '{zone_name}'...") # Log start.
        # Apply adaptive thresholding (Gaussian method, inverted binary to get defects as white).
        adaptive_thresh_mask = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.config.ADAPTIVE_THRESH_BLOCK_SIZE, self.config.ADAPTIVE_THRESH_C)
        defects_masked = cv2.bitwise_and(adaptive_thresh_mask, adaptive_thresh_mask, mask=zone_mask) # Apply zone mask.
        _log_message(f"Adaptive Threshold defect detection for zone '{zone_name}' complete.") # Log completion.
        return defects_masked # Return the resulting defect mask.

    # --- Defect Combination and Analysis (from Part 2) ---
    def _combine_defect_masks(self, defect_maps: Dict[str, Optional[np.ndarray]], image_shape: Tuple[int,int]) -> np.ndarray:
        """Combines defect masks from multiple methods using a voting or weighted scheme."""
        combine_start_time = _start_timer() # Start timer.
        _log_message("Combining defect masks from multiple methods...") # Log start.
        h, w = image_shape # Get image shape.
        vote_map = np.zeros((h, w), dtype=np.float32) # Initialize vote map with zeros.
        # Iterate through each defect map from different methods.
        for method_name, mask in defect_maps.items():
            if mask is not None: # Check if the mask is valid.
                base_method_key = method_name.split('_')[0] # Extract base method name for weight lookup.
                weight = self.config.CONFIDENCE_WEIGHTS.get(base_method_key, 0.5) # Get weight, default to 0.5 if not in config.
                vote_map[mask == 255] += weight # Add weighted vote for detected pixels.
        # Determine the threshold for confirming a defect based on configured min methods.
        confirmation_threshold = float(self.config.MIN_METHODS_FOR_CONFIRMED_DEFECT)
        # Create the final combined binary mask where vote meets threshold.
        combined_mask = np.where(vote_map >= confirmation_threshold, 255, 0).astype(np.uint8)
        _log_duration("Combine Defect Masks", combine_start_time, self.current_image_result) # Log duration.
        return combined_mask # Return the final combined mask.

    def _analyze_defect_contours(self, combined_defect_mask: np.ndarray, original_image_filename: str, all_defect_maps_by_method: Dict[str, Optional[np.ndarray]]) -> List[DefectInfo]:
        """Analyzes contours from the combined defect mask to extract defect properties."""
        analysis_start_time = _start_timer() # Start timer.
        _log_message("Analyzing defect contours...") # Log start.
        detected_defects: List[DefectInfo] = [] # List to store DefectInfo objects.
        # Find contours in the combined defect mask.
        contours, _ = cv2.findContours(combined_defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defect_counter = 0 # Initialize defect ID counter for the current image.

        # Iterate through each detected contour.
        for contour in contours:
            area_px = cv2.contourArea(contour) # Calculate area of the contour.
            if area_px < self.config.MIN_DEFECT_AREA_PX: continue # Skip small contours.
            defect_counter += 1 # Increment defect counter.
            # defect_id_str = f"{Path(original_image_filename).stem}_{defect_counter}" # Create unique defect ID string.
            M = cv2.moments(contour); cx = int(M['m10']/(M['m00']+1e-5)); cy = int(M['m01']/(M['m00']+1e-5)) # Calculate centroid.
            x,y,w,h = cv2.boundingRect(contour); perimeter_px = cv2.arcLength(contour, True) # Get bounding box and perimeter.

            zone_name = "unknown" # Default zone name.
            if self.current_image_result and self.current_image_result.detected_zones: # Check if zone info available.
                for zn, z_info in self.current_image_result.detected_zones.items(): # Iterate zones.
                    if z_info.mask is not None and z_info.mask[cy, cx] > 0: zone_name = zn; break # Assign zone if centroid is in mask.

            aspect_ratio = float(w)/h if h > 0 else 0.0 # Calculate aspect ratio.
            # Determine defect type (Scratch if LEI contributed significantly, else Region/Linear Region).
            is_scratch_type = False # Flag for scratch type.
            if any('lei' in method_name.lower() for method_name in all_defect_maps_by_method.keys()): # Check if LEI was run.
                current_contour_mask = np.zeros_like(combined_defect_mask, dtype=np.uint8) # Create mask for current contour.
                cv2.drawContours(current_contour_mask, [contour], -1, 255, thickness=cv2.FILLED) # Draw contour on mask.
                for method_name_full, method_mask_map in all_defect_maps_by_method.items(): # Iterate method maps.
                    if method_mask_map is not None and 'lei' in method_name_full.lower(): # If LEI map.
                        overlap = cv2.bitwise_and(current_contour_mask, method_mask_map) # Calculate overlap.
                        if np.sum(overlap > 0) > 0.5 * area_px: is_scratch_type = True; break # If >50% overlap, mark as scratch.
            defect_type = "Scratch" if is_scratch_type else ("Region" if aspect_ratio < 3.0 and aspect_ratio > 0.33 else "Linear Region") # Classify type.

            # Identify contributing detection methods.
            contrib_methods = sorted(list(set(mn.split('_')[0] for mn, mm in all_defect_maps_by_method.items() if mm is not None and mm[cy,cx]>0)))
            # Calculate confidence score.
            conf = len(contrib_methods) / len(self.config.CONFIDENCE_WEIGHTS) if len(self.config.CONFIDENCE_WEIGHTS)>0 else 0.0

            # Prepare DefectMeasurement objects for area and perimeter.
            area_meas = DefectMeasurement(value_px=area_px)
            perim_meas = DefectMeasurement(value_px=perimeter_px)
            # Determine major/minor dimensions.
            major_dim_px, minor_dim_px = (max(w,h), min(w,h)) # Default to bbox dimensions.
            if defect_type == "Scratch" and len(contour) >= 5: # For scratches, use minAreaRect if enough points.
                rect = cv2.minAreaRect(contour) # Fit oriented rectangle.
                major_dim_px, minor_dim_px = max(rect[1]), min(rect[1]) # Get length and width.
            elif defect_type == "Region": # For regions, use equivalent diameter.
                major_dim_px = np.sqrt(4 * area_px / np.pi)
                minor_dim_px = major_dim_px # Diameter is same for major/minor.

            major_dim_meas = DefectMeasurement(value_px=major_dim_px) # Major dimension measurement.
            minor_dim_meas = DefectMeasurement(value_px=minor_dim_px) # Minor dimension measurement.

            # Convert measurements to microns if pixels_per_micron is available.
            if self.pixels_per_micron and self.pixels_per_micron > 0:
                area_meas.value_um = area_px / (self.pixels_per_micron**2)
                perim_meas.value_um = perimeter_px / self.pixels_per_micron
                major_dim_meas.value_um = major_dim_px / self.pixels_per_micron if major_dim_px is not None else None
                minor_dim_meas.value_um = minor_dim_px / self.pixels_per_micron if minor_dim_px is not None else None

            # Create DefectInfo object with all collected data.
            defect_info = DefectInfo(defect_id=defect_counter, zone_name=zone_name, defect_type=defect_type, centroid_px=(cx,cy), area=area_meas, perimeter=perim_meas, bounding_box_px=(x,y,w,h), major_dimension=major_dim_meas, minor_dimension=minor_dim_meas, confidence_score=min(conf,1.0), detection_methods=contrib_methods, contour=contour)
            detected_defects.append(defect_info) # Add to list of defects.
        _log_message(f"Analyzed {len(detected_defects)} defects from combined mask.") # Log number of defects analyzed.
        _log_duration("Defect Contour Analysis", analysis_start_time, self.current_image_result) # Log duration.
        return detected_defects # Return list of DefectInfo objects.

    # --- Reporting Methods (Part 3) ---
    def _generate_annotated_image(self, original_bgr_image: np.ndarray, image_res: ImageResult) -> Optional[np.ndarray]:
        """Generates an image with detected zones and defects annotated."""
        _log_message(f"Generating annotated image for {image_res.filename}...") # Log start.
        annotated_image = original_bgr_image.copy() # Create a copy to draw on.

        # Draw detected zones.
        for zone_name, zone_info in image_res.detected_zones.items(): # Iterate through detected zones.
            zone_def = next((zd for zd in self.active_zone_definitions if zd.name == zone_name), None) # Find zone definition for color.
            if zone_def and zone_info.mask is not None: # If definition and mask exist.
                contours, _ = cv2.findContours(zone_info.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours of zone mask.
                cv2.drawContours(annotated_image, contours, -1, zone_def.color_bgr, self.config.LINE_THICKNESS + 1) # Draw zone contours.
                if contours: # If contours exist for labeling.
                    c = contours[0] # Use first contour for text position.
                    text_pos = tuple(c[c[:, :, 1].argmin()][0]) # Topmost point of contour.
                    cv2.putText(annotated_image, zone_name, (text_pos[0], text_pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE * 1.2, zone_def.color_bgr, self.config.LINE_THICKNESS) # Add zone label.

        # Draw detected defects.
        for defect in image_res.defects: # Iterate through defects.
            defect_color = self.config.DEFECT_COLORS.get(defect.defect_type, (255, 255, 255)) # Get defect color.
            x, y, w, h = defect.bounding_box_px # Get bounding box.
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), defect_color, self.config.LINE_THICKNESS) # Draw bounding box.
            if defect.contour is not None: cv2.drawContours(annotated_image, [defect.contour], -1, defect_color, self.config.LINE_THICKNESS) # Draw contour.
            size_info = f"{defect.major_dimension.value_um:.1f}um" if defect.major_dimension.value_um is not None else (f"{defect.major_dimension.value_px:.0f}px" if defect.major_dimension.value_px is not None else "") # Get size info.
            label = f"ID{defect.defect_id}:{defect.defect_type[:3]}:{size_info} (C:{defect.confidence_score:.2f})" # Create label.
            cv2.putText(annotated_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, defect_color, self.config.LINE_THICKNESS) # Add defect label.

        # Add overall status and defect counts to the image.
        cv2.putText(annotated_image, f"File: {image_res.filename}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE*1.1, (230,230,230), self.config.LINE_THICKNESS)
        cv2.putText(annotated_image, f"Status: {image_res.stats.status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE*1.1, (230,230,230), self.config.LINE_THICKNESS)
        cv2.putText(annotated_image, f"Total Defects: {image_res.stats.total_defects}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE*1.1, (230,230,230), self.config.LINE_THICKNESS)
        _log_message(f"Annotated image generated for {image_res.filename}.") # Log completion.
        return annotated_image # Return annotated image.

    def _generate_defect_histogram(self, image_res: ImageResult) -> Optional[plt.Figure]:
        """Generates a polar histogram of defect distribution."""
        _log_message(f"Generating defect histogram for {image_res.filename}...") # Log start.
        # Check for defects and cladding zone (for center).
        if not image_res.defects or not image_res.detected_zones.get("cladding"):
            _log_message("No defects or cladding center not found, skipping histogram.", level="WARNING")
            return None # Return None if data missing.

        cladding_zone_info = image_res.detected_zones.get("cladding") # Get cladding info.
        if not cladding_zone_info or cladding_zone_info.center_px is None: # Check if center exists.
            _log_message("Cladding center is None, cannot generate polar histogram.", level="WARNING")
            return None # Return None if center missing.
        fiber_center_x, fiber_center_y = cladding_zone_info.center_px # Get fiber center.

        angles, radii, defect_plot_colors = [], [], [] # Initialize lists for plot data.
        for defect in image_res.defects: # Iterate defects.
            dx = defect.centroid_px[0] - fiber_center_x; dy = defect.centroid_px[1] - fiber_center_y # Calculate relative position.
            angles.append(np.arctan2(dy, dx)); radii.append(np.sqrt(dx**2 + dy**2)) # Calculate angle and radius.
            # Get BGR color, convert to RGB for matplotlib by reversing and normalizing.
            bgr_color = self.config.DEFECT_COLORS.get(defect.defect_type, (0,0,0)) # Default to black.
            rgb_color_normalized = (bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0) # BGR to RGB normalized.
            defect_plot_colors.append(rgb_color_normalized) # Add color.

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8)) # Create polar plot.
        ax.scatter(angles, radii, c=defect_plot_colors, s=50, alpha=0.75, edgecolors='k') # Scatter plot of defects.

        # Draw zone boundaries.
        for zone_name, zone_info in image_res.detected_zones.items(): # Iterate zones.
            zone_def = next((zd for zd in self.active_zone_definitions if zd.name == zone_name), None) # Get zone definition.
            if zone_def and zone_info.radius_px > 0: # If valid zone.
                # Convert BGR to RGB for plot color.
                plot_color_rgb = (zone_def.color_bgr[2]/255.0, zone_def.color_bgr[1]/255.0, zone_def.color_bgr[0]/255.0)
                ax.plot(np.linspace(0, 2 * np.pi, 100), [zone_info.radius_px] * 100, color=plot_color_rgb, linestyle='--', label=zone_name) # Draw zone circle.
        ax.set_title(f"Defect Distribution: {image_res.filename}", va='bottom') # Set title.
        # Set radial limit based on detected cladding radius or max defect radius.
        max_r_display = cladding_zone_info.radius_px * 2.5 if cladding_zone_info else (max(radii) * 1.1 if radii else 100)
        ax.set_rmax(max_r_display)
        ax.legend() # Show legend.
        plt.tight_layout() # Adjust layout.
        _log_message(f"Defect histogram generated for {image_res.filename}.") # Log completion.
        return fig # Return figure.

    def _save_individual_image_report_csv(self, image_res: ImageResult, image_output_dir: Path):
        """Saves a detailed CSV report for a single image's defects."""
        _log_message(f"Saving individual CSV report for {image_res.filename}...") # Log start.
        report_path = image_output_dir / f"{Path(image_res.filename).stem}_defect_report.csv" # Define CSV path.
        image_res.report_csv_path = report_path # Store path in result object.
        fieldnames = [ # Define CSV fieldnames.
            "Defect_ID", "Zone", "Type", "Centroid_X_px", "Centroid_Y_px",
            "Area_px2", "Area_um2", "Perimeter_px", "Perimeter_um",
            "Major_Dim_px", "Major_Dim_um", "Minor_Dim_px", "Minor_Dim_um",
            "Confidence", "Detection_Methods"
        ]
        with open(report_path, 'w', newline='', encoding='utf-8') as csvfile: # Open CSV for writing.
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames) # Create CSV writer.
            writer.writeheader() # Write header.
            for defect in image_res.defects: # Iterate defects.
                writer.writerow({ # Write defect data row.
                    "Defect_ID": defect.defect_id, "Zone": defect.zone_name, "Type": defect.defect_type,
                    "Centroid_X_px": defect.centroid_px[0], "Centroid_Y_px": defect.centroid_px[1],
                    "Area_px2": f"{defect.area.value_px:.2f}" if defect.area.value_px is not None else "N/A",
                    "Area_um2": f"{defect.area.value_um:.2f}" if defect.area.value_um is not None else "N/A",
                    "Perimeter_px": f"{defect.perimeter.value_px:.2f}" if defect.perimeter.value_px is not None else "N/A",
                    "Perimeter_um": f"{defect.perimeter.value_um:.2f}" if defect.perimeter.value_um is not None else "N/A",
                    "Major_Dim_px": f"{defect.major_dimension.value_px:.2f}" if defect.major_dimension.value_px is not None else "N/A",
                    "Major_Dim_um": f"{defect.major_dimension.value_um:.2f}" if defect.major_dimension.value_um is not None else "N/A",
                    "Minor_Dim_px": f"{defect.minor_dimension.value_px:.2f}" if defect.minor_dimension.value_px is not None else "N/A",
                    "Minor_Dim_um": f"{defect.minor_dimension.value_um:.2f}" if defect.minor_dimension.value_um is not None else "N/A",
                    "Confidence": f"{defect.confidence_score:.3f}",
                    "Detection_Methods": "; ".join(defect.detection_methods)
                })
        _log_message(f"Individual CSV report saved to {report_path}") # Log completion.

    def _save_image_artifacts(self, original_bgr_image: np.ndarray, image_res: ImageResult):
        """Saves all generated artifacts for a single image."""
        _log_message(f"Saving artifacts for {image_res.filename}...") # Log start.
        image_specific_output_dir = self.output_dir_path / Path(image_res.filename).stem # Define image-specific output dir.
        image_specific_output_dir.mkdir(parents=True, exist_ok=True) # Create directory.

        if self.config.SAVE_ANNOTATED_IMAGE: # If save annotated image enabled.
            annotated_img = self._generate_annotated_image(original_bgr_image, image_res) # Generate annotated image.
            if annotated_img is not None: # If successful.
                annotated_path = image_specific_output_dir / f"{Path(image_res.filename).stem}_annotated.jpg" # Define path.
                cv2.imwrite(str(annotated_path), annotated_img) # Save image.
                image_res.annotated_image_path = annotated_path # Store path.
                _log_message(f"Annotated image saved to {annotated_path}") # Log saving.

        if self.config.DETAILED_REPORT_PER_IMAGE and image_res.defects: # If save detailed CSV and defects exist.
            self._save_individual_image_report_csv(image_res, image_specific_output_dir) # Save CSV.

        if self.config.SAVE_HISTOGRAM: # If save histogram enabled.
            histogram_fig = self._generate_defect_histogram(image_res) # Generate histogram.
            if histogram_fig: # If successful.
                histogram_path = image_specific_output_dir / f"{Path(image_res.filename).stem}_histogram.png" # Define path.
                histogram_fig.savefig(str(histogram_path), dpi=150) # Save figure.
                plt.close(histogram_fig) # Close figure.
                image_res.histogram_path = histogram_path # Store path.
                _log_message(f"Defect histogram saved to {histogram_path}") # Log saving.

        if self.config.SAVE_DEFECT_MAPS and image_res.intermediate_defect_maps: # If save defect maps enabled.
            maps_dir = image_specific_output_dir / "defect_maps"; maps_dir.mkdir(exist_ok=True) # Create maps subdir.
            for map_name, defect_map_img in image_res.intermediate_defect_maps.items(): # Iterate maps.
                if defect_map_img is not None: cv2.imwrite(str(maps_dir / f"{map_name}.png"), defect_map_img) # Save map.
            _log_message(f"Intermediate defect maps saved to {maps_dir}") # Log saving.
        _log_message(f"Artifacts saved for {image_res.filename}.") # Log completion.

    def _save_batch_summary_report_csv(self):
        """Saves a summary CSV report for the entire batch."""
        _log_message("Saving batch summary report...") # Log start.
        if not self.batch_results_summary_list: _log_message("No batch results to save.", level="WARNING"); return # Exit if no results.
        summary_path = self.output_dir_path / self.config.BATCH_SUMMARY_FILENAME # Define summary path.
        try:
            summary_df = pd.DataFrame(self.batch_results_summary_list) # Create DataFrame.
            summary_df.to_csv(summary_path, index=False, encoding='utf-8') # Save to CSV.
            _log_message(f"Batch summary report saved to {summary_path}") # Log success.
        except Exception as e: _log_message(f"Error saving batch summary report: {e}", level="ERROR") # Log error.

    # --- Main Orchestration Methods (Part 3) ---
    def process_single_image(self, image_path: Path) -> ImageResult:
        """Orchestrates the full analysis pipeline for a single image."""
        single_image_start_time = _start_timer() # Start timer for this image.
        _log_message(f"--- Starting processing for image: {image_path.name} ---") # Log start.
        # Initialize ImageResult for current image.
        self.current_image_result = ImageResult(filename=image_path.name, timestamp=datetime.now(), fiber_specs_used=self.fiber_specs, operating_mode=self.operating_mode)

        original_bgr_image = self._load_single_image(image_path) # Load image.
        if original_bgr_image is None: # Handle load failure.
            self.current_image_result.error_message = "Failed to load image."; self.current_image_result.stats.status = "Error"
            _log_duration(f"Processing {image_path.name} (Load Error)", single_image_start_time, self.current_image_result)
            return self.current_image_result # Return error result.

        processed_images = self._preprocess_image(original_bgr_image) # Preprocess.
        if not processed_images: # Handle preprocess failure.
            self.current_image_result.error_message = "Image preprocessing failed."; self.current_image_result.stats.status = "Error"
            _log_duration(f"Processing {image_path.name} (Preproc Error)", single_image_start_time, self.current_image_result)
            return self.current_image_result # Return error result.

        center_radius_tuple = self._find_fiber_center_and_radius(processed_images) # Find fiber center/radius.
        if center_radius_tuple is None: # Handle failure to find fiber.
            self.current_image_result.error_message = "Could not detect fiber center/cladding."; self.current_image_result.stats.status = "Error - No Fiber"
            _log_duration(f"Processing {image_path.name} (No Fiber)", single_image_start_time, self.current_image_result)
            return self.current_image_result # Return error result.
        fiber_center_px, detected_cladding_radius_px = center_radius_tuple # Unpack result.

        self._calculate_pixels_per_micron(detected_cladding_radius_px) # Calculate px/µm.
        if self.operating_mode == "MICRON_INFERRED" and not self.pixels_per_micron: # Handle inference failure.
            _log_message("MICRON_INFERRED failed. Effective mode: PIXEL_ONLY.", level="WARNING")
            self.current_image_result.operating_mode = "PIXEL_ONLY (Inference Failed)" # Update mode in result.

        self.current_image_result.detected_zones = self._create_zone_masks(original_bgr_image.shape[:2], fiber_center_px, detected_cladding_radius_px) # Create zone masks.

        all_defect_maps: Dict[str, Optional[np.ndarray]] = {} # Store all raw defect maps.
        for zone_name, zone_info in self.current_image_result.detected_zones.items(): # Iterate zones.
            if zone_info.mask is None: continue # Skip if no mask.
            _log_message(f"Detecting defects in zone: {zone_name}") # Log zone.
            gray_detect = processed_images.get('clahe_enhanced', processed_images['original_gray']) # Select image for detection.
            # Run various detection methods.
            all_defect_maps[f"do2mr_{zone_name}"] = self._detect_region_defects_do2mr(gray_detect, zone_info.mask, zone_name)
            all_defect_maps[f"lei_{zone_name}"] = self._detect_scratches_lei(gray_detect, zone_info.mask, zone_name)
            all_defect_maps[f"canny_{zone_name}"] = self._detect_defects_canny(processed_images.get('gaussian_blurred', gray_detect), zone_info.mask, zone_name)
            all_defect_maps[f"adaptive_thresh_{zone_name}"] = self._detect_defects_adaptive_thresh(processed_images.get('bilateral_filtered', gray_detect), zone_info.mask, zone_name)
        self.current_image_result.intermediate_defect_maps = {k:v for k,v in all_defect_maps.items() if v is not None} # Store intermediate maps.

        final_combined_mask = self._combine_defect_masks(all_defect_maps, original_bgr_image.shape[:2]) # Combine all defect maps.
        self.current_image_result.defects = self._analyze_defect_contours(final_combined_mask, image_path.name, all_defect_maps) # Analyze contours.

        # Update statistics.
        stats = self.current_image_result.stats # Get stats object.
        stats.total_defects = len(self.current_image_result.defects) # Total defects.
        for defect in self.current_image_result.defects: # Count defects per zone.
            if defect.zone_name == "core": stats.core_defects +=1
            elif defect.zone_name == "cladding": stats.cladding_defects +=1
            elif defect.zone_name == "ferrule_contact": stats.ferrule_defects +=1
            elif defect.zone_name == "adhesive": stats.adhesive_defects +=1
        stats.status = "Review" # Default status, can be updated by pass/fail criteria later.

        self._save_image_artifacts(original_bgr_image, self.current_image_result) # Save all artifacts.
        stats.processing_time_s = _log_duration(f"Processing {image_path.name}", single_image_start_time) # Log total time for image.
        _log_message(f"--- Finished processing for image: {image_path.name} ---") # Log end of image processing.
        return self.current_image_result # Return the result for this image.

    def process_image_batch(self, image_paths: List[Path]):
        """Processes a batch of images provided as a list of paths."""
        batch_start_time = _start_timer() # Start timer for batch.
        _log_message(f"Starting batch processing for {len(image_paths)} images...") # Log start of batch.
        self.batch_results_summary_list = [] # Clear previous batch summary.

        # Iterate through each image path in the batch.
        for i, image_path in enumerate(image_paths):
            _log_message(f"Processing image {i+1}/{len(image_paths)}: {image_path.name}") # Log progress.
            image_res = self.process_single_image(image_path) # Process the single image.
            # Create a summary dictionary for this image.
            summary_item = {
                "Filename": image_res.filename, "Timestamp": image_res.timestamp.isoformat(),
                "Operating_Mode": image_res.operating_mode, "Status": image_res.stats.status,
                "Total_Defects": image_res.stats.total_defects, "Core_Defects": image_res.stats.core_defects,
                "Cladding_Defects": image_res.stats.cladding_defects, "Ferrule_Defects": image_res.stats.ferrule_defects,
                "Adhesive_Defects": image_res.stats.adhesive_defects,
                "Processing_Time_s": f"{image_res.stats.processing_time_s:.2f}",
                "Microns_Per_Pixel": f"{1.0/self.pixels_per_micron:.4f}" if self.pixels_per_micron and self.pixels_per_micron > 0 else "N/A", # µm/px
                "Error": image_res.error_message if image_res.error_message else ""
            }
            self.batch_results_summary_list.append(summary_item) # Add summary to the batch list.

            # Reset pixels_per_micron for the next image if it was inferred for the current one.
            # This ensures that if the next image is in MICRON_CALCULATED mode, it uses its own basis,
            # or if it's also MICRON_INFERRED, it starts fresh.
            if image_res.operating_mode == "MICRON_INFERRED" or \
               (image_res.operating_mode == "PIXEL_ONLY (Inference Failed)"):
                self.pixels_per_micron = None # Reset for next image.
                # Re-determine operating mode based on initial user specs for the next image.
                # This logic might need refinement if specs are per-image rather than per-batch.
                # For now, assume initial specs apply to the batch, but inference is per-image.
                if self.fiber_specs.cladding_diameter_um is not None:
                     self.operating_mode = "MICRON_CALCULATED"
                else:
                     self.operating_mode = "PIXEL_ONLY"
                self._initialize_zone_parameters() # Re-initialize zone defs if mode potentially changed for next image.

        self._save_batch_summary_report_csv() # Save the batch summary report.
        _log_duration("Batch Processing", batch_start_time) # Log total batch processing time.
        _log_message(f"--- Batch processing complete. {len(image_paths)} images processed. ---") # Log completion.

# --- Main Execution Function ---
def main():
    """Main function to drive the fiber inspection script."""
    print("=" * 70); print(" Advanced Automated Optical Fiber End Face Inspector"); print("=" * 70) # Welcome message.
    script_start_time = _start_timer() # Start timer for the whole script.

    try:
        config = InspectorConfig() # Create inspector configuration.
        inspector = FiberInspector(config) # Create FiberInspector instance.
        inspector._get_user_specifications() # Get user input for fiber specs.
        image_paths = inspector._get_image_paths_from_user() # Get image paths from user.
        if not image_paths: _log_message("No images to process. Exiting.", level="INFO"); return # Exit if no images.
        inspector.process_image_batch(image_paths) # Process the batch of images.
    except FileNotFoundError as fnf_error: _log_message(f"Error: {fnf_error}", level="CRITICAL") # Handle file not found.
    except ValueError as val_error: _log_message(f"Input Error: {val_error}", level="CRITICAL") # Handle value error.
    except Exception as e: # Handle other unexpected errors.
        _log_message(f"An unexpected error occurred: {e}", level="CRITICAL")
        import traceback; traceback.print_exc() # Print full traceback for debugging.
    finally:
        _log_duration("Total Script Execution", script_start_time) # Log total script execution time.
        print("=" * 70); print("Inspection Run Finished."); print("=" * 70) # End message.

# --- Script Entry Point ---
if __name__ == "__main__":
    main() # Call the main function when the script is executed.
