#!/usr/bin/env python3
"""
Part 1: Configuration, Data Structures, and Initial Setup
Advanced Fiber Optic End Face Defect Detection System
=====================================================
This script implements a highly accurate, multi-method approach to detecting defects
on fiber optic connector end faces. It combines DO2MR (Difference of Min-Max Ranking)
for region-based defects and LEI (Linear Enhancement Inspector) for scratch detection,
with significant improvements over the original research paper implementation.

Author: Gemini AI
Date: June 4, 2025
Version: 1.0
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
# Note: concurrent.futures and pandas will be used in later parts.

# Suppress warnings that might clutter the output
warnings.filterwarnings('ignore') # Ignores runtime warnings, e.g., from division by zero in calculations.

# --- Data Structures ---

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
    # Relative factors to the primary detected radius (e.g., cladding radius) if in pixel_only mode,
    # or absolute radii in microns if specs are provided.
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
    radius_px: float  # Radius in pixels.
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
    defect_id: int  # Unique identifier for the defect within an image.
    zone_name: str  # Name of the zone where the defect is primarily located.
    defect_type: str  # Type of defect (e.g., "Region", "Scratch").
    centroid_px: Tuple[int, int]  # Centroid coordinates (x, y) in pixels.
    area: DefectMeasurement = field(default_factory=DefectMeasurement) # Area of the defect.
    perimeter: DefectMeasurement = field(default_factory=DefectMeasurement) # Perimeter of the defect.
    bounding_box_px: Tuple[int, int, int, int]  # Bounding box (x, y, width, height) in pixels.
    # For scratches: length, width. For regions: equivalent diameter.
    major_dimension: DefectMeasurement = field(default_factory=DefectMeasurement) # Primary dimension (e.g. length of scratch, diameter of pit)
    minor_dimension: DefectMeasurement = field(default_factory=DefectMeasurement) # Secondary dimension (e.g. width of scratch)
    confidence_score: float = 0.0  # Confidence score for the detection (0.0 to 1.0).
    detection_methods: List[str] = field(default_factory=list)  # List of methods that identified this defect.
    contour: Optional[np.ndarray] = None # The contour of the defect in pixels.

@dataclass
class ImageAnalysisStats:
    """Statistics for a single image analysis."""
    total_defects: int = 0 # Total number of defects found.
    core_defects: int = 0 # Number of defects in the core.
    cladding_defects: int = 0 # Number of defects in the cladding.
    ferrule_defects: int = 0 # Number of defects in the ferrule.
    adhesive_defects: int = 0 # Number of defects in the adhesive area.
    processing_time_s: float = 0.0 # Time taken to process the image in seconds.
    status: str = "Pending" # Pass/Fail/Review status.
    microns_per_pixel: Optional[float] = None # Calculated conversion ratio for this image.

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
    # For storing intermediate processing images/masks for debugging or detailed visualization
    intermediate_defect_maps: Dict[str, np.ndarray] = field(default_factory=dict)


# --- Configuration Class ---

@dataclass
class InspectorConfig:
    """Class to hold all configuration parameters for the fiber inspection process."""
    # General Settings
    OUTPUT_DIR_NAME: str = "fiber_inspection_output"  # Name of the directory to save results.
    MIN_DEFECT_AREA_PX: int = 10  # Minimum area in pixels for a contour to be considered a defect.
    PERFORM_CALIBRATION: bool = False # Whether to attempt system calibration with a target.
    CALIBRATION_IMAGE_PATH: Optional[str] = None # Path to the calibration target image.
    CALIBRATION_DOT_SPACING_UM: float = 10.0 # Known spacing of dots on calibration target in microns.
    CALIBRATION_FILE_JSON: str = "calibration_data.json" # File to save/load calibration data.

    # Fiber Zone Definitions (Default for PIXEL_ONLY mode, scaled if cladding detected)
    # Factors are relative to the primary detected radius (assumed to be cladding)
    # For MICRON modes, these r_min/max values will be treated as direct micron values.
    DEFAULT_ZONES: List[ZoneDefinition] = field(default_factory=lambda: [
        ZoneDefinition(name="core", r_min_factor_or_um=0.0, r_max_factor_or_um=0.4, # e.g. 50um core for 125um cladding
                       color_bgr=(255, 0, 0), max_defect_size_um=5.0, defects_allowed=True), # Blue for core
        ZoneDefinition(name="cladding", r_min_factor_or_um=0.4, r_max_factor_or_um=1.0,
                       color_bgr=(0, 255, 0), max_defect_size_um=10.0, defects_allowed=True), # Green for cladding
        ZoneDefinition(name="ferrule_contact", r_min_factor_or_um=1.0, r_max_factor_or_um=2.0, # e.g. up to 250um for ferrule
                       color_bgr=(0, 0, 255), max_defect_size_um=25.0, defects_allowed=True), # Red for ferrule
        ZoneDefinition(name="adhesive", r_min_factor_or_um=2.0, r_max_factor_or_um=2.2, # Area just beyond ferrule
                       color_bgr=(0, 255, 255), max_defect_size_um=50.0, defects_allowed=False) # Yellow for adhesive
    ])

    # Image Preprocessing
    GAUSSIAN_BLUR_KERNEL_SIZE: Tuple[int, int] = (7, 7)  # Kernel size for Gaussian blur.
    GAUSSIAN_BLUR_SIGMA: int = 2  # Sigma for Gaussian blur.
    BILATERAL_FILTER_D: int = 9 # Diameter of each pixel neighborhood for bilateral filter.
    BILATERAL_FILTER_SIGMA_COLOR: int = 75 # Filter sigma in the color space.
    BILATERAL_FILTER_SIGMA_SPACE: int = 75 # Filter sigma in the coordinate space.
    CLAHE_CLIP_LIMIT: float = 2.0 # Clip limit for CLAHE.
    CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8) # Tile grid size for CLAHE.

    # Hough Circle Transform Parameters (multiple sets for robustness)
    HOUGH_DP_VALUES: List[float] = field(default_factory=lambda: [1.0, 1.2, 1.5]) # Inverse ratio of accumulator resolution.
    HOUGH_MIN_DIST_FACTOR: float = 0.1 # Minimum distance between centers of detected circles, as a factor of image smaller dimension.
    HOUGH_PARAM1_VALUES: List[int] = field(default_factory=lambda: [50, 70, 100])  # Upper threshold for Canny edge detector in Hough.
    HOUGH_PARAM2_VALUES: List[int] = field(default_factory=lambda: [25, 30, 40])  # Accumulator threshold for circle detection.
    HOUGH_MIN_RADIUS_FACTOR: float = 0.05 # Minimum circle radius as a factor of image smaller dimension.
    HOUGH_MAX_RADIUS_FACTOR: float = 0.6 # Maximum circle radius as a factor of image smaller dimension.
    CIRCLE_CONFIDENCE_THRESHOLD: float = 0.3 # Minimum confidence for a detected circle to be considered valid.

    # DO2MR (Region-Based Defect) Parameters
    DO2MR_KERNEL_SIZES: List[Tuple[int, int]] = field(default_factory=lambda: [(5, 5), (9, 9), (13, 13)]) # Structuring element sizes.
    DO2MR_GAMMA_VALUES: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0]) # Sensitivity parameter for thresholding residual.
    DO2MR_MEDIAN_BLUR_KERNEL_SIZE: int = 5 # Kernel size for median blur on residual image.
    DO2MR_MORPH_OPEN_KERNEL_SIZE: Tuple[int, int] = (3, 3) # Kernel for morphological opening post-threshold.

    # LEI (Scratch Detection) Parameters
    LEI_KERNEL_LENGTHS: List[int] = field(default_factory=lambda: [11, 17, 23]) # Lengths of the linear detector.
    LEI_ANGLE_STEP: int = 15  # Angular resolution for scratch detection (degrees).
    LEI_THRESHOLD_FACTOR: float = 2.0 # Factor for Otsu or adaptive thresholding on response map.
    LEI_MORPH_CLOSE_KERNEL_SIZE: Tuple[int, int] = (5, 1) # Kernel for morphological closing (elongated for scratches).
    LEI_MIN_SCRATCH_AREA_PX: int = 15 # Minimum area for a scratch.

    # Additional Defect Detection Parameters
    CANNY_LOW_THRESHOLD: int = 50 # Low threshold for Canny edge detection.
    CANNY_HIGH_THRESHOLD: int = 150 # High threshold for Canny edge detection.
    ADAPTIVE_THRESH_BLOCK_SIZE: int = 11 # Block size for adaptive thresholding.
    ADAPTIVE_THRESH_C: int = 2 # Constant subtracted from the mean in adaptive thresholding.

    # Ensemble/Confidence Parameters
    MIN_METHODS_FOR_CONFIRMED_DEFECT: int = 2 # Min number of methods that must detect a defect.
    CONFIDENCE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: { # Weights for different detection methods.
        "do2mr": 1.0,
        "lei": 1.0,
        "canny": 0.6,
        "adaptive_thresh": 0.7,
        "otsu_global": 0.5,
    })

    # Visualization Parameters
    DEFECT_COLORS: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: { # BGR colors for defect types.
        "Region": (0, 255, 255),  # Yellow
        "Scratch": (255, 0, 255),  # Magenta
        "Contamination": (255, 165, 0), # Orange
        "Pit": (0, 128, 255), # Light Orange
        "Chip": (128, 0, 128) # Purple
    })
    FONT_SCALE: float = 0.5 # Font scale for annotations.
    LINE_THICKNESS: int = 1 # Line thickness for drawing.

    # Reporting
    BATCH_SUMMARY_FILENAME: str = "batch_inspection_summary.csv" # Filename for the overall batch summary.
    DETAILED_REPORT_PER_IMAGE: bool = True # Whether to generate a detailed CSV for each image.
    SAVE_ANNOTATED_IMAGE: bool = True # Whether to save the annotated image.
    SAVE_DEFECT_MAPS: bool = False # Whether to save intermediate defect maps (for debugging).
    SAVE_HISTOGRAM: bool = True # Whether to save the polar defect distribution histogram.


# --- Utility Functions ---

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
    """Logs the duration of an operation and stores it."""
    # Calculate the elapsed time.
    duration = time.perf_counter() - start_time
    # Log the duration message.
    _log_message(f"Operation '{operation_name}' completed in {duration:.4f} seconds.")
    # If an ImageResult object is provided, store the timing information there (to be implemented).
    # For now, this is a placeholder for more structured timing logs.
    if image_result:
        # This part would be filled in when ImageResult has a dedicated timing dictionary.
        pass # e.g., image_result.timing_log[operation_name] = duration
    return duration


# --- Main Inspector Class (Initial Structure) ---

class FiberInspector:
    """
    Main class to orchestrate the fiber optic end face inspection process.
    Handles image loading, processing, defect detection, and reporting.
    """

    def __init__(self, config: Optional[InspectorConfig] = None):
        """
        Initializes the FiberInspector instance.
        Args:
            config: An InspectorConfig object. If None, a default config is used.
        """
        # Store the provided or default configuration.
        self.config = config if config else InspectorConfig()
        # Initialize FiberSpecifications to store details about the fiber being inspected.
        self.fiber_specs = FiberSpecifications()
        # Initialize pixels_per_micron; will be set if calibration occurs or specs are used.
        self.pixels_per_micron: Optional[float] = None
        # Operating mode: "PIXEL_ONLY", "MICRON_CALCULATED" (from specs), or "MICRON_INFERRED" (cladding detected).
        self.operating_mode: str = "PIXEL_ONLY"
        # Placeholder for storing results of the current image being processed.
        self.current_image_result: Optional[ImageResult] = None
        # List to store summary results for all images in a batch.
        self.batch_summary_results: List[Dict[str, Any]] = []
        # Path to the main output directory where all results will be saved.
        self.output_dir_path: Path = Path(self.config.OUTPUT_DIR_NAME)
        # Create the output directory if it doesn't exist.
        self.output_dir_path.mkdir(parents=True, exist_ok=True)
        # Log the initialization of the inspector.
        _log_message("FiberInspector initialized.", level="DEBUG")
        # Initialize zone definitions based on the config and potentially user input later.
        self.active_zone_definitions: List[ZoneDefinition] = [] # Will be populated by _initialize_zone_parameters

    def _get_user_specifications(self):
        """
        Prompts the user for fiber specifications (core/cladding diameters, fiber type)
        and updates the internal state (self.fiber_specs, self.operating_mode).
        """
        # Start timer for this operation.
        start_time = _start_timer()
        # Log the start of user input collection.
        _log_message("Starting user specification input...")

        print("\n--- Fiber Optic Specifications ---")
        # Ask user if they want to provide specifications.
        provide_specs_input = input("Do you want to provide known fiber specifications (microns)? (y/n, default: n): ").strip().lower()

        if provide_specs_input == 'y':
            # User wants to provide specifications.
            _log_message("User chose to provide fiber specifications.")
            try:
                # Prompt for core diameter.
                core_dia_str = input(f"Enter CORE diameter in microns (e.g., 9, 50, 62.5) (optional, press Enter to skip): ").strip()
                if core_dia_str: # Check if input is not empty.
                    self.fiber_specs.core_diameter_um = float(core_dia_str) # Convert to float.
                    _log_message(f"User provided core diameter: {self.fiber_specs.core_diameter_um} µm.")

                # Prompt for cladding diameter.
                clad_dia_str = input(f"Enter CLADDING diameter in microns (e.g., 125) (optional, default: {self.fiber_specs.cladding_diameter_um}): ").strip()
                if clad_dia_str: # Check if input is not empty.
                    self.fiber_specs.cladding_diameter_um = float(clad_dia_str) # Convert to float.
                    _log_message(f"User provided cladding diameter: {self.fiber_specs.cladding_diameter_um} µm.")
                else: # If user skips, use default.
                     _log_message(f"User skipped cladding diameter, using default: {self.fiber_specs.cladding_diameter_um} µm.")


                # Prompt for ferrule diameter.
                ferrule_dia_str = input(f"Enter FERRULE outer diameter in microns (e.g., 250) (optional, default: {self.fiber_specs.ferrule_diameter_um}): ").strip()
                if ferrule_dia_str: # Check if input is not empty.
                    self.fiber_specs.ferrule_diameter_um = float(ferrule_dia_str) # Convert to float.
                    _log_message(f"User provided ferrule diameter: {self.fiber_specs.ferrule_diameter_um} µm.")
                else: # If user skips, use default.
                    _log_message(f"User skipped ferrule diameter, using default: {self.fiber_specs.ferrule_diameter_um} µm.")

                # Prompt for fiber type.
                self.fiber_specs.fiber_type = input("Enter fiber type (e.g., single-mode, multi-mode) (optional): ").strip()
                _log_message(f"User provided fiber type: '{self.fiber_specs.fiber_type}'.")

                # Set operating mode based on provided cladding diameter.
                if self.fiber_specs.cladding_diameter_um is not None:
                    self.operating_mode = "MICRON_CALCULATED" # Measurements will be converted to microns.
                    _log_message(f"Operating mode set to MICRON_CALCULATED based on provided cladding diameter.")
                else:
                    self.operating_mode = "PIXEL_ONLY" # Fallback if cladding diameter is crucial but missing.
                    _log_message("Cladding diameter not provided, falling back to PIXEL_ONLY mode.", level="WARNING")

            except ValueError:
                # Handle invalid float conversion.
                _log_message("Invalid input for diameter. Proceeding in PIXEL_ONLY mode.", level="ERROR")
                self.operating_mode = "PIXEL_ONLY" # Revert to pixel mode on error.
                # Reset any partially set specs to ensure consistency.
                self.fiber_specs = FiberSpecifications()
        else:
            # User chose not to provide specifications.
            self.operating_mode = "PIXEL_ONLY" # Default to pixel-based measurements.
            _log_message("User chose to skip fiber specifications. Operating mode set to PIXEL_ONLY.")

        # Log the duration of this operation.
        _log_duration("User Specification Input", start_time)
        # Initialize zone parameters based on the (potentially updated) operating mode and fiber specs
        self._initialize_zone_parameters()


    def _initialize_zone_parameters(self):
        """
        Initializes or adjusts self.active_zone_definitions based on the operating mode
        and provided fiber specifications.
        """
        # Log the start of zone parameter initialization.
        _log_message("Initializing zone parameters...")
        self.active_zone_definitions = [] # Clear any previous definitions.

        if self.operating_mode == "MICRON_CALCULATED" and self.fiber_specs.cladding_diameter_um is not None:
            # Use user-provided micron values to define zones directly in microns.
            # This assumes the config's r_min/max_factor_or_um are intended as micron values in this mode.
            # Example: if core_diameter_um is 9, cladding is 125, ferrule 250
            core_r_um = self.fiber_specs.core_diameter_um / 2.0 if self.fiber_specs.core_diameter_um else 0
            cladding_r_um = self.fiber_specs.cladding_diameter_um / 2.0
            ferrule_r_um = self.fiber_specs.ferrule_diameter_um / 2.0 if self.fiber_specs.ferrule_diameter_um else cladding_r_um * 2.0 # Approx if not given
            adhesive_r_um = ferrule_r_um * 1.1 # Example: adhesive zone 10% larger than ferrule

            self.active_zone_definitions = [
                ZoneDefinition(name="core", r_min_factor_or_um=0.0, r_max_factor_or_um=core_r_um,
                               color_bgr=next(z.color_bgr for z in self.config.DEFAULT_ZONES if z.name=="core"),
                               max_defect_size_um=next(z.max_defect_size_um for z in self.config.DEFAULT_ZONES if z.name=="core")),
                ZoneDefinition(name="cladding", r_min_factor_or_um=core_r_um, r_max_factor_or_um=cladding_r_um,
                               color_bgr=next(z.color_bgr for z in self.config.DEFAULT_ZONES if z.name=="cladding"),
                               max_defect_size_um=next(z.max_defect_size_um for z in self.config.DEFAULT_ZONES if z.name=="cladding")),
                ZoneDefinition(name="ferrule_contact", r_min_factor_or_um=cladding_r_um, r_max_factor_or_um=ferrule_r_um,
                               color_bgr=next(z.color_bgr for z in self.config.DEFAULT_ZONES if z.name=="ferrule_contact"),
                               max_defect_size_um=next(z.max_defect_size_um for z in self.config.DEFAULT_ZONES if z.name=="ferrule_contact")),
                ZoneDefinition(name="adhesive", r_min_factor_or_um=ferrule_r_um, r_max_factor_or_um=adhesive_r_um,
                               color_bgr=next(z.color_bgr for z in self.config.DEFAULT_ZONES if z.name=="adhesive"),
                               max_defect_size_um=next(z.max_defect_size_um for z in self.config.DEFAULT_ZONES if z.name=="adhesive"))
            ]
            _log_message(f"Zone parameters initialized for MICRON_CALCULATED mode: Core R={core_r_um}µm, Clad R={cladding_r_um}µm, Ferrule R={ferrule_r_um}µm.")

        elif self.operating_mode == "MICRON_INFERRED":
            # Zone factors will be applied to the *detected* cladding radius in pixels,
            # and then converted to microns using the inferred pixels_per_micron.
            # The r_min/max_factor_or_um from config are treated as *relative factors* to the detected cladding radius.
            self.active_zone_definitions = self.config.DEFAULT_ZONES
            _log_message("Zone parameters set to default factors for MICRON_INFERRED mode. Actual sizes depend on detection.")

        else: # PIXEL_ONLY mode
            # Use default zone definitions from config, treating r_min/max_factor_or_um as relative factors
            # to the detected primary (cladding) radius in pixels.
            self.active_zone_definitions = self.config.DEFAULT_ZONES
            _log_message("Zone parameters initialized with default relative factors for PIXEL_ONLY mode.")

    def _get_image_paths_from_user(self) -> List[Path]:
        """
        Prompts the user for the path to a directory containing images.
        Returns:
            A list of Path objects for valid image files found in the directory.
        """
        # Start timer for this operation.
        start_time = _start_timer()
        # Log the start of image path collection.
        _log_message("Starting image path collection...")
        image_paths: List[Path] = [] # Initialize list to store image paths.

        while True: # Loop until a valid directory is provided.
            # Prompt user for the directory path.
            dir_path_str = input("Enter the path to the directory containing fiber images: ").strip()
            # Convert string path to a Path object.
            image_dir = Path(dir_path_str)

            # Check if the provided path is a directory.
            if not image_dir.is_dir():
                # Log error if path is not a valid directory.
                _log_message(f"Error: The path '{image_dir}' is not a valid directory. Please try again.", level="ERROR")
                continue # Ask for input again.

            # Define supported image extensions.
            supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            # Iterate through files in the directory and collect valid image paths.
            for item in image_dir.iterdir():
                # Check if item is a file and has a supported extension.
                if item.is_file() and item.suffix.lower() in supported_extensions:
                    image_paths.append(item) # Add valid image path to the list.

            # Check if any images were found.
            if not image_paths:
                # Log warning if no images are found and prompt again.
                _log_message(f"No images found in directory: {image_dir}. Please check the path or directory content.", level="WARNING")
                # Optionally, allow user to exit or try a different path. Here, we'll just loop.
            else:
                # Log the number of images found.
                _log_message(f"Found {len(image_paths)} images in '{image_dir}'.")
                break # Exit loop if images are found.
        # Log the duration of this operation.
        _log_duration("Image Path Collection", start_time)
        # Return the list of image paths.
        return image_paths

    def _load_single_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Loads a single image from the given path.
        Args:
            image_path: The Path object pointing to the image file.
        Returns:
            The loaded image as a NumPy array (BGR format), or None if loading fails.
        """
        # Log the attempt to load an image.
        _log_message(f"Loading image: {image_path.name}")
        try:
            # Read the image using OpenCV.
            image = cv2.imread(str(image_path))
            # Check if the image was loaded successfully.
            if image is None:
                # Log error if image loading failed.
                _log_message(f"Failed to load image: {image_path}", level="ERROR")
                return None
            # Ensure image is in BGR format (OpenCV default for color images).
            # If grayscale, it will remain grayscale. If it has an alpha channel, it might need handling.
            # For simplicity, we assume BGR or Grayscale.
            if len(image.shape) == 2: # Grayscale image
                _log_message(f"Image '{image_path.name}' is grayscale.")
                # Convert to BGR for consistent processing if needed by downstream functions,
                # or handle grayscale specifically in those functions.
                # For now, let's convert to BGR to simplify later annotation steps.
                # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4: # BGRA image
                 _log_message(f"Image '{image_path.name}' has an alpha channel. Converting to BGR.")
                 image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) # Convert BGRA to BGR.

            _log_message(f"Successfully loaded image: {image_path.name} with shape {image.shape}")
            return image # Return the loaded image.
        except Exception as e:
            # Log any other exceptions during image loading.
            _log_message(f"An error occurred while loading image {image_path}: {e}", level="ERROR")
            return None # Return None on error.

# --- Placeholder for Main Execution (will be in Part 3) ---
# if __name__ == "__main__":
#     # This is where the script execution would start.
#     # For Part 1, we are just defining the classes and functions.
#     _log_message("Configuration and Setup (Part 1) loaded.", level="DEBUG")
#
#     # Example of how to use the config:
#     # default_config = InspectorConfig()
#     # _log_message(f"Default min defect area: {default_config.MIN_DEFECT_AREA_PX}px")
#
#     # Example of instantiating the inspector (though its main logic is not yet here)
#     # inspector_instance = FiberInspector(config=default_config)
#     # inspector_instance._get_user_specifications()
#     # image_files = inspector_instance._get_image_paths_from_user()
#     # if image_files:
#     #     first_image = inspector_instance._load_single_image(image_files[0])
#     #     if first_image is not None:
#     #         _log_message("First image loaded successfully for testing.")
#     pass
