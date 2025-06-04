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
    defect_id: int  # Unique identifier for the defect within an image.
    zone_name: str  # Name of the zone where the defect is primarily located.
    defect_type: str  # Type of defect (e.g., "Region", "Scratch").
    centroid_px: Tuple[int, int]  # Centroid coordinates (x, y) in pixels.
    area: DefectMeasurement = field(default_factory=DefectMeasurement) # Area of the defect.
    perimeter: DefectMeasurement = field(default_factory=DefectMeasurement) # Perimeter of the defect.
    bounding_box_px: Tuple[int, int, int, int]  # Bounding box (x, y, width, height) in pixels.
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
    OUTPUT_DIR_NAME: str = "fiber_inspection_output"
    MIN_DEFECT_AREA_PX: int = 10
    PERFORM_CALIBRATION: bool = False
    CALIBRATION_IMAGE_PATH: Optional[str] = None
    CALIBRATION_DOT_SPACING_UM: float = 10.0
    CALIBRATION_FILE_JSON: str = "calibration_data.json"
    DEFAULT_ZONES: List[ZoneDefinition] = field(default_factory=lambda: [
        ZoneDefinition(name="core", r_min_factor_or_um=0.0, r_max_factor_or_um=0.4,
                       color_bgr=(255, 0, 0), max_defect_size_um=5.0), # Blue
        ZoneDefinition(name="cladding", r_min_factor_or_um=0.4, r_max_factor_or_um=1.0,
                       color_bgr=(0, 255, 0), max_defect_size_um=10.0), # Green
        ZoneDefinition(name="ferrule_contact", r_min_factor_or_um=1.0, r_max_factor_or_um=2.0,
                       color_bgr=(0, 0, 255), max_defect_size_um=25.0), # Red
        ZoneDefinition(name="adhesive", r_min_factor_or_um=2.0, r_max_factor_or_um=2.2,
                       color_bgr=(0, 255, 255), max_defect_size_um=50.0, defects_allowed=False) # Yellow
    ])
    GAUSSIAN_BLUR_KERNEL_SIZE: Tuple[int, int] = (7, 7)
    GAUSSIAN_BLUR_SIGMA: int = 2
    BILATERAL_FILTER_D: int = 9
    BILATERAL_FILTER_SIGMA_COLOR: int = 75
    BILATERAL_FILTER_SIGMA_SPACE: int = 75
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)
    HOUGH_DP_VALUES: List[float] = field(default_factory=lambda: [1.0, 1.2, 1.5])
    HOUGH_MIN_DIST_FACTOR: float = 0.1
    HOUGH_PARAM1_VALUES: List[int] = field(default_factory=lambda: [50, 70, 100])
    HOUGH_PARAM2_VALUES: List[int] = field(default_factory=lambda: [25, 30, 40])
    HOUGH_MIN_RADIUS_FACTOR: float = 0.05
    HOUGH_MAX_RADIUS_FACTOR: float = 0.6
    CIRCLE_CONFIDENCE_THRESHOLD: float = 0.3
    DO2MR_KERNEL_SIZES: List[Tuple[int, int]] = field(default_factory=lambda: [(5, 5), (9, 9), (13, 13)])
    DO2MR_GAMMA_VALUES: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0])
    DO2MR_MEDIAN_BLUR_KERNEL_SIZE: int = 5
    DO2MR_MORPH_OPEN_KERNEL_SIZE: Tuple[int, int] = (3, 3)
    LEI_KERNEL_LENGTHS: List[int] = field(default_factory=lambda: [11, 17, 23])
    LEI_ANGLE_STEP: int = 15
    LEI_THRESHOLD_FACTOR: float = 2.0
    LEI_MORPH_CLOSE_KERNEL_SIZE: Tuple[int, int] = (5, 1) # (length, thickness)
    LEI_MIN_SCRATCH_AREA_PX: int = 15
    CANNY_LOW_THRESHOLD: int = 50
    CANNY_HIGH_THRESHOLD: int = 150
    ADAPTIVE_THRESH_BLOCK_SIZE: int = 11
    ADAPTIVE_THRESH_C: int = 2
    MIN_METHODS_FOR_CONFIRMED_DEFECT: int = 2
    CONFIDENCE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "do2mr": 1.0, "lei": 1.0, "canny": 0.6, "adaptive_thresh": 0.7, "otsu_global": 0.5,
    })
    DEFECT_COLORS: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "Region": (0, 255, 255), "Scratch": (255, 0, 255), "Contamination": (255, 165, 0),
        "Pit": (0, 128, 255), "Chip": (128, 0, 128)
    })
    FONT_SCALE: float = 0.5
    LINE_THICKNESS: int = 1
    BATCH_SUMMARY_FILENAME: str = "batch_inspection_summary.csv"
    DETAILED_REPORT_PER_IMAGE: bool = True
    SAVE_ANNOTATED_IMAGE: bool = True
    SAVE_DEFECT_MAPS: bool = False
    SAVE_HISTOGRAM: bool = True

# --- Utility Functions (from Part 1) ---
def _log_message(message: str, level: str = "INFO"):
    """Prints a timestamped log message to the console."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{current_time}] [{level.upper()}] {message}")

def _start_timer() -> float:
    """Returns the current time to start a timer."""
    return time.perf_counter()

def _log_duration(operation_name: str, start_time: float, image_result: Optional[ImageResult] = None):
    """Logs the duration of an operation."""
    duration = time.perf_counter() - start_time
    _log_message(f"Operation '{operation_name}' completed in {duration:.4f} seconds.")
    # Placeholder for storing timing in ImageResult if needed later
    # if image_result and hasattr(image_result, 'timing_log'):
    #     image_result.timing_log[operation_name] = duration
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
        provide_specs_input = input("Provide known fiber specifications (microns)? (y/n, default: n): ").strip().lower()

        if provide_specs_input == 'y': # If user wants to provide specs.
            _log_message("User chose to provide fiber specifications.") # Log choice.
            try:
                core_dia_str = input(f"Enter CORE diameter in microns (e.g., 9, 50, 62.5) (optional, press Enter to skip): ").strip()
                if core_dia_str: self.fiber_specs.core_diameter_um = float(core_dia_str)
                clad_dia_str = input(f"Enter CLADDING diameter in microns (e.g., 125) (default: {self.fiber_specs.cladding_diameter_um}): ").strip()
                if clad_dia_str: self.fiber_specs.cladding_diameter_um = float(clad_dia_str)
                ferrule_dia_str = input(f"Enter FERRULE outer diameter in microns (e.g., 250) (default: {self.fiber_specs.ferrule_diameter_um}): ").strip()
                if ferrule_dia_str: self.fiber_specs.ferrule_diameter_um = float(ferrule_dia_str)
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
        if self.operating_mode == "MICRON_CALCULATED" and self.fiber_specs.cladding_diameter_um is not None:
            core_r_um = self.fiber_specs.core_diameter_um / 2.0 if self.fiber_specs.core_diameter_um else 0.0
            cladding_r_um = self.fiber_specs.cladding_diameter_um / 2.0
            ferrule_r_um = self.fiber_specs.ferrule_diameter_um / 2.0 if self.fiber_specs.ferrule_diameter_um else cladding_r_um * 2.0
            adhesive_r_um = ferrule_r_um * 1.1 # Example: adhesive zone 10% larger than ferrule

            # Find corresponding default zone definitions for color and max_defect_size_um
            default_core = next((z for z in self.config.DEFAULT_ZONES if z.name == "core"), None)
            default_cladding = next((z for z in self.config.DEFAULT_ZONES if z.name == "cladding"), None)
            default_ferrule = next((z for z in self.config.DEFAULT_ZONES if z.name == "ferrule_contact"), None)
            default_adhesive = next((z for z in self.config.DEFAULT_ZONES if z.name == "adhesive"), None)

            self.active_zone_definitions = [
                ZoneDefinition(name="core", r_min_factor_or_um=0.0, r_max_factor_or_um=core_r_um,
                               color_bgr=default_core.color_bgr if default_core else (255,0,0),
                               max_defect_size_um=default_core.max_defect_size_um if default_core else 5.0),
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
        else: # PIXEL_ONLY or MICRON_INFERRED (initially uses factors)
            self.active_zone_definitions = self.config.DEFAULT_ZONES # Use default factors.
            _log_message(f"Zone parameters set to default factors for {self.operating_mode} mode.")

    def _get_image_paths_from_user(self) -> List[Path]:
        """Prompts for image directory and returns list of image Paths."""
        start_time = _start_timer() # Start timer.
        _log_message("Starting image path collection...") # Log start.
        image_paths: List[Path] = [] # Initialize list.
        while True: # Loop until valid directory.
            dir_path_str = input("Enter the path to the directory containing fiber images: ").strip()
            image_dir = Path(dir_path_str) # Convert to Path object.
            if not image_dir.is_dir(): # Check if directory is valid.
                _log_message(f"Error: The path '{image_dir}' is not a valid directory. Please try again.", level="ERROR")
                continue # Retry input.
            supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'] # Supported extensions.
            for item in image_dir.iterdir(): # Iterate files in directory.
                if item.is_file() and item.suffix.lower() in supported_extensions: # Check if file and valid extension.
                    image_paths.append(item) # Add to list.
            if not image_paths: # If no images found.
                _log_message(f"No images found in directory: {image_dir}. Please check the path or directory content.", level="WARNING")
            else: # If images found.
                _log_message(f"Found {len(image_paths)} images in '{image_dir}'.")
                break # Exit loop.
        _log_duration("Image Path Collection", start_time) # Log duration.
        return image_paths # Return list of paths.

    def _load_single_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Loads a single image from path."""
        _log_message(f"Loading image: {image_path.name}") # Log loading attempt.
        try:
            image = cv2.imread(str(image_path)) # Read image.
            if image is None: # Check if loading failed.
                _log_message(f"Failed to load image: {image_path}", level="ERROR")
                return None # Return None on failure.
            if len(image.shape) == 2: # If grayscale.
                _log_message(f"Image '{image_path.name}' is grayscale. Will be used as is or converted if necessary by specific functions.")
            elif image.shape[2] == 4: # If BGRA.
                 _log_message(f"Image '{image_path.name}' has an alpha channel. Converting to BGR.")
                 image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) # Convert to BGR.
            _log_message(f"Successfully loaded image: {image_path.name} with shape {image.shape}")
            return image # Return loaded image.
        except Exception as e: # Catch other exceptions.
            _log_message(f"An error occurred while loading image {image_path}: {e}", level="ERROR")
            return None # Return None on error.

    # --- Preprocessing Methods (from Part 2) ---
    def _preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Applies various preprocessing techniques to the input image."""
        preprocess_start_time = _start_timer() # Start timer.
        _log_message("Starting image preprocessing...") # Log start.
        if image is None: _log_message("Input image for preprocessing is None.", level="ERROR"); return {} # Handle None input.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 and image.shape[2] == 3 else image.copy()
        processed_images: Dict[str, np.ndarray] = {'original_gray': gray.copy()} # Store original gray.
        try: processed_images['gaussian_blurred'] = cv2.GaussianBlur(gray, self.config.GAUSSIAN_BLUR_KERNEL_SIZE, self.config.GAUSSIAN_BLUR_SIGMA)
        except Exception as e: _log_message(f"Error during Gaussian Blur: {e}", level="WARNING"); processed_images['gaussian_blurred'] = gray.copy()
        try: processed_images['bilateral_filtered'] = cv2.bilateralFilter(gray, self.config.BILATERAL_FILTER_D, self.config.BILATERAL_FILTER_SIGMA_COLOR, self.config.BILATERAL_FILTER_SIGMA_SPACE)
        except Exception as e: _log_message(f"Error during Bilateral Filter: {e}", level="WARNING"); processed_images['bilateral_filtered'] = gray.copy()
        try:
            clahe = cv2.createCLAHE(clipLimit=self.config.CLAHE_CLIP_LIMIT, tileGridSize=self.config.CLAHE_TILE_GRID_SIZE)
            processed_images['clahe_enhanced'] = clahe.apply(processed_images.get('bilateral_filtered', gray))
        except Exception as e: _log_message(f"Error during CLAHE: {e}", level="WARNING"); processed_images['clahe_enhanced'] = gray.copy()
        try: processed_images['hist_equalized'] = cv2.equalizeHist(gray)
        except Exception as e: _log_message(f"Error during Histogram Equalization: {e}", level="WARNING"); processed_images['hist_equalized'] = gray.copy()
        _log_duration("Image Preprocessing", preprocess_start_time, self.current_image_result) # Log duration.
        return processed_images # Return processed images.

    # --- Fiber Center and Zone Detection Methods (from Part 2) ---
    def _find_fiber_center_and_radius(self, processed_images: Dict[str, np.ndarray]) -> Optional[Tuple[Tuple[int, int], float]]:
        """Robustly finds the primary circular feature center and radius."""
        detection_start_time = _start_timer() # Start timer.
        _log_message("Starting fiber center and radius detection...") # Log start.
        all_detected_circles: List[Tuple[int, int, int, float, str]] = [] # List for candidates.
        h, w = processed_images['original_gray'].shape[:2] # Image dimensions.
        min_dist_circles = int(min(h, w) * self.config.HOUGH_MIN_DIST_FACTOR) # Min distance between circles.
        min_r_hough = int(min(h, w) * self.config.HOUGH_MIN_RADIUS_FACTOR) # Min radius for Hough.
        max_r_hough = int(min(h, w) * self.config.HOUGH_MAX_RADIUS_FACTOR) # Max radius for Hough.

        for image_key in ['gaussian_blurred', 'bilateral_filtered', 'clahe_enhanced']: # Iterate suitable images.
            img_to_process = processed_images.get(image_key) # Get image.
            if img_to_process is None: continue # Skip if not available.
            for dp in self.config.HOUGH_DP_VALUES: # Iterate Hough dp values.
                for param1 in self.config.HOUGH_PARAM1_VALUES: # Iterate Hough param1 values.
                    for param2 in self.config.HOUGH_PARAM2_VALUES: # Iterate Hough param2 values.
                        try:
                            circles = cv2.HoughCircles(img_to_process, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist_circles, param1=param1, param2=param2, minRadius=min_r_hough, maxRadius=max_r_hough)
                            if circles is not None: # If circles found.
                                circles = np.uint16(np.around(circles)) # Convert to int.
                                for i in circles[0, :]: # Process each circle.
                                    cx, cy, r = int(i[0]), int(i[1]), int(i[2]) # Extract parameters.
                                    dist_to_img_center = np.sqrt((cx - w//2)**2 + (cy - h//2)**2) # Distance to image center.
                                    norm_r = r / max_r_hough if max_r_hough > 0 else 0 # Normalized radius.
                                    confidence = (param2 / max(self.config.HOUGH_PARAM2_VALUES)) * 0.5 + norm_r * 0.5 - (dist_to_img_center / (min(h,w)/2)) * 0.2 # Calculate confidence.
                                    all_detected_circles.append((cx, cy, r, max(0, min(1, confidence)), image_key)) # Add to list.
                        except Exception as e: _log_message(f"Error in HoughCircles on {image_key}: {e}", level="WARNING") # Log error.

        if not all_detected_circles: # If no circles detected.
            _log_message("No circles detected by Hough Transform.", level="WARNING")
            _log_duration("Fiber Center Detection (No Circles)", detection_start_time, self.current_image_result)
            return None # Return None.

        all_detected_circles.sort(key=lambda x: x[3], reverse=True) # Sort by confidence.
        best_cx, best_cy, best_r, best_conf, src = all_detected_circles[0] # Get best circle.

        if best_conf < self.config.CIRCLE_CONFIDENCE_THRESHOLD: # Check confidence threshold.
            _log_message(f"Best circle confidence ({best_conf:.2f}) from {src} is below threshold.", level="WARNING")
            _log_duration("Fiber Center Detection (Low Confidence)", detection_start_time, self.current_image_result)
            return None # Return None if low confidence.

        _log_message(f"Best fiber center: ({best_cx}, {best_cy}), R={best_r}px. Conf: {best_conf:.2f} (from {src}).")
        _log_duration("Fiber Center Detection", detection_start_time, self.current_image_result) # Log duration.
        return (best_cx, best_cy), float(best_r) # Return center and radius.

    def _calculate_pixels_per_micron(self, detected_cladding_radius_px: float) -> Optional[float]:
        """Calculates pixels_per_micron ratio."""
        calc_start_time = _start_timer() # Start timer.
        _log_message("Calculating pixels per micron...") # Log start.
        calculated_ppm: Optional[float] = None # Initialize.
        if self.operating_mode in ["MICRON_CALCULATED", "MICRON_INFERRED"]: # Check mode.
            if self.fiber_specs.cladding_diameter_um and self.fiber_specs.cladding_diameter_um > 0: # Check spec.
                if detected_cladding_radius_px > 0: # Check detected radius.
                    calculated_ppm = (2 * detected_cladding_radius_px) / self.fiber_specs.cladding_diameter_um # Calculate ratio.
                    self.pixels_per_micron = calculated_ppm # Store in instance.
                    if self.current_image_result: self.current_image_result.stats.microns_per_pixel = 1.0 / calculated_ppm if calculated_ppm > 0 else None # Store µm/px.
                    _log_message(f"Calculated px/µm: {calculated_ppm:.4f} (µm/px: {1/calculated_ppm:.4f}).")
                else: _log_message("Detected cladding radius invalid for px/µm calc.", level="WARNING")
            else: _log_message("Cladding diameter spec missing for px/µm calc.", level="WARNING")
        else: _log_message("Not in micron mode, skipping px/µm calc.", level="DEBUG")
        _log_duration("Pixels per Micron Calculation", calc_start_time, self.current_image_result) # Log duration.
        return calculated_ppm # Return ratio.

    def _create_zone_masks(self, image_shape: Tuple[int, int], fiber_center_px: Tuple[int, int], detected_cladding_radius_px: float) -> Dict[str, DetectedZoneInfo]:
        """Creates binary masks for each defined fiber zone."""
        mask_start_time = _start_timer() # Start timer.
        _log_message("Creating zone masks...") # Log start.
        detected_zones_info: Dict[str, DetectedZoneInfo] = {} # Initialize dict for zone info.
        h, w = image_shape[:2]; cx, cy = fiber_center_px # Get dimensions and center.
        y_coords, x_coords = np.ogrid[:h, :w] # Create coordinate grids.
        dist_sq_map = (x_coords - cx)**2 + (y_coords - cy)**2 # Squared distance from center map.

        for zone_def in self.active_zone_definitions: # Iterate active zone definitions.
            r_min_px, r_max_px = 0.0, 0.0 # Initialize pixel radii.
            r_min_um, r_max_um = None, None # Initialize micron radii.

            if self.operating_mode == "PIXEL_ONLY" or (self.operating_mode == "MICRON_INFERRED" and not self.pixels_per_micron):
                r_min_px = zone_def.r_min_factor_or_um * detected_cladding_radius_px # Calculate min radius in px.
                r_max_px = zone_def.r_max_factor_or_um * detected_cladding_radius_px # Calculate max radius in px.
                if self.operating_mode == "MICRON_INFERRED" and self.pixels_per_micron and self.pixels_per_micron > 0 : # If inferred and ppm available
                    r_min_um = r_min_px / self.pixels_per_micron # Convert min radius to um.
                    r_max_um = r_max_px / self.pixels_per_micron # Convert max radius to um.
            elif self.operating_mode == "MICRON_CALCULATED" or (self.operating_mode == "MICRON_INFERRED" and self.pixels_per_micron):
                if self.pixels_per_micron and self.pixels_per_micron > 0: # If ppm available.
                    r_min_px = zone_def.r_min_factor_or_um * self.pixels_per_micron # Convert min um to px.
                    r_max_px = zone_def.r_max_factor_or_um * self.pixels_per_micron # Convert max um to px.
                    r_min_um = zone_def.r_min_factor_or_um # Min radius in um is direct from def.
                    r_max_um = zone_def.r_max_factor_or_um # Max radius in um is direct from def.
                else: # Fallback if ppm missing.
                    _log_message(f"px/µm missing for '{zone_def.name}' in {self.operating_mode}. Using factors as pixels.", level="WARNING")
                    r_min_px = zone_def.r_min_factor_or_um # Use factor as px.
                    r_max_px = zone_def.r_max_factor_or_um # Use factor as px.

            zone_mask_np = ((dist_sq_map >= r_min_px**2) & (dist_sq_map < r_max_px**2)).astype(np.uint8) * 255 # Create mask.
            detected_zones_info[zone_def.name] = DetectedZoneInfo(name=zone_def.name, center_px=fiber_center_px, radius_px=r_max_px, radius_um=r_max_um, mask=zone_mask_np) # Store info.
            _log_message(f"Mask for '{zone_def.name}': r_min={r_min_px:.2f}px, r_max={r_max_px:.2f}px.") # Log details.
        _log_duration("Zone Mask Creation", mask_start_time, self.current_image_result) # Log duration.
        return detected_zones_info # Return zone info.

    # --- Defect Detection Algorithms (from Part 2) ---
    def _detect_region_defects_do2mr(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """Detects region-based defects using DO2MR-inspired method."""
        do2mr_start_time = _start_timer() # Start timer.
        _log_message(f"Starting DO2MR for zone '{zone_name}'...") # Log start.
        if image_gray is None or zone_mask is None: _log_message("Input None for DO2MR.", level="ERROR"); return None # Handle None input.
        masked_image = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask) # Apply zone mask.
        vote_map = np.zeros_like(image_gray, dtype=np.float32) # Initialize vote map.

        for kernel_size in self.config.DO2MR_KERNEL_SIZES: # Iterate kernel sizes.
            struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size) # Create structuring element.
            min_filtered = cv2.erode(masked_image, struct_element) # Min filter.
            max_filtered = cv2.dilate(masked_image, struct_element) # Max filter.
            residual = cv2.subtract(max_filtered, min_filtered) # Calculate residual.
            blur_ksize = self.config.DO2MR_MEDIAN_BLUR_KERNEL_SIZE # Median blur kernel size.
            res_blurred = cv2.medianBlur(residual, blur_ksize) if blur_ksize > 0 else residual # Apply median blur.

            for gamma in self.config.DO2MR_GAMMA_VALUES: # Iterate gamma values.
                masked_res_vals = res_blurred[zone_mask > 0] # Get residual values in zone.
                if masked_res_vals.size == 0: continue # Skip if empty.
                mean_val, std_val = np.mean(masked_res_vals), np.std(masked_res_vals) # Calculate mean and std.
                thresh_val = np.clip(mean_val + gamma * std_val, 0, 255) # Calculate threshold.
                _, defect_mask_pass = cv2.threshold(res_blurred, thresh_val, 255, cv2.THRESH_BINARY) # Apply threshold.
                defect_mask_pass = cv2.bitwise_and(defect_mask_pass, defect_mask_pass, mask=zone_mask) # Apply zone mask.
                open_k = self.config.DO2MR_MORPH_OPEN_KERNEL_SIZE # Morph open kernel size.
                if open_k[0] > 0 and open_k[1] > 0: # If kernel valid.
                    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, open_k) # Create open kernel.
                    defect_mask_pass = cv2.morphologyEx(defect_mask_pass, cv2.MORPH_OPEN, open_kernel) # Apply morph open.
                vote_map += (defect_mask_pass / 255.0) # Add to vote map.

        num_param_sets = len(self.config.DO2MR_KERNEL_SIZES) * len(self.config.DO2MR_GAMMA_VALUES) # Total param sets.
        min_votes = max(1, int(num_param_sets * 0.3)) # Min votes required (e.g., 30%).
        combined_map = np.where(vote_map >= min_votes, 255, 0).astype(np.uint8) # Create combined map.
        _log_duration(f"DO2MR for {zone_name}", do2mr_start_time, self.current_image_result) # Log duration.
        return combined_map # Return combined map.

    def _detect_scratches_lei(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """Detects linear scratches using LEI-inspired method."""
        lei_start_time = _start_timer() # Start timer.
        _log_message(f"Starting LEI for zone '{zone_name}'...") # Log start.
        if image_gray is None or zone_mask is None: _log_message("Input None for LEI.", level="ERROR"); return None # Handle None input.
        masked_image = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask) # Apply zone mask.
        enhanced_image = cv2.equalizeHist(masked_image) # Enhance contrast.
        enhanced_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=zone_mask) # Re-apply mask.
        max_response_map = np.zeros_like(enhanced_image, dtype=np.float32) # Initialize max response map.

        for kernel_length in self.config.LEI_KERNEL_LENGTHS: # Iterate kernel lengths.
            for angle_deg in range(0, 180, self.config.LEI_ANGLE_STEP): # Iterate angles.
                line_kernel_base = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1)) # Create base kernel.
                rot_matrix = cv2.getRotationMatrix2D((kernel_length // 2, 0), angle_deg, 1.0) # Get rotation matrix.
                bbox_size = int(np.ceil(kernel_length * 1.5)) # Calculate bounding box size for rotated kernel.
                rotated_kernel = cv2.warpAffine(line_kernel_base, rot_matrix, (bbox_size, bbox_size)) # Rotate kernel.
                if np.sum(rotated_kernel) > 0: rotated_kernel = rotated_kernel.astype(np.float32) / np.sum(rotated_kernel) # Normalize kernel.
                else: continue # Skip if kernel sum is zero.
                response = cv2.filter2D(enhanced_image.astype(np.float32), -1, rotated_kernel) # Apply filter.
                max_response_map = np.maximum(max_response_map, response) # Update max response.

        if np.max(max_response_map) > 0: cv2.normalize(max_response_map, max_response_map, 0, 255, cv2.NORM_MINMAX) # Normalize response map.
        response_8u = max_response_map.astype(np.uint8) # Convert to 8-bit.
        _, scratch_mask = cv2.threshold(response_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Threshold.
        close_k_shape = self.config.LEI_MORPH_CLOSE_KERNEL_SIZE # Morph close kernel shape.
        if close_k_shape[0] > 0 and close_k_shape[1] > 0: # If kernel valid.
            # Use a general elliptical kernel for closing scratches, as orientation is varied.
            # The config LEI_MORPH_CLOSE_KERNEL_SIZE might be better interpreted as (length, thickness) for a specific orientation,
            # but for a combined response map, a general small closing is safer.
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) # Example general closing kernel.
            scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, close_kernel) # Apply morph close.
        scratch_mask = cv2.bitwise_and(scratch_mask, scratch_mask, mask=zone_mask) # Apply zone mask.
        _log_duration(f"LEI for {zone_name}", lei_start_time, self.current_image_result) # Log duration.
        return scratch_mask # Return scratch mask.

    def _detect_defects_canny(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """Detects defects using Canny edge detection."""
        _log_message(f"Starting Canny for zone '{zone_name}'...") # Log start.
        edges = cv2.Canny(image_gray, self.config.CANNY_LOW_THRESHOLD, self.config.CANNY_HIGH_THRESHOLD) # Apply Canny.
        edges_masked = cv2.bitwise_and(edges, edges, mask=zone_mask) # Apply zone mask.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # Create kernel for closing.
        closed_edges = cv2.morphologyEx(edges_masked, cv2.MORPH_CLOSE, kernel) # Apply morph close.
        _log_message(f"Canny for zone '{zone_name}' complete.") # Log completion.
        return closed_edges # Return result.

    def _detect_defects_adaptive_thresh(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """Detects defects using adaptive thresholding."""
        _log_message(f"Starting Adaptive Thresh for zone '{zone_name}'...") # Log start.
        adaptive_thresh_mask = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.config.ADAPTIVE_THRESH_BLOCK_SIZE, self.config.ADAPTIVE_THRESH_C) # Apply adaptive threshold.
        defects_masked = cv2.bitwise_and(adaptive_thresh_mask, adaptive_thresh_mask, mask=zone_mask) # Apply zone mask.
        _log_message(f"Adaptive Thresh for zone '{zone_name}' complete.") # Log completion.
        return defects_masked # Return result.

    # --- Defect Combination and Analysis (from Part 2) ---
    def _combine_defect_masks(self, defect_maps: Dict[str, Optional[np.ndarray]], image_shape: Tuple[int,int]) -> np.ndarray:
        """Combines defect masks from multiple methods."""
        combine_start_time = _start_timer() # Start timer.
        _log_message("Combining defect masks...") # Log start.
        h, w = image_shape # Get image shape.
        vote_map = np.zeros((h, w), dtype=np.float32) # Initialize vote map.
        for method_name, mask in defect_maps.items(): # Iterate defect maps.
            if mask is not None: # If mask valid.
                base_method_key = method_name.split('_')[0] # Get base method key for weight.
                weight = self.config.CONFIDENCE_WEIGHTS.get(base_method_key, 0.5) # Get weight.
                vote_map[mask == 255] += weight # Add weighted vote.
        confirmation_threshold = float(self.config.MIN_METHODS_FOR_CONFIRMED_DEFECT) # Get confirmation threshold.
        combined_mask = np.where(vote_map >= confirmation_threshold, 255, 0).astype(np.uint8) # Create combined mask.
        _log_duration("Combine Defect Masks", combine_start_time, self.current_image_result) # Log duration.
        return combined_mask # Return combined mask.

    def _analyze_defect_contours(self, combined_defect_mask: np.ndarray, original_image_filename: str, all_defect_maps_by_method: Dict[str, Optional[np.ndarray]]) -> List[DefectInfo]:
        """Analyzes contours from combined_defect_mask to extract properties."""
        analysis_start_time = _start_timer() # Start timer.
        _log_message("Analyzing defect contours...") # Log start.
        detected_defects: List[DefectInfo] = [] # Initialize list for defects.
        contours, _ = cv2.findContours(combined_defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours.
        defect_counter = 0 # Initialize defect counter.

        for contour in contours: # Iterate contours.
            area_px = cv2.contourArea(contour) # Calculate area.
            if area_px < self.config.MIN_DEFECT_AREA_PX: continue # Skip small contours.
            defect_counter += 1 # Increment counter.
            # defect_id_str = f"{Path(original_image_filename).stem}_{defect_counter}" # Create defect ID string.
            M = cv2.moments(contour); cx = int(M['m10']/(M['m00']+1e-5)); cy = int(M['m01']/(M['m00']+1e-5)) # Calculate centroid.
            x,y,w,h = cv2.boundingRect(contour); perimeter_px = cv2.arcLength(contour, True) # Get bounding box and perimeter.

            zone_name = "unknown" # Default zone name.
            if self.current_image_result and self.current_image_result.detected_zones: # Check if zones available.
                for zn, z_info in self.current_image_result.detected_zones.items(): # Iterate zones.
                    if z_info.mask is not None and z_info.mask[cy, cx] > 0: zone_name = zn; break # Assign zone if centroid in mask.

            aspect_ratio = float(w)/h if h > 0 else 0.0 # Calculate aspect ratio.
            # Simple type classification, LEI maps are explicitly for scratches.
            # Check if this defect significantly overlaps with LEI results.
            is_scratch_type = False # Flag for scratch type.
            if 'lei' in [m.split('_')[0] for m in all_defect_maps_by_method.keys()]: # If LEI was run.
                # Create a mask for the current contour
                current_contour_mask = np.zeros_like(combined_defect_mask, dtype=np.uint8)
                cv2.drawContours(current_contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
                for method_name_full, method_mask_map in all_defect_maps_by_method.items():
                    if method_mask_map is not None and 'lei' in method_name_full.lower():
                        overlap = cv2.bitwise_and(current_contour_mask, method_mask_map)
                        if np.sum(overlap > 0) > 0.5 * area_px: # If >50% of contour area overlaps with an LEI map
                            is_scratch_type = True
                            break
            defect_type = "Scratch" if is_scratch_type else ("Region" if aspect_ratio < 3.0 and aspect_ratio > 0.33 else "Linear Region")


            contrib_methods = sorted(list(set(mn.split('_')[0] for mn, mm in all_defect_maps_by_method.items() if mm is not None and mm[cy,cx]>0))) # Get contributing methods.
            conf = len(contrib_methods) / len(self.config.CONFIDENCE_WEIGHTS) if len(self.config.CONFIDENCE_WEIGHTS)>0 else 0.0 # Calculate confidence.

            area_meas = DefectMeasurement(value_px=area_px) # Area measurement.
            perim_meas = DefectMeasurement(value_px=perimeter_px) # Perimeter measurement.
            major_dim_px, minor_dim_px = (max(w,h), min(w,h)) if defect_type != "Scratch" else (max(cv2.minAreaRect(contour)[1]), min(cv2.minAreaRect(contour)[1])) if len(contour) >= 5 else (max(w,h), min(w,h)) # Major/minor dimensions.
            major_dim_meas = DefectMeasurement(value_px=major_dim_px) # Major dimension measurement.
            minor_dim_meas = DefectMeasurement(value_px=minor_dim_px) # Minor dimension measurement.

            if self.pixels_per_micron and self.pixels_per_micron > 0: # If ppm available.
                area_meas.value_um = area_px / (self.pixels_per_micron**2) # Area in um^2.
                perim_meas.value_um = perimeter_px / self.pixels_per_micron # Perimeter in um.
                major_dim_meas.value_um = major_dim_px / self.pixels_per_micron if major_dim_px is not None else None # Major dim in um.
                minor_dim_meas.value_um = minor_dim_px / self.pixels_per_micron if minor_dim_px is not None else None # Minor dim in um.

            defect_info = DefectInfo(defect_id=defect_counter, zone_name=zone_name, defect_type=defect_type, centroid_px=(cx,cy), area=area_meas, perimeter=perim_meas, bounding_box_px=(x,y,w,h), major_dimension=major_dim_meas, minor_dimension=minor_dim_meas, confidence_score=min(conf,1.0), detection_methods=contrib_methods, contour=contour) # Create DefectInfo.
            detected_defects.append(defect_info) # Add to list.
        _log_message(f"Analyzed {len(detected_defects)} defects.") # Log count.
        _log_duration("Defect Contour Analysis", analysis_start_time, self.current_image_result) # Log duration.
        return detected_defects # Return defects.

    # --- Reporting Methods (Part 3) ---
    def _generate_annotated_image(self, original_bgr_image: np.ndarray, image_res: ImageResult) -> Optional[np.ndarray]:
        """Generates an image with detected zones and defects annotated."""
        # Log start of annotation.
        _log_message(f"Generating annotated image for {image_res.filename}...")
        # Create a copy of the original image to draw on.
        annotated_image = original_bgr_image.copy()

        # Draw detected zones.
        for zone_name, zone_info in image_res.detected_zones.items():
            # Find the ZoneDefinition to get the color.
            zone_def = next((zd for zd in self.active_zone_definitions if zd.name == zone_name), None)
            if zone_def and zone_info.mask is not None: # Check if zone definition and mask exist.
                # Draw the contour of the zone mask for better visualization than just a circle.
                contours, _ = cv2.findContours(zone_info.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Draw all contours found for the zone mask.
                cv2.drawContours(annotated_image, contours, -1, zone_def.color_bgr, self.config.LINE_THICKNESS + 1)
                # Put text label for the zone.
                # Find a point on the contour to place text (e.g., topmost point).
                if contours: # If contours exist.
                    # Get the first contour (assuming it's the main one).
                    c = contours[0]
                    # Get the topmost point of the contour.
                    text_pos = tuple(c[c[:, :, 1].argmin()][0]) # Topmost point.
                    # Adjust text position slightly for better visibility.
                    cv2.putText(annotated_image, zone_name, (text_pos[0], text_pos[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE * 1.2, zone_def.color_bgr, self.config.LINE_THICKNESS)

        # Draw detected defects.
        for defect in image_res.defects:
            # Get defect color from config, default to white if not found.
            defect_color = self.config.DEFECT_COLORS.get(defect.defect_type, (255, 255, 255))
            # Draw bounding box.
            x, y, w, h = defect.bounding_box_px
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), defect_color, self.config.LINE_THICKNESS)
            # Draw contour if available
            if defect.contour is not None:
                 cv2.drawContours(annotated_image, [defect.contour], -1, defect_color, self.config.LINE_THICKNESS)

            # Add label: ID, Type, Size (e.g., Area or Major Dim).
            size_info = "" # Initialize size information string.
            # Check if major dimension in microns is available.
            if defect.major_dimension.value_um is not None:
                size_info = f"{defect.major_dimension.value_um:.1f}um"
            # Else, check if major dimension in pixels is available.
            elif defect.major_dimension.value_px is not None:
                size_info = f"{defect.major_dimension.value_px:.0f}px"
            # Create label string.
            label = f"ID{defect.defect_id}:{defect.defect_type[:3]}:{size_info} (C:{defect.confidence_score:.2f})"
            # Put text label near the bounding box.
            cv2.putText(annotated_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        self.config.FONT_SCALE, defect_color, self.config.LINE_THICKNESS)

        # Add overall status and defect counts to the image.
        cv2.putText(annotated_image, f"File: {image_res.filename}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE*1.1, (230,230,230), self.config.LINE_THICKNESS)
        cv2.putText(annotated_image, f"Status: {image_res.stats.status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE*1.1, (230,230,230), self.config.LINE_THICKNESS)
        cv2.putText(annotated_image, f"Total Defects: {image_res.stats.total_defects}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE*1.1, (230,230,230), self.config.LINE_THICKNESS)
        # Log completion.
        _log_message(f"Annotated image generated for {image_res.filename}.")
        return annotated_image # Return the annotated image.

    def _generate_defect_histogram(self, image_res: ImageResult) -> Optional[plt.Figure]:
        """Generates a polar histogram of defect distribution."""
        _log_message(f"Generating defect histogram for {image_res.filename}...") # Log start.
        # Check if there are any defects to plot.
        if not image_res.defects or not image_res.detected_zones.get("cladding"): # Requires cladding center.
            _log_message("No defects or cladding center not found, skipping histogram.", level="WARNING")
            return None # Return None if no defects or center.

        # Get the center of the fiber (e.g., cladding center).
        cladding_zone_info = image_res.detected_zones.get("cladding")
        # Ensure cladding_zone_info and its center_px are not None.
        if not cladding_zone_info or cladding_zone_info.center_px is None:
            _log_message("Cladding center is None, cannot generate polar histogram.", level="WARNING")
            return None

        fiber_center_x, fiber_center_y = cladding_zone_info.center_px

        # Prepare data for histogram.
        angles = [] # List to store angles of defects.
        radii = [] # List to store radial distances of defects.
        defect_colors_for_plot = [] # List to store colors for defects.

        # Iterate through defects.
        for defect in image_res.defects:
            # Calculate relative position of defect centroid to fiber center.
            dx = defect.centroid_px[0] - fiber_center_x
            dy = defect.centroid_px[1] - fiber_center_y
            # Calculate angle and radius.
            angles.append(np.arctan2(dy, dx)) # Angle in radians.
            radii.append(np.sqrt(dx**2 + dy**2)) # Radial distance in pixels.
            # Assign color based on defect type.
            defect_colors_for_plot.append(self.config.DEFECT_COLORS.get(defect.defect_type, (0,0,0))) # Default to black.

        # Create polar plot.
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8)) # Create figure and polar axes.
        # Scatter plot of defects.
        ax.scatter(angles, radii, c=[np.array(c)/255.0 for c in defect_colors_for_plot], s=50, alpha=0.75, edgecolors='k') # Convert BGR to RGB for plot.

        # Draw zone boundaries as circles on the polar plot.
        for zone_name, zone_info in image_res.detected_zones.items():
            zone_def = next((zd for zd in self.active_zone_definitions if zd.name == zone_name), None) # Get zone definition.
            if zone_def and zone_info.radius_px > 0: # If valid zone.
                # Draw circle representing the outer boundary of the zone.
                ax.plot(np.linspace(0, 2 * np.pi, 100), [zone_info.radius_px] * 100,
                        color=tuple(c/255 for c in reversed(zone_def.color_bgr)), linestyle='--', label=zone_name) # Convert BGR to RGB.
        # Set plot title.
        ax.set_title(f"Defect Distribution: {image_res.filename}", va='bottom')
        # Set radial limits (rmax).
        ax.set_rmax(max(radii) * 1.1 if radii else detected_cladding_radius_px * 2.5) # Adjust rmax.
        ax.legend() # Show legend.
        plt.tight_layout() # Adjust layout.
        _log_message(f"Defect histogram generated for {image_res.filename}.") # Log completion.
        return fig # Return the figure object.

    def _save_individual_image_report_csv(self, image_res: ImageResult, image_output_dir: Path):
        """Saves a detailed CSV report for a single image's defects."""
        # Log start of CSV report saving.
        _log_message(f"Saving individual CSV report for {image_res.filename}...")
        # Define CSV file path.
        report_path = image_output_dir / f"{Path(image_res.filename).stem}_defect_report.csv"
        # Store path in ImageResult.
        image_res.report_csv_path = report_path

        # Define fieldnames for the CSV header.
        fieldnames = [
            "Defect_ID", "Zone", "Type", "Centroid_X_px", "Centroid_Y_px",
            "Area_px2", "Area_um2", "Perimeter_px", "Perimeter_um",
            "Major_Dim_px", "Major_Dim_um", "Minor_Dim_px", "Minor_Dim_um",
            "Confidence", "Detection_Methods"
        ]
        # Open CSV file for writing.
        with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Create CSV writer.
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write header row.
            writer.writeheader()
            # Write data for each defect.
            for defect in image_res.defects:
                writer.writerow({
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
        # Log completion.
        _log_message(f"Individual CSV report saved to {report_path}")

    def _save_image_artifacts(self, original_bgr_image: np.ndarray, image_res: ImageResult):
        """Saves all generated artifacts (annotated image, CSV, histogram) for a single image."""
        # Log start of artifact saving.
        _log_message(f"Saving artifacts for {image_res.filename}...")
        # Create a subdirectory for this image's results within the main output directory.
        image_specific_output_dir = self.output_dir_path / Path(image_res.filename).stem
        # Create the directory if it doesn't exist.
        image_specific_output_dir.mkdir(parents=True, exist_ok=True)

        # Save annotated image if enabled in config.
        if self.config.SAVE_ANNOTATED_IMAGE:
            # Generate annotated image.
            annotated_img = self._generate_annotated_image(original_bgr_image, image_res)
            if annotated_img is not None: # If generation successful.
                # Define path for annotated image.
                annotated_path = image_specific_output_dir / f"{Path(image_res.filename).stem}_annotated.jpg"
                # Save the image.
                cv2.imwrite(str(annotated_path), annotated_img)
                # Store path in ImageResult.
                image_res.annotated_image_path = annotated_path
                # Log saving.
                _log_message(f"Annotated image saved to {annotated_path}")

        # Save detailed CSV report if enabled.
        if self.config.DETAILED_REPORT_PER_IMAGE and image_res.defects: # And if there are defects.
            # Save the CSV report.
            self._save_individual_image_report_csv(image_res, image_specific_output_dir)

        # Save defect histogram if enabled.
        if self.config.SAVE_HISTOGRAM:
            # Generate histogram figure.
            histogram_fig = self._generate_defect_histogram(image_res)
            if histogram_fig: # If figure generated.
                # Define path for histogram.
                histogram_path = image_specific_output_dir / f"{Path(image_res.filename).stem}_histogram.png"
                # Save the figure.
                histogram_fig.savefig(str(histogram_path), dpi=150)
                # Close the figure to free memory.
                plt.close(histogram_fig)
                # Store path in ImageResult.
                image_res.histogram_path = histogram_path
                # Log saving.
                _log_message(f"Defect histogram saved to {histogram_path}")

        # Save intermediate defect maps if enabled (for debugging).
        if self.config.SAVE_DEFECT_MAPS and image_res.intermediate_defect_maps:
            # Create subdirectory for defect maps.
            maps_dir = image_specific_output_dir / "defect_maps"
            maps_dir.mkdir(exist_ok=True) # Create if not exists.
            # Iterate and save each map.
            for map_name, defect_map_img in image_res.intermediate_defect_maps.items():
                if defect_map_img is not None: # If map exists.
                    # Define path for the map.
                    map_path = maps_dir / f"{map_name}.png"
                    # Save the map.
                    cv2.imwrite(str(map_path), defect_map_img)
            _log_message(f"Intermediate defect maps saved to {maps_dir}") # Log saving.
        _log_message(f"Artifacts saved for {image_res.filename}.") # Log completion.

    def _save_batch_summary_report_csv(self):
        """Saves a summary CSV report for the entire batch of processed images."""
        # Log start of batch summary saving.
        _log_message("Saving batch summary report...")
        # Check if there are any results to summarize.
        if not self.batch_results_summary_list:
            _log_message("No batch results to save in summary.", level="WARNING")
            return # Exit if no results.

        # Define path for the batch summary CSV.
        summary_path = self.output_dir_path / self.config.BATCH_SUMMARY_FILENAME
        try:
            # Create a Pandas DataFrame from the list of summary dictionaries.
            summary_df = pd.DataFrame(self.batch_results_summary_list)
            # Save DataFrame to CSV.
            summary_df.to_csv(summary_path, index=False, encoding='utf-8')
            # Log successful saving.
            _log_message(f"Batch summary report saved to {summary_path}")
        except Exception as e:
            # Log error if saving fails.
            _log_message(f"Error saving batch summary report: {e}", level="ERROR")


    # --- Main Orchestration Methods (Part 3) ---
    def process_single_image(self, image_path: Path) -> ImageResult:
        """Orchestrates the full analysis pipeline for a single image."""
        # Record start time for processing this image.
        single_image_start_time = _start_timer()
        # Log start of processing for the image.
        _log_message(f"--- Starting processing for image: {image_path.name} ---")

        # Initialize ImageResult object to store all data for this image.
        self.current_image_result = ImageResult(
            filename=image_path.name,
            timestamp=datetime.now(),
            fiber_specs_used=self.fiber_specs, # Store specs used.
            operating_mode=self.operating_mode # Store current operating mode.
        )

        # 1. Load Image
        original_bgr_image = self._load_single_image(image_path) # Load the image.
        if original_bgr_image is None: # If loading failed.
            self.current_image_result.error_message = f"Failed to load image." # Set error message.
            self.current_image_result.stats.status = "Error" # Set status to Error.
            _log_duration(f"Processing {image_path.name} (Load Error)", single_image_start_time, self.current_image_result)
            return self.current_image_result # Return result with error.

        # 2. Preprocess Image
        # Get various preprocessed versions of the image.
        processed_images = self._preprocess_image(original_bgr_image)
        if not processed_images: # If preprocessing failed.
            self.current_image_result.error_message = "Image preprocessing failed." # Set error message.
            self.current_image_result.stats.status = "Error" # Set status to Error.
            _log_duration(f"Processing {image_path.name} (Preproc Error)", single_image_start_time, self.current_image_result)
            return self.current_image_result # Return result with error.

        # 3. Find Fiber Center and Radius (assumed to be cladding)
        # Attempt to find the main circular feature.
        center_radius_tuple = self._find_fiber_center_and_radius(processed_images)
        if center_radius_tuple is None: # If no reliable circle found.
            self.current_image_result.error_message = "Could not reliably detect fiber center/cladding." # Set error.
            self.current_image_result.stats.status = "Error - No Fiber Found" # Set status.
            _log_duration(f"Processing {image_path.name} (No Fiber)", single_image_start_time, self.current_image_result)
            return self.current_image_result # Return result with error.
        fiber_center_px, detected_cladding_radius_px = center_radius_tuple # Unpack result.

        # 4. Calculate Pixels per Micron (if applicable)
        # This updates self.pixels_per_micron and self.current_image_result.stats.microns_per_pixel
        self._calculate_pixels_per_micron(detected_cladding_radius_px)
        # If MICRON_INFERRED and calculation failed, it might revert to PIXEL_ONLY effectively.
        # The operating_mode in current_image_result should reflect the mode actually used for measurements.
        # If pixels_per_micron is None after this, measurements will be in pixels.
        if self.operating_mode == "MICRON_INFERRED" and not self.pixels_per_micron:
            _log_message("MICRON_INFERRED mode failed to establish px/µm. Proceeding effectively as PIXEL_ONLY for measurements.", level="WARNING")
            self.current_image_result.operating_mode = "PIXEL_ONLY (Inference Failed)"


        # 5. Create Zone Masks
        # Generate masks for core, cladding, ferrule, etc.
        self.current_image_result.detected_zones = self._create_zone_masks(
            original_bgr_image.shape[:2], fiber_center_px, detected_cladding_radius_px
        )

        # 6. Defect Detection in each relevant zone
        all_defect_maps_by_method: Dict[str, Optional[np.ndarray]] = {} # Store all raw defect maps.
        # Iterate through detected zones to perform defect detection.
        for zone_name, zone_info in self.current_image_result.detected_zones.items():
            if zone_info.mask is None: continue # Skip if mask is not available.
            _log_message(f"Detecting defects in zone: {zone_name}") # Log zone processing.
            # Select appropriate grayscale image for detection (e.g., CLAHE enhanced or original gray).
            gray_for_detection = processed_images.get('clahe_enhanced', processed_images['original_gray'])

            # Run DO2MR
            do2mr_map = self._detect_region_defects_do2mr(gray_for_detection, zone_info.mask, zone_name)
            if do2mr_map is not None: all_defect_maps_by_method[f"do2mr_{zone_name}"] = do2mr_map
            # Run LEI
            lei_map = self._detect_scratches_lei(gray_for_detection, zone_info.mask, zone_name)
            if lei_map is not None: all_defect_maps_by_method[f"lei_{zone_name}"] = lei_map
            # Run Canny (example additional method)
            canny_map = self._detect_defects_canny(processed_images.get('gaussian_blurred', gray_for_detection), zone_info.mask, zone_name)
            if canny_map is not None: all_defect_maps_by_method[f"canny_{zone_name}"] = canny_map
            # Run Adaptive Threshold (example additional method)
            adaptive_map = self._detect_defects_adaptive_thresh(processed_images.get('bilateral_filtered', gray_for_detection), zone_info.mask, zone_name)
            if adaptive_map is not None: all_defect_maps_by_method[f"adaptive_thresh_{zone_name}"] = adaptive_map

        # Store all generated defect maps (primarily for debugging/SAVE_DEFECT_MAPS).
        self.current_image_result.intermediate_defect_maps = {k:v for k,v in all_defect_maps_by_method.items() if v is not None}


        # 7. Combine defect masks from all zones and methods
        # Create a single combined mask from all methods across all zones.
        # This might need refinement if inter-zone defects are handled specially.
        # For now, combine all detected defect pixels.
        global_combined_defect_mask = np.zeros(original_bgr_image.shape[:2], dtype=np.uint8)
        # Iterate through specific method maps (like do2mr_core, lei_cladding etc.)
        for method_map_key, specific_method_map in all_defect_maps_by_method.items():
            if specific_method_map is not None:
                 # This simple ORing might not be ideal. The _combine_defect_masks was designed for one zone.
                 # A better approach would be to combine per zone, then merge, or have _combine_defect_masks take a list of maps.
                 # For now, let's use the existing _combine_defect_masks with all maps.
                 pass # This combination logic needs to be thought out for multi-zone.

        # Let's simplify: combine all *types* of defect maps (DO2MR, LEI, etc.) globally for now.
        # This needs a more sophisticated combination strategy if defects are to be strictly confined to their original detection zone
        # after combination. The current _analyze_defect_contours re-assigns zones based on centroid.
        maps_for_global_combination: Dict[str, Optional[np.ndarray]] = {}
        # Collect all maps of a certain type (e.g., all do2mr maps from all zones)
        # This is a simplification. A more robust way is needed.
        # For now, let's assume _combine_defect_masks intelligently handles the keys or we pass it a list.
        # We will directly use the combined maps from each method type applied across all zones (implicitly).
        # The current _analyze_defect_contours will then take the globally combined mask.

        # Create a truly global combined mask from all individual method maps
        final_combined_mask_for_analysis = self._combine_defect_masks(all_defect_maps_by_method, original_bgr_image.shape[:2])


        # 8. Analyze Defect Contours
        # Extract detailed information about each defect.
        self.current_image_result.defects = self._analyze_defect_contours(final_combined_mask_for_analysis, image_path.name, all_defect_maps_by_method)

        # 9. Update Stats
        self.current_image_result.stats.total_defects = len(self.current_image_result.defects) # Total defects.
        # Count defects per zone.
        for defect in self.current_image_result.defects:
            if defect.zone_name == "core": self.current_image_result.stats.core_defects += 1
            elif defect.zone_name == "cladding": self.current_image_result.stats.cladding_defects += 1
            elif defect.zone_name == "ferrule_contact": self.current_image_result.stats.ferrule_defects += 1
            elif defect.zone_name == "adhesive": self.current_image_result.stats.adhesive_defects += 1
        # Placeholder for Pass/Fail status - to be implemented based on criteria.
        self.current_image_result.stats.status = "Review" # Default status.

        # 10. Save Artifacts (Annotated Image, CSV, Histogram)
        # Save all generated reports and visual aids.
        self._save_image_artifacts(original_bgr_image, self.current_image_result)

        # Finalize processing time for this image.
        self.current_image_result.stats.processing_time_s = _log_duration(f"Processing {image_path.name}", single_image_start_time)
        _log_message(f"--- Finished processing for image: {image_path.name} ---")
        # Return the comprehensive result for this image.
        return self.current_image_result

    def process_image_batch(self, image_paths: List[Path]):
        """Processes a batch of images."""
        # Log start of batch processing.
        batch_start_time = _start_timer()
        _log_message(f"Starting batch processing for {len(image_paths)} images...")
        # Clear previous batch summary.
        self.batch_results_summary_list = []

        # Iterate through each image path in the batch.
        for i, image_path in enumerate(image_paths):
            _log_message(f"Processing image {i+1}/{len(image_paths)}: {image_path.name}")
            # Process the single image.
            image_res = self.process_single_image(image_path)
            # Create a summary dictionary for this image.
            summary_item = {
                "Filename": image_res.filename,
                "Timestamp": image_res.timestamp.isoformat(),
                "Operating_Mode": image_res.operating_mode,
                "Status": image_res.stats.status,
                "Total_Defects": image_res.stats.total_defects,
                "Core_Defects": image_res.stats.core_defects,
                "Cladding_Defects": image_res.stats.cladding_defects,
                "Ferrule_Defects": image_res.stats.ferrule_defects,
                "Adhesive_Defects": image_res.stats.adhesive_defects,
                "Processing_Time_s": f"{image_res.stats.processing_time_s:.2f}",
                "Microns_Per_Pixel": f"{1.0/self.pixels_per_micron:.4f}" if self.pixels_per_micron else "N/A",
                "Error": image_res.error_message if image_res.error_message else ""
            }
            # Add summary to the batch list.
            self.batch_results_summary_list.append(summary_item)
            # Reset pixels_per_micron for the next image if it was inferred.
            if image_res.operating_mode == "MICRON_INFERRED" or \
               (image_res.operating_mode == "PIXEL_ONLY (Inference Failed)"): # If inference was attempted
                self.pixels_per_micron = None # Reset for next image.
                self.operating_mode = self.fiber_specs.cladding_diameter_um is not None and self.fiber_specs.cladding_diameter_um > 0 \
                                   and self.fiber_specs.core_diameter_um is not None \
                                   and self.fiber_specs.core_diameter_um > 0 \
                                   if "MICRON_CALCULATED" else "PIXEL_ONLY" # Re-evaluate based on initial specs for next image.
                self._initialize_zone_parameters() # Re-init zone defs if mode changed.


        # Save the batch summary report.
        self._save_batch_summary_report_csv()
        # Log completion of batch processing.
        _log_duration("Batch Processing", batch_start_time)
        _log_message(f"--- Batch processing complete. {len(image_paths)} images processed. ---")

# --- Main Execution Function ---
def main():
    """Main function to drive the fiber inspection script."""
    # Print welcome message.
    print("=" * 70)
    print(" Advanced Automated Optical Fiber End Face Inspector")
    print("=" * 70)
    # Record start time of the entire script.
    script_start_time = _start_timer()

    try:
        # Create an instance of the inspector configuration.
        config = InspectorConfig()
        # Create an instance of the FiberInspector.
        inspector = FiberInspector(config)

        # Get user input for fiber specifications.
        inspector._get_user_specifications()
        # Get paths to images to be processed.
        image_paths = inspector._get_image_paths_from_user()

        # Check if any images were found.
        if not image_paths:
            _log_message("No images to process. Exiting.", level="INFO")
            return # Exit if no images.

        # Process the batch of images.
        inspector.process_image_batch(image_paths)

    except FileNotFoundError as fnf_error: # Handle file not found errors.
        _log_message(f"Error: {fnf_error}", level="CRITICAL")
    except ValueError as val_error: # Handle value errors (e.g., invalid number input).
        _log_message(f"Input Error: {val_error}", level="CRITICAL")
    except Exception as e: # Handle any other unexpected errors.
        _log_message(f"An unexpected error occurred: {e}", level="CRITICAL")
        import traceback # Import traceback for detailed error info.
        traceback.print_exc() # Print full traceback.
    finally:
        # Log total execution time of the script.
        _log_duration("Total Script Execution", script_start_time)
        print("=" * 70) # Print end message.
        print("Inspection Run Finished.")
        print("=" * 70)

# --- Script Entry Point ---
if __name__ == "__main__":
    # Call the main function when the script is executed.
    main()
