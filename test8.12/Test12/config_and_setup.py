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
import pandas as pd  # Pandas for easy CSV writing of batch summary
from dataclasses import dataclass, field  # Dataclasses for structured data
# Note: concurrent.futures will be used in later parts.

# Suppress warnings that might clutter the output
warnings.filterwarnings('ignore')  # Ignores runtime warnings, e.g., from division by zero in calculations.

# --- Data Structures ---

@dataclass
class FiberSpecifications:
    """Data structure to hold user-provided or default fiber optic specifications."""
    core_diameter_um: Optional[float] = None  # Diameter of the fiber core in micrometers.
    cladding_diameter_um: Optional[float] = 125.0  # Diameter of the fiber cladding in micrometers (default for many fibers).
    ferrule_diameter_um: Optional[float] = 250.0  # Outer diameter of the ferrule in micrometers (approximate).
    fiber_type: str = "unknown"  # Type of fiber, e.g., "single-mode", "multi-mode".

@dataclass
class ZoneDefinition:
    """Data structure to define parameters for a fiber zone."""
    name: str  # Name of the zone (e.g., "core", "cladding").
    # Relative factors to the primary detected radius (e.g., cladding radius) if in pixel_only mode,
    # or absolute radii in microns if specs are provided.
    r_min_factor_or_um: float  # Minimum radius factor (relative to main radius) or absolute radius in um.
    r_max_factor_or_um: float  # Maximum radius factor (relative to main radius) or absolute radius in um.
    color_bgr: Tuple[int, int, int]  # BGR color for visualizing this zone.
    max_defect_size_um: Optional[float] = None  # Maximum allowable defect size in this zone in micrometers (for pass/fail).
    defects_allowed: bool = True  # Whether defects are generally allowed in this zone.

@dataclass
class DetectedZoneInfo:
    """Data structure to hold information about a detected zone in an image."""
    name: str  # Name of the zone.
    center_px: Tuple[int, int]  # Center coordinates (x, y) in pixels.
    radius_px: float  # Radius in pixels.
    radius_um: Optional[float] = None  # Radius in micrometers (if conversion is available).
    mask: Optional[np.ndarray] = None  # Binary mask for the zone.

@dataclass
class DefectMeasurement:
    """Data structure for defect measurements."""
    value_px: Optional[float] = None  # Measurement in pixels.
    value_um: Optional[float] = None  # Measurement in micrometers.

@dataclass
class DefectInfo:
    """Data structure to hold detailed information about a detected defect."""
    defect_id: int  # Unique identifier for the defect within an image.
    zone_name: str  # Name of the zone where the defect is primarily located.
    defect_type: str  # Type of defect (e.g., "Region", "Scratch").
    centroid_px: Tuple[int, int]  # Centroid coordinates (x, y) in pixels.
    bounding_box_px: Tuple[int, int, int, int]  # Bounding box (x, y, width, height) in pixels.
    area: DefectMeasurement = field(default_factory=DefectMeasurement)  # Area of the defect.
    perimeter: DefectMeasurement = field(default_factory=DefectMeasurement)  # Perimeter of the defect.
    major_dimension: DefectMeasurement = field(default_factory=DefectMeasurement)  # Primary dimension (e.g. length of scratch, diameter of pit)
    minor_dimension: DefectMeasurement = field(default_factory=DefectMeasurement)  # Secondary dimension (e.g. width of scratch)
    confidence_score: float = 0.0  # Confidence score for the detection (0.0 to 1.0).
    detection_methods: List[str] = field(default_factory=list)  # List of methods that identified this defect.
    contour: Optional[np.ndarray] = None  # The contour of the defect in pixels.

@dataclass
class ImageAnalysisStats:
    """Statistics for a single image analysis."""
    total_defects: int = 0  # Total number of defects found.
    core_defects: int = 0  # Number of defects in the core.
    cladding_defects: int = 0  # Number of defects in the cladding.
    ferrule_defects: int = 0  # Number of defects in the ferrule.
    adhesive_defects: int = 0  # Number of defects in the adhesive area.
    processing_time_s: float = 0.0  # Time taken to process the image in seconds.
    status: str = "Pending"  # Pass/Fail/Review status.
    microns_per_pixel: Optional[float] = None  # Calculated conversion ratio for this image.

@dataclass
class ImageResult:
    """Data structure to store all results for a single processed image."""
    filename: str  # Original filename of the image.
    timestamp: datetime  # Timestamp of when the analysis was performed.
    fiber_specs_used: FiberSpecifications  # Fiber specifications used for this image.
    operating_mode: str  # "PIXEL_ONLY" or "MICRON_CALCULATED" or "MICRON_INFERRED".
    detected_zones: Dict[str, DetectedZoneInfo] = field(default_factory=dict)  # Information about detected zones.
    defects: List[DefectInfo] = field(default_factory=list)  # List of detected defects.
    stats: ImageAnalysisStats = field(default_factory=ImageAnalysisStats)  # Summary statistics for the image.
    annotated_image_path: Optional[Path] = None  # Path to the saved annotated image.
    report_csv_path: Optional[Path] = None  # Path to the saved CSV report for this image.
    histogram_path: Optional[Path] = None  # Path to the saved defect distribution histogram.
    error_message: Optional[str] = None  # Error message if processing failed.
    intermediate_defect_maps: Dict[str, np.ndarray] = field(default_factory=dict)  # For debugging.
    timing_log: Dict[str, float] = field(default_factory=dict)  # Store per-step durations

# --- Configuration Class ---

@dataclass
class InspectorConfig:
    """Class to hold all configuration parameters for the fiber inspection process."""
    # General Settings
    OUTPUT_DIR_NAME: str = "fiber_inspection_output"  # Name of the directory to save results.
    MIN_DEFECT_AREA_PX: int = 10  # Minimum area in pixels for a contour to be considered a defect.
    PERFORM_CALIBRATION: bool = False  # Whether to attempt system calibration with a target.
    CALIBRATION_IMAGE_PATH: Optional[str] = None  # Path to the calibration target image.
    CALIBRATION_DOT_SPACING_UM: float = 10.0  # Known spacing of dots on calibration target in microns.
    CALIBRATION_FILE_JSON: str = "calibration_data.json"  # File to save/load calibration data.

    # Fiber Zone Definitions (Default for PIXEL_ONLY mode, scaled if cladding detected)
    DEFAULT_ZONES: List[ZoneDefinition] = field(default_factory=lambda: [
        ZoneDefinition(
            name="core",
            r_min_factor_or_um=0.0,
            r_max_factor_or_um=0.4,
            color_bgr=(255, 0, 0),
            max_defect_size_um=5.0,
            defects_allowed=True
        ),
        ZoneDefinition(
            name="cladding",
            r_min_factor_or_um=0.4,
            r_max_factor_or_um=1.0,
            color_bgr=(0, 255, 0),
            max_defect_size_um=10.0,
            defects_allowed=True
        ),
        ZoneDefinition(
            name="ferrule_contact",
            r_min_factor_or_um=1.0,
            r_max_factor_or_um=2.0,
            color_bgr=(0, 0, 255),
            max_defect_size_um=25.0,
            defects_allowed=True
        ),
        ZoneDefinition(
            name="adhesive",
            r_min_factor_or_um=2.0,
            r_max_factor_or_um=2.2,
            color_bgr=(0, 255, 255),
            max_defect_size_um=50.0,
            defects_allowed=False
        )
    ])

    # Image Preprocessing
    GAUSSIAN_BLUR_KERNEL_SIZE: Tuple[int, int] = (7, 7)  # Kernel size for Gaussian blur.
    GAUSSIAN_BLUR_SIGMA: int = 2  # Sigma for Gaussian blur.
    BILATERAL_FILTER_D: int = 9  # Diameter of each pixel neighborhood for bilateral filter.
    BILATERAL_FILTER_SIGMA_COLOR: int = 75  # Filter sigma in the color space.
    BILATERAL_FILTER_SIGMA_SPACE: int = 75  # Filter sigma in the coordinate space.
    CLAHE_CLIP_LIMIT: float = 2.0  # Clip limit for CLAHE.
    CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)  # Tile grid size for CLAHE.

    # Hough Circle Transform Parameters (multiple sets for robustness)
    HOUGH_DP_VALUES: List[float] = field(default_factory=lambda: [1.0, 1.2, 1.5])  # Inverse ratio of accumulator resolution.
    HOUGH_MIN_DIST_FACTOR: float = 0.25  # Minimum distance between centers of detected circles, as a factor of image smaller dimension.
    HOUGH_PARAM1_VALUES: List[int] = field(default_factory=lambda: [70, 100, 130])  # Upper threshold for Canny edge detector in Hough.
    HOUGH_PARAM2_VALUES: List[int] = field(default_factory=lambda: [35, 45, 55])  # Accumulator threshold for circle detection.
    HOUGH_MIN_RADIUS_FACTOR: float = 0.1  # Minimum circle radius as a factor of image smaller dimension.
    HOUGH_MAX_RADIUS_FACTOR: float = 0.45  # Maximum circle radius as a factor of image smaller dimension.
    CIRCLE_CONFIDENCE_THRESHOLD: float = 0.3  # Minimum confidence for a detected circle to be considered valid.

    # DO2MR (Region-Based Defect) Parameters
    DO2MR_KERNEL_SIZES: List[Tuple[int, int]] = field(default_factory=lambda: [(5, 5), (9, 9), (13, 13)])  # Structuring element sizes.
    DO2MR_GAMMA_VALUES: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0])  # Sensitivity parameter for thresholding residual.
    DO2MR_MEDIAN_BLUR_KERNEL_SIZE: int = 5  # Kernel size for median blur on residual image.
    DO2MR_MORPH_OPEN_KERNEL_SIZE: Tuple[int, int] = (3, 3)  # Kernel for morphological opening post-threshold.

    # LEI (Scratch Detection) Parameters
    LEI_KERNEL_LENGTHS: List[int] = field(default_factory=lambda: [11, 17, 23])  # Lengths of the linear detector.
    LEI_ANGLE_STEP: int = 15  # Angular resolution for scratch detection (degrees).
    LEI_THRESHOLD_FACTOR: float = 2.0  # Factor for Otsu or adaptive thresholding on response map.
    LEI_MORPH_CLOSE_KERNEL_SIZE: Tuple[int, int] = (5, 1)  # Kernel for morphological closing (elongated for scratches).
    LEI_MIN_SCRATCH_AREA_PX: int = 15  # Minimum area for a scratch.

    # Additional Defect Detection Parameters
    CANNY_LOW_THRESHOLD: int = 50  # Low threshold for Canny edge detection.
    CANNY_HIGH_THRESHOLD: int = 150  # High threshold for Canny edge detection.
    ADAPTIVE_THRESH_BLOCK_SIZE: int = 11  # Block size for adaptive thresholding.
    ADAPTIVE_THRESH_C: int = 2  # Constant subtracted from the mean in adaptive thresholding.

    # Ensemble/Confidence Parameters
    MIN_METHODS_FOR_CONFIRMED_DEFECT: int = 2  # Min number of methods that must detect a defect.
    CONFIDENCE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {  # Weights for different detection methods.
        "do2mr": 1.0,
        "lei": 1.0,
        "canny": 0.6,
        "adaptive_thresh": 0.7,
        "otsu_global": 0.5,
    })

    # Visualization Parameters
    DEFECT_COLORS: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {  # BGR colors for defect types.
        "Region": (0, 255, 255),  # Yellow
        "Scratch": (255, 0, 255),  # Magenta
        "Contamination": (255, 165, 0),  # Orange
        "Pit": (0, 128, 255),  # Light Orange
        "Chip": (128, 0, 128),  # Purple
        "Linear Region": (255, 105, 180)  # Hot Pink for Linear Region
    })
    FONT_SCALE: float = 0.5  # Font scale for annotations.
    LINE_THICKNESS: int = 1  # Line thickness for drawing.

    # Reporting
    BATCH_SUMMARY_FILENAME: str = "batch_inspection_summary.csv"  # Filename for the overall batch summary.
    DETAILED_REPORT_PER_IMAGE: bool = True  # Whether to generate a detailed CSV for each image.
    SAVE_ANNOTATED_IMAGE: bool = True  # Whether to save the annotated image.
    SAVE_DEFECT_MAPS: bool = False  # Whether to save intermediate defect maps (for debugging).
    SAVE_HISTOGRAM: bool = True  # Whether to save the polar defect distribution histogram.

# --- Utility Functions ---

def _log_message(message: str, level: str = "INFO"):
    """Prints a timestamped log message to the console."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{current_time}] [{level.upper()}] {message}")

def _start_timer() -> float:
    """Returns the current time to start a timer."""
    return time.perf_counter()

def _log_duration(operation_name: str, start_time: float, image_result: Optional[ImageResult] = None):
    """Logs the duration of an operation and stores it."""
    duration = time.perf_counter() - start_time
    _log_message(f"Operation '{operation_name}' completed in {duration:.4f} seconds.")
    if image_result and hasattr(image_result, "timing_log"):
        image_result.timing_log[operation_name] = duration
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
        _log_message("FiberInspector initialized.", level="DEBUG")
        # Initialize zone definitions based on the config and potentially user input later.
        self.active_zone_definitions: List[ZoneDefinition] = []
        self._initialize_zone_parameters()

    def _get_user_specifications(self):
        """
        Prompts the user for fiber specifications (core/cladding diameters, fiber type)
        and updates the internal state (self.fiber_specs, self.operating_mode).
        """
        start_time = _start_timer()
        _log_message("Starting user specification input...")
        print("\n--- Fiber Optic Specifications ---")
        provide_specs_input = input("Provide known fiber specifications (microns)? (y/n, default: n): ").strip().lower()

        if provide_specs_input == 'y':
            _log_message("User chose to provide fiber specifications.")
            try:
                # Prompt for core diameter
                core_dia_str = input(f"Enter CORE diameter in microns (e.g., 9, 50, 62.5) (optional, press Enter to skip): ").strip()
                if core_dia_str:
                    self.fiber_specs.core_diameter_um = float(core_dia_str)
                # Prompt for cladding diameter
                clad_dia_str = input(f"Enter CLADDING diameter in microns (e.g., 125) (default: {self.fiber_specs.cladding_diameter_um}): ").strip()
                if clad_dia_str:
                    self.fiber_specs.cladding_diameter_um = float(clad_dia_str)
                # Prompt for ferrule diameter
                ferrule_dia_str = input(f"Enter FERRULE outer diameter in microns (e.g., 250) (default: {self.fiber_specs.ferrule_diameter_um}): ").strip()
                if ferrule_dia_str:
                    self.fiber_specs.ferrule_diameter_um = float(ferrule_dia_str)
                # Prompt for fiber type
                self.fiber_specs.fiber_type = input("Enter fiber type (e.g., single-mode, multi-mode) (optional): ").strip()

                if self.fiber_specs.cladding_diameter_um is not None:
                    self.operating_mode = "MICRON_CALCULATED"
                    _log_message(
                        f"Operating mode set to MICRON_CALCULATED. "
                        f"Specs: Core={self.fiber_specs.core_diameter_um}, "
                        f"Clad={self.fiber_specs.cladding_diameter_um}, "
                        f"Ferrule={self.fiber_specs.ferrule_diameter_um}, "
                        f"Type='{self.fiber_specs.fiber_type}'."
                    )
                else:
                    self.operating_mode = "PIXEL_ONLY"
                    _log_message("Cladding diameter not provided, falling back to PIXEL_ONLY mode.", level="WARNING")
            except ValueError:
                _log_message("Invalid input for diameter. Proceeding in PIXEL_ONLY mode.", level="ERROR")
                self.operating_mode = "PIXEL_ONLY"
                self.fiber_specs = FiberSpecifications()
        else:
            self.operating_mode = "PIXEL_ONLY"
            _log_message("User chose to skip fiber specifications. Operating mode set to PIXEL_ONLY.")

        _log_duration("User Specification Input", start_time)
        self._initialize_zone_parameters()

    def _initialize_zone_parameters(self):
        """
        Initializes self.active_zone_definitions based on operating_mode and fiber_specs.
        """
        _log_message("Initializing zone parameters...")
        self.active_zone_definitions = []

        if self.operating_mode == "MICRON_CALCULATED" and self.fiber_specs.cladding_diameter_um:
            # Convert diameters to radii in microns
            core_r_um = (self.fiber_specs.core_diameter_um / 2.0) if self.fiber_specs.core_diameter_um else 0.0
            cladding_r_um = self.fiber_specs.cladding_diameter_um / 2.0
            ferrule_r_um = (self.fiber_specs.ferrule_diameter_um / 2.0) if self.fiber_specs.ferrule_diameter_um else cladding_r_um * 2.0
            adhesive_r_um = ferrule_r_um * 1.1  # 10% larger than ferrule

            default_core = next((z for z in self.config.DEFAULT_ZONES if z.name == "core"), None)
            default_cladding = next((z for z in self.config.DEFAULT_ZONES if z.name == "cladding"), None)
            default_ferrule = next((z for z in self.config.DEFAULT_ZONES if z.name == "ferrule_contact"), None)
            default_adhesive = next((z for z in self.config.DEFAULT_ZONES if z.name == "adhesive"), None)

            self.active_zone_definitions = [
                ZoneDefinition(
                    name="core",
                    r_min_factor_or_um=0.0,
                    r_max_factor_or_um=core_r_um,
                    color_bgr=default_core.color_bgr if default_core else (255, 0, 0),
                    max_defect_size_um=default_core.max_defect_size_um if default_core else 5.0
                ),
                ZoneDefinition(
                    name="cladding",
                    r_min_factor_or_um=core_r_um,
                    r_max_factor_or_um=cladding_r_um,
                    color_bgr=default_cladding.color_bgr if default_cladding else (0, 255, 0),
                    max_defect_size_um=default_cladding.max_defect_size_um if default_cladding else 10.0
                ),
                ZoneDefinition(
                    name="ferrule_contact",
                    r_min_factor_or_um=cladding_r_um,
                    r_max_factor_or_um=ferrule_r_um,
                    color_bgr=default_ferrule.color_bgr if default_ferrule else (0, 0, 255),
                    max_defect_size_um=default_ferrule.max_defect_size_um if default_ferrule else 25.0
                ),
                ZoneDefinition(
                    name="adhesive",
                    r_min_factor_or_um=ferrule_r_um,
                    r_max_factor_or_um=adhesive_r_um,
                    color_bgr=default_adhesive.color_bgr if default_adhesive else (0, 255, 255),
                    max_defect_size_um=default_adhesive.max_defect_size_um if default_adhesive else 50.0,
                    defects_allowed=default_adhesive.defects_allowed if default_adhesive else False
                )
            ]
            _log_message(f"Zone parameters set for MICRON_CALCULATED: Core R={core_r_um}µm, Clad R={cladding_r_um}µm.")
        else:
            # Treat MICRON_INFERRED as same as PIXEL_ONLY for now
            self.active_zone_definitions = self.config.DEFAULT_ZONES
            _log_message(f"Zone parameters set to default relative factors for {self.operating_mode} mode.")

    def _get_image_paths_from_user(self) -> List[Path]:
        """
        Prompts the user to enter a directory path, then returns a list of all image files
        (.jpg, .png, etc.) found in that directory.
        """
        start_time = _start_timer()
        _log_message("Starting image path collection...")
        image_paths: List[Path] = []

        while True:
            dir_path_str = input("Enter the path to the directory containing fiber images: ").strip()
            image_dir = Path(dir_path_str)
            if not image_dir.is_dir():
                _log_message(f"Error: The path '{image_dir}' is not a valid directory. Please try again.", level="ERROR")
                continue

            supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            for item in image_dir.iterdir():
                if item.is_file() and item.suffix.lower() in supported_extensions:
                    image_paths.append(item)

            if not image_paths:
                _log_message(f"No images found in directory: {image_dir}. Please check the path or directory content.", level="WARNING")
            else:
                _log_message(f"Found {len(image_paths)} images in '{image_dir}'.")
                break

        _log_duration("Image Path Collection", start_time)
        return image_paths

    def _load_single_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Loads a single image from the given path and handles grayscale/alpha channels gracefully.
        """
        _log_message(f"Loading image: {image_path.name}")
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                _log_message(f"Failed to load image: {image_path}", level="ERROR")
                return None

            # Convert BGRA to BGR if alpha channel present
            if len(image.shape) == 3 and image.shape[2] == 4:
                _log_message(f"Image '{image_path.name}' has an alpha channel. Converting to BGR.")
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            # Grayscale is fine as-is

            _log_message(f"Successfully loaded image: {image_path.name} with shape {image.shape}")
            return image
        except Exception as e:
            _log_message(f"An error occurred while loading image {image_path}: {e}", level="ERROR")
            return None

    # --- Preprocessing Methods ---

    def _preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Applies various preprocessing techniques to the input image.
        Returns a dictionary with keys:
            'original_gray', 'gaussian_blurred', 'bilateral_filtered', 'clahe_enhanced', 'hist_equalized'
        """
        preprocess_start_time = _start_timer()
        _log_message("Starting image preprocessing...")

        if image is None:
            _log_message("Input image for preprocessing is None.", level="ERROR")
            return {}

        # Convert to grayscale if color, else copy
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray = image.copy()
        else:
            _log_message(f"Unsupported image format for preprocessing: shape {image.shape}", level="ERROR")
            return {}

        processed_images: Dict[str, np.ndarray] = {}
        processed_images['original_gray'] = gray.copy()

        # 1. Gaussian Blur
        try:
            processed_images['gaussian_blurred'] = cv2.GaussianBlur(
                gray,
                self.config.GAUSSIAN_BLUR_KERNEL_SIZE,
                self.config.GAUSSIAN_BLUR_SIGMA
            )
        except Exception as e:
            _log_message(f"Error during Gaussian Blur: {e}", level="WARNING")
            processed_images['gaussian_blurred'] = gray.copy()

        # 2. Bilateral Filter
        try:
            processed_images['bilateral_filtered'] = cv2.bilateralFilter(
                gray,
                self.config.BILATERAL_FILTER_D,
                self.config.BILATERAL_FILTER_SIGMA_COLOR,
                self.config.BILATERAL_FILTER_SIGMA_SPACE
            )
        except Exception as e:
            _log_message(f"Error during Bilateral Filter: {e}", level="WARNING")
            processed_images['bilateral_filtered'] = gray.copy()

        # 3. CLAHE
        try:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.CLAHE_CLIP_LIMIT,
                tileGridSize=self.config.CLAHE_TILE_GRID_SIZE
            )
            processed_images['clahe_enhanced'] = clahe.apply(processed_images.get('bilateral_filtered', gray))
        except Exception as e:
            _log_message(f"Error during CLAHE: {e}", level="WARNING")
            processed_images['clahe_enhanced'] = gray.copy()

        # 4. Histogram Equalization
        try:
            processed_images['hist_equalized'] = cv2.equalizeHist(gray)
        except Exception as e:
            _log_message(f"Error during Histogram Equalization: {e}", level="WARNING")
            processed_images['hist_equalized'] = gray.copy()

        _log_duration("Image Preprocessing", preprocess_start_time, self.current_image_result)
        return processed_images

    # --- Fiber Center and Zone Detection Methods ---

    def _find_fiber_center_and_radius(self, processed_images: Dict[str, np.ndarray]) -> Optional[Tuple[Tuple[int, int], float]]:
        """
        Robustly finds the primary circular feature (assumed cladding) center and radius
        by performing multiple Hough Circle Transform attempts on preprocessed images.
        """
        detection_start_time = _start_timer()
        _log_message("Starting fiber center and radius detection...")

        all_detected_circles: List[Tuple[int, int, int, float, str]] = []
        h, w = processed_images['original_gray'].shape[:2]
        min_dist_circles = int(min(h, w) * self.config.HOUGH_MIN_DIST_FACTOR)
        min_radius_hough = int(min(h, w) * self.config.HOUGH_MIN_RADIUS_FACTOR)
        max_radius_hough = int(min(h, w) * self.config.HOUGH_MAX_RADIUS_FACTOR)

        for image_key in ['gaussian_blurred', 'bilateral_filtered', 'clahe_enhanced']:
            img_to_process = processed_images.get(image_key)
            if img_to_process is None:
                continue

            for dp in self.config.HOUGH_DP_VALUES:
                for param1 in self.config.HOUGH_PARAM1_VALUES:
                    for param2 in self.config.HOUGH_PARAM2_VALUES:
                        try:
                            circles = cv2.HoughCircles(
                                img_to_process,
                                cv2.HOUGH_GRADIENT,
                                dp=dp,
                                minDist=min_dist_circles,
                                param1=param1,
                                param2=param2,
                                minRadius=min_radius_hough,
                                maxRadius=max_radius_hough
                            )
                            if circles is not None:
                                circles = np.uint16(np.around(circles))
                                for i in circles[0, :]:
                                    cx, cy, r = int(i[0]), int(i[1]), int(i[2])
                                    dist_to_img_center = np.sqrt((cx - w // 2) ** 2 + (cy - h // 2) ** 2)
                                    normalized_r = r / max_radius_hough if max_radius_hough > 0 else 0
                                    confidence = (
                                        (param2 / max(self.config.HOUGH_PARAM2_VALUES)) * 0.5
                                        + normalized_r * 0.5
                                        - (dist_to_img_center / (min(h, w) / 2)) * 0.2
                                    )
                                    confidence = max(0, min(1, confidence))
                                    all_detected_circles.append((cx, cy, r, confidence, image_key))
                        except Exception as e:
                            _log_message(
                                f"Error in HoughCircles on {image_key} with params ({dp},{param1},{param2}): {e}",
                                level="WARNING"
                            )

        if not all_detected_circles:
            _log_message("No circles detected by Hough Transform.", level="WARNING")
            _log_duration("Fiber Center Detection (No Circles)", detection_start_time, self.current_image_result)
            return None

        # Pick the circle with highest confidence
        all_detected_circles.sort(key=lambda x: x[3], reverse=True)
        best_cx, best_cy, best_r, best_conf, source = all_detected_circles[0]
        if best_conf < self.config.CIRCLE_CONFIDENCE_THRESHOLD:
            _log_message(
                f"Best detected circle confidence ({best_conf:.2f} from {source}) is below threshold "
                f"({self.config.CIRCLE_CONFIDENCE_THRESHOLD}).",
                level="WARNING"
            )
            _log_duration("Fiber Center Detection (Low Confidence)", detection_start_time, self.current_image_result)
            return None

        _log_message(
            f"Best fiber center detected at ({best_cx}, {best_cy}) with radius {best_r}px. "
            f"Confidence: {best_conf:.2f} (from {source})."
        )
        _log_duration("Fiber Center Detection", detection_start_time, self.current_image_result)
        return (best_cx, best_cy), float(best_r)

    def _calculate_pixels_per_micron(self, detected_cladding_radius_px: float) -> Optional[float]:
        """
        Calculates the pixels_per_micron ratio based on the known cladding diameter (um)
        and the detected cladding radius in pixels.
        """
        calc_start_time = _start_timer()
        _log_message("Calculating pixels per micron...")

        ppm: Optional[float] = None
        if self.operating_mode in ["MICRON_CALCULATED", "MICRON_INFERRED"]:
            if self.fiber_specs.cladding_diameter_um and detected_cladding_radius_px > 0:
                ppm = (2 * detected_cladding_radius_px) / self.fiber_specs.cladding_diameter_um
                self.pixels_per_micron = ppm
                if self.current_image_result:
                    self.current_image_result.stats.microns_per_pixel = (1.0 / ppm) if ppm > 0 else None
                _log_message(f"Calculated pixels_per_micron: {ppm:.4f} px/µm (µm/px: {1/ppm:.4f}).")
            else:
                _log_message("Cladding diameter or detected radius invalid, cannot calculate px/µm.", level="WARNING")
        else:
            _log_message("Not in MICRON_CALCULATED or MICRON_INFERRED mode, skipping px/µm calculation.", level="DEBUG")

        _log_duration("Pixels per Micron Calculation", calc_start_time, self.current_image_result)
        return ppm

    def _create_zone_masks(
        self,
        image_shape: Tuple[int, int],
        fiber_center_px: Tuple[int, int],
        detected_cladding_radius_px: float
    ) -> Dict[str, DetectedZoneInfo]:
        """
        Creates binary masks for each defined fiber zone (core, cladding, ferrule, etc.).
        Returns a dict keyed by zone name, each with a DetectedZoneInfo.
        """
        mask_start_time = _start_timer()
        _log_message("Creating zone masks...")

        detected_zones_info: Dict[str, DetectedZoneInfo] = {}
        h, w = image_shape[:2]
        cx, cy = fiber_center_px

        # Radial distance squared from (cx, cy)
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2

        for zone_def in self.active_zone_definitions:
            # Compute radii in pixels (r_min_px, r_max_px)
            if self.operating_mode in ["PIXEL_ONLY", "MICRON_INFERRED"]:
                r_min_px = zone_def.r_min_factor_or_um * detected_cladding_radius_px
                r_max_px = zone_def.r_max_factor_or_um * detected_cladding_radius_px
                if self.operating_mode == "MICRON_INFERRED" and self.pixels_per_micron:
                    r_min_um = r_min_px / self.pixels_per_micron
                    r_max_um = r_max_px / self.pixels_per_micron
                else:
                    r_min_um = None
                    r_max_um = None
            else:  # MICRON_CALCULATED
                if self.pixels_per_micron and self.pixels_per_micron > 0:
                    r_min_px = zone_def.r_min_factor_or_um * self.pixels_per_micron
                    r_max_px = zone_def.r_max_factor_or_um * self.pixels_per_micron
                    r_min_um = zone_def.r_min_factor_or_um
                    r_max_um = zone_def.r_max_factor_or_um
                else:
                    _log_message(
                        f"Pixels_per_micron not available in MICRON_CALCULATED mode for zone '{zone_def.name}'. "
                        "Mask creation might be inaccurate.",
                        level="WARNING"
                    )
                    r_min_px = zone_def.r_min_factor_or_um
                    r_max_px = zone_def.r_max_factor_or_um
                    r_min_um = None
                    r_max_um = None

            # **FIXED**: Use dist_from_center_sq (not dist_sq_map)
            zone_mask_np = (
                (dist_from_center_sq >= (r_min_px ** 2))
                & (dist_from_center_sq < (r_max_px ** 2))
            ).astype(np.uint8) * 255

            detected_zones_info[zone_def.name] = DetectedZoneInfo(
                name=zone_def.name,
                center_px=fiber_center_px,
                radius_px=r_max_px,
                radius_um=r_max_um,
                mask=zone_mask_np
            )
            _log_message(
                f"Created mask for zone '{zone_def.name}': r_min={r_min_px:.2f}px, r_max={r_max_px:.2f}px."
            )

        _log_duration("Zone Mask Creation", mask_start_time, self.current_image_result)
        return detected_zones_info

    # --- Defect Detection Algorithm Implementations ---

    def _detect_region_defects_do2mr(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """
        Detects region-based defects (dirt, pits, contamination) using a DO2MR-inspired method.
        Returns a binary mask of detected region defects, or None if an error occurs.
        """
        do2mr_start_time = _start_timer()
        _log_message(f"Starting DO2MR region defect detection for zone '{zone_name}'...")

        if image_gray is None or zone_mask is None:
            _log_message("Input image or mask is None for DO2MR.", level="ERROR")
            return None

        masked_image = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask)
        vote_map = np.zeros_like(image_gray, dtype=np.float32)

        for kernel_size in self.config.DO2MR_KERNEL_SIZES:
            struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            min_filtered = cv2.erode(masked_image, struct_element)
            max_filtered = cv2.dilate(masked_image, struct_element)
            residual = cv2.subtract(max_filtered, min_filtered)

            if self.config.DO2MR_MEDIAN_BLUR_KERNEL_SIZE > 0:
                res_blurred = cv2.medianBlur(residual, self.config.DO2MR_MEDIAN_BLUR_KERNEL_SIZE)
            else:
                res_blurred = residual

            for gamma in self.config.DO2MR_GAMMA_VALUES:
                masked_res_vals = res_blurred[zone_mask > 0]
                if masked_res_vals.size == 0:
                    _log_message(
                        f"Zone mask for '{zone_name}' is empty or residual is all zero. "
                        f"Skipping DO2MR for gamma={gamma}, kernel={kernel_size}.",
                        level="WARNING"
                    )
                    continue

                mean_val = np.mean(masked_res_vals)
                std_val = np.std(masked_res_vals)
                thresh_val = np.clip(mean_val + gamma * std_val, 0, 255)

                _, defect_mask_pass = cv2.threshold(res_blurred, thresh_val, 255, cv2.THRESH_BINARY)
                defect_mask_pass = cv2.bitwise_and(defect_mask_pass, defect_mask_pass, mask=zone_mask)

                if self.config.DO2MR_MORPH_OPEN_KERNEL_SIZE[0] > 0 and self.config.DO2MR_MORPH_OPEN_KERNEL_SIZE[1] > 0:
                    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.config.DO2MR_MORPH_OPEN_KERNEL_SIZE)
                    defect_mask_pass = cv2.morphologyEx(defect_mask_pass, cv2.MORPH_OPEN, open_kernel)

                vote_map += (defect_mask_pass / 255.0)

        num_param_sets = len(self.config.DO2MR_KERNEL_SIZES) * len(self.config.DO2MR_GAMMA_VALUES)
        min_votes = max(1, int(num_param_sets * 0.3))  # Require ~30% agreement
        combined_map = np.where(vote_map >= min_votes, 255, 0).astype(np.uint8)

        _log_duration(f"DO2MR Detection for {zone_name}", do2mr_start_time, self.current_image_result)
        return combined_map

    def _detect_scratches_lei(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """
        Detects linear scratches using an LEI-inspired method.
        Returns a binary mask of detected scratches (0/255), or None on error.
        """
        lei_start_time = _start_timer()
        _log_message(f"Starting LEI scratch detection for zone '{zone_name}'...")

        if image_gray is None or zone_mask is None:
            _log_message("Input image or mask is None for LEI.", level="ERROR")
            return None

        masked_image = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask)
        enhanced_image = cv2.equalizeHist(masked_image)
        enhanced_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=zone_mask)
        max_response_map = np.zeros_like(enhanced_image, dtype=np.float32)

        for kernel_length in self.config.LEI_KERNEL_LENGTHS:
            for angle_deg in range(0, 180, self.config.LEI_ANGLE_STEP):
                line_kernel_base = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
                rot_matrix = cv2.getRotationMatrix2D((kernel_length // 2, 0), angle_deg, 1.0)
                bbox_size = int(np.ceil(kernel_length * 1.5))
                rotated_kernel = cv2.warpAffine(line_kernel_base, rot_matrix, (bbox_size, bbox_size))

                if np.sum(rotated_kernel) > 0:
                    rotated_kernel = rotated_kernel.astype(np.float32) / np.sum(rotated_kernel)
                else:
                    continue

                response = cv2.filter2D(enhanced_image.astype(np.float32), -1, rotated_kernel)
                max_response_map = np.maximum(max_response_map, response)

        if np.max(max_response_map) > 0:
            cv2.normalize(max_response_map, max_response_map, 0, 255, cv2.NORM_MINMAX)
        response_8u = max_response_map.astype(np.uint8)

        _, scratch_mask = cv2.threshold(response_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use a generic ellipse for closing since scratches can be at any orientation
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, close_kernel)

        scratch_mask = cv2.bitwise_and(scratch_mask, scratch_mask, mask=zone_mask)
        _log_duration(f"LEI Scratch Detection for {zone_name}", lei_start_time, self.current_image_result)
        return scratch_mask

    def _detect_defects_canny(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """
        Detects line-like defects using Canny edges plus morphological cleaning.
        Returns a 0/255 binary mask of candidate edges within the zone.
        """
        canny_start_time = _start_timer()
        _log_message(f"Starting Canny defect detection for zone '{zone_name}'...")

        if image_gray is None or zone_mask is None:
            _log_message("Input image or mask is None for Canny.", level="ERROR")
            return None

        edges = cv2.Canny(image_gray, self.config.CANNY_LOW_THRESHOLD, self.config.CANNY_HIGH_THRESHOLD)
        # Dilate to thicken edges, then erode to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        canny_mask = np.zeros_like(image_gray, dtype=np.uint8)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.LEI_MIN_SCRATCH_AREA_PX:
                continue
            cv2.drawContours(canny_mask, [cnt], -1, color=255, thickness=cv2.FILLED)

        canny_mask = cv2.bitwise_and(canny_mask, canny_mask, mask=zone_mask)
        _log_duration(f"Canny Detection for {zone_name}", canny_start_time, self.current_image_result)
        return canny_mask

    def _detect_defects_adaptive_thresh(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
        """
        Detects defects using adaptive thresholding plus morphological operations.
        Returns a binary (0/255) mask of candidate defects within the zone.
        """
        _log_message(f"Starting adaptive threshold defect detection for zone '{zone_name}'...")
        if image_gray is None or zone_mask is None:
            _log_message("Input image or mask is None for adaptive_thresh.", level="ERROR")
            return None

        # Mask the grayscale zone
        masked = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask)

        try:
            block_size = self.config.ADAPTIVE_THRESH_BLOCK_SIZE
            if block_size % 2 == 0:
                block_size += 1
            thresh_img = cv2.adaptiveThreshold(
                masked,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                self.config.ADAPTIVE_THRESH_C
            )
        except Exception as e:
            _log_message(f"Error during adaptive threshold for zone '{zone_name}': {e}", level="WARNING")
            return None

        # Morphological opening & closing
        open_k = (3, 3)
        if open_k[0] > 0 and open_k[1] > 0:
            open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, open_k)
            thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, open_kernel)

        close_k = (3, 3)
        if close_k[0] > 0 and close_k[1] > 0:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, close_k)
            thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, close_kernel)

        # Ensure inside the zone
        defect_mask = cv2.bitwise_and(thresh_img, thresh_img, mask=zone_mask)
        return defect_mask

    def _combine_defect_masks(self, all_defect_maps: Dict[str, Optional[np.ndarray]], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Combines all intermediate defect maps into one final binary mask.
        We simply OR all non‐None masks together so that any pixel flagged by at least one method remains.
        """
        combined = np.zeros(image_shape, dtype=np.uint8)
        for key, mask in all_defect_maps.items():
            if mask is None:
                continue
            binary_mask = (mask > 0).astype(np.uint8) * 255
            combined = cv2.bitwise_or(combined, binary_mask)
        return combined

    def _analyze_defect_contours(
        self,
        combined_mask: np.ndarray,
        image_filename: str,
        all_defect_maps: Dict[str, Optional[np.ndarray]]
    ) -> List[DefectInfo]:
        """
        Finds all contours in the combined_mask, computes metrics, and returns a list of DefectInfo.
        """
        defects_list: List[DefectInfo] = []
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        defect_id = 1
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < self.config.MIN_DEFECT_AREA_PX:
                continue  # Skip tiny artifacts

            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            # Determine zone by checking which zone mask contains the centroid
            zone_name = "unknown"
            for zid, zone_info in self.current_image_result.detected_zones.items():
                if zone_info.mask[cy, cx] > 0:
                    zone_name = zid
                    break

            # Determine detection methods that flagged this contour
            detected_methods: List[str] = []
            for method_key, mask in all_defect_maps.items():
                if mask is None:
                    continue
                mask_roi = mask[y : y + h, x : x + w]
                cnt_adjusted = cnt.copy()
                cnt_adjusted[:, 0, 0] -= x
                cnt_adjusted[:, 0, 1] -= y
                # Draw that contour in a zeroed ROI, then AND with mask_roi
                roi_contour = np.zeros_like(mask_roi)
                cv2.drawContours(roi_contour, [cnt_adjusted], -1, color=255, thickness=-1)
                if cv2.countNonZero(roi_contour & mask_roi) > 0:
                    method_name = method_key.split("_")[0]
                    if method_name not in detected_methods:
                        detected_methods.append(method_name)

            # Assign defect_type
            defect_type = "Region"
            if "lei" in detected_methods:
                defect_type = "Scratch"
            elif "canny" in detected_methods and "do2mr" not in detected_methods:
                defect_type = "Scratch"
            elif "do2mr" in detected_methods:
                defect_type = "Region"

            # Measurements
            area = DefectMeasurement(value_px=area_px)
            perimeter_px = cv2.arcLength(cnt, True)
            perimeter = DefectMeasurement(value_px=perimeter_px)
            major_dim = DefectMeasurement(value_px=max(w, h))
            minor_dim = DefectMeasurement(value_px=min(w, h))

            if self.pixels_per_micron and self.pixels_per_micron > 0:
                area.value_um = area_px / (self.pixels_per_micron ** 2)
                perimeter.value_um = perimeter_px / self.pixels_per_micron
                major_dim.value_um = major_dim.value_px / self.pixels_per_micron
                minor_dim.value_um = minor_dim.value_px / self.pixels_per_micron

            di = DefectInfo(
                defect_id=defect_id,
                zone_name=zone_name,
                defect_type=defect_type,
                centroid_px=(cx, cy),
                bounding_box_px=(x, y, w, h),
                area=area,
                perimeter=perimeter,
                major_dimension=major_dim,
                minor_dimension=minor_dim,
                confidence_score=1.0,
                detection_methods=detected_methods,
                contour=cnt
            )
            defects_list.append(di)
            defect_id += 1

        return defects_list

    # --- Main Orchestration Methods (Part 3) ---

    def process_single_image(self, image_path: Path) -> ImageResult:
        """
        Orchestrates the full analysis pipeline for a single image, returning an ImageResult.
        """
        single_image_start_time = _start_timer()
        _log_message(f"--- Starting processing for image: {image_path.name} ---")

        # Initialize ImageResult
        self.current_image_result = ImageResult(
            filename=image_path.name,
            timestamp=datetime.now(),
            fiber_specs_used=self.fiber_specs,
            operating_mode=self.operating_mode
        )

        original_bgr_image = self._load_single_image(image_path)
        if original_bgr_image is None:
            self.current_image_result.error_message = "Failed to load image."
            self.current_image_result.stats.status = "Error"
            _log_duration(f"Processing {image_path.name} (Load Error)", single_image_start_time, self.current_image_result)
            return self.current_image_result

        processed_images = self._preprocess_image(original_bgr_image)
        if not processed_images:
            self.current_image_result.error_message = "Image preprocessing failed."
            self.current_image_result.stats.status = "Error"
            _log_duration(f"Processing {image_path.name} (Preproc Error)", single_image_start_time, self.current_image_result)
            return self.current_image_result

        center_radius_tuple = self._find_fiber_center_and_radius(processed_images)
        if center_radius_tuple is None:
            self.current_image_result.error_message = "Could not detect fiber center/cladding."
            self.current_image_result.stats.status = "Error - No Fiber"
            _log_duration(f"Processing {image_path.name} (No Fiber)", single_image_start_time, self.current_image_result)
            return self.current_image_result

        fiber_center_px, detected_cladding_radius_px = center_radius_tuple
        self._calculate_pixels_per_micron(detected_cladding_radius_px)
        if self.operating_mode == "MICRON_INFERRED" and not self.pixels_per_micron:
            _log_message("MICRON_INFERRED failed. Effective mode: PIXEL_ONLY.", level="WARNING")
            self.current_image_result.operating_mode = "PIXEL_ONLY (Inference Failed)"

        # Create zone masks
        self.current_image_result.detected_zones = self._create_zone_masks(
            original_bgr_image.shape[:2],
            fiber_center_px,
            detected_cladding_radius_px
        )

        # Run all defect detection methods per zone
        all_defect_maps: Dict[str, Optional[np.ndarray]] = {}
        for zone_name, zone_info in self.current_image_result.detected_zones.items():
            if zone_info.mask is None:
                continue
            _log_message(f"Detecting defects in zone: {zone_name}")
            gray_detect = processed_images.get('clahe_enhanced', processed_images['original_gray'])

            all_defect_maps[f"do2mr_{zone_name}"] = self._detect_region_defects_do2mr(gray_detect, zone_info.mask, zone_name)
            all_defect_maps[f"lei_{zone_name}"] = self._detect_scratches_lei(gray_detect, zone_info.mask, zone_name)
            all_defect_maps[f"canny_{zone_name}"] = self._detect_defects_canny(processed_images.get('gaussian_blurred', gray_detect), zone_info.mask, zone_name)
            all_defect_maps[f"adaptive_thresh_{zone_name}"] = self._detect_defects_adaptive_thresh(processed_images.get('bilateral_filtered', gray_detect), zone_info.mask, zone_name)

        # Store intermediate maps
        self.current_image_result.intermediate_defect_maps = {k: v for k, v in all_defect_maps.items() if v is not None}

        # Combine maps and analyze contours
        final_combined_mask = self._combine_defect_masks(all_defect_maps, original_bgr_image.shape[:2])
        self.current_image_result.defects = self._analyze_defect_contours(final_combined_mask, image_path.name, all_defect_maps)

        # Update statistics
        stats = self.current_image_result.stats
        stats.total_defects = len(self.current_image_result.defects)
        for defect in self.current_image_result.defects:
            if defect.zone_name == "core":
                stats.core_defects += 1
            elif defect.zone_name == "cladding":
                stats.cladding_defects += 1
            elif defect.zone_name == "ferrule_contact":
                stats.ferrule_defects += 1
            elif defect.zone_name == "adhesive":
                stats.adhesive_defects += 1
        stats.status = "Review"  # Default; you can update later based on pass/fail logic

        # Save artifacts (annotated image, CSVs, histogram, etc.)
        self._save_image_artifacts(original_bgr_image, self.current_image_result)

        stats.processing_time_s = _log_duration(f"Processing {image_path.name}", single_image_start_time)
        _log_message(f"--- Finished processing for image: {image_path.name} ---")
        return self.current_image_result

    def process_image_batch(self, image_paths: List[Path]):
        """
        Processes a batch of images provided as a list of Paths.
        Generates both per-image and batch summary CSVs.
        """
        batch_start_time = _start_timer()
        _log_message(f"Starting batch processing for {len(image_paths)} images...")
        self.batch_summary_results = []

        for i, image_path in enumerate(image_paths):
            _log_message(f"Processing image {i+1}/{len(image_paths)}: {image_path.name}")
            image_res = self.process_single_image(image_path)

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
                "Microns_Per_Pixel": f"{1.0/self.pixels_per_micron:.4f}" if self.pixels_per_micron and self.pixels_per_micron > 0 else "N/A",
                "Error": image_res.error_message if image_res.error_message else ""
            }
            self.batch_summary_results.append(summary_item)

            if image_res.operating_mode == "MICRON_INFERRED":
                # Reset for next image
                self.pixels_per_micron = None
                # Decide next operating_mode
                if (
                    self.fiber_specs.cladding_diameter_um and self.fiber_specs.cladding_diameter_um > 0
                    and self.fiber_specs.core_diameter_um and self.fiber_specs.core_diameter_um > 0
                ):
                    self.operating_mode = "MICRON_CALCULATED"
                else:
                    self.operating_mode = "PIXEL_ONLY"

                self._initialize_zone_parameters()

        self._save_batch_summary_report_csv()
        _log_duration("Batch Processing", batch_start_time)
        _log_message(f"--- Batch processing complete. {len(image_paths)} images processed. ---")

    def _save_image_artifacts(self, original_bgr_image: np.ndarray, image_res: ImageResult):
        """
        Saves all generated artifacts for a single image:
        - Annotated image (with zones & defect boxes)
        - Per-image CSV report
        - Polar histogram
        - (Optionally) intermediate defect maps
        """
        _log_message(f"Saving artifacts for {image_res.filename}...")
        image_specific_output_dir = self.output_dir_path / Path(image_res.filename).stem
        image_specific_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Annotated image
        if self.config.SAVE_ANNOTATED_IMAGE:
            annotated_img = self._generate_annotated_image(original_bgr_image, image_res)
            if annotated_img is not None:
                annotated_path = image_specific_output_dir / f"{Path(image_res.filename).stem}_annotated.jpg"
                cv2.imwrite(str(annotated_path), annotated_img)
                image_res.annotated_image_path = annotated_path
                _log_message(f"Annotated image saved to {annotated_path}")

        # 2. Detailed CSV per image
        if self.config.DETAILED_REPORT_PER_IMAGE and image_res.defects:
            self._save_individual_image_report_csv(image_res, image_specific_output_dir)

        # 3. Polar histogram
        if self.config.SAVE_HISTOGRAM:
            histogram_fig = self._generate_defect_histogram(image_res)
            if histogram_fig:
                histogram_path = image_specific_output_dir / f"{Path(image_res.filename).stem}_histogram.png"
                histogram_fig.savefig(str(histogram_path), dpi=150)
                plt.close(histogram_fig)
                image_res.histogram_path = histogram_path
                _log_message(f"Defect histogram saved to {histogram_path}")

        # 4. Intermediate defect maps (optional)
        if self.config.SAVE_DEFECT_MAPS and image_res.intermediate_defect_maps:
            maps_dir = image_specific_output_dir / "defect_maps"
            maps_dir.mkdir(exist_ok=True)
            for map_name, defect_map_img in image_res.intermediate_defect_maps.items():
                if defect_map_img is not None:
                    cv2.imwrite(str(maps_dir / f"{map_name}.png"), defect_map_img)
            _log_message(f"Intermediate defect maps saved to {maps_dir}")

        _log_message(f"Artifacts saved for {image_res.filename}.")

    def _save_individual_image_report_csv(self, image_res: ImageResult, output_dir: Path):
        """
        Saves a detailed CSV listing every defect found in a single image.
        Each row has: ID, Zone, Type, Centroid_px, Centroid_um (if available), 
        BBox_px, BBox_um, Area_px, Area_um, Perim_px, Perim_um, Major_px, Major_um, Minor_px, Minor_um, Confidence, Methods
        """
        report_path = output_dir / f"{Path(image_res.filename).stem}_report.csv"
        try:
            with open(report_path, mode='w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    "Defect_ID", "Zone", "Type", "Centroid_X_px", "Centroid_Y_px", "Centroid_X_um", "Centroid_Y_um",
                    "BBox_X_px", "BBox_Y_px", "BBox_W_px", "BBox_H_px",
                    "BBox_W_um", "BBox_H_um", "Area_px", "Area_um", "Perimeter_px", "Perimeter_um",
                    "Major_px", "Major_um", "Minor_px", "Minor_um", "Confidence", "Detection_Methods"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for defect in image_res.defects:
                    cx_px, cy_px = defect.centroid_px
                    cx_um = (cx_px / self.pixels_per_micron) if (defect.centroid_px and self.pixels_per_micron) else None
                    cy_um = (cy_px / self.pixels_per_micron) if (defect.centroid_px and self.pixels_per_micron) else None
                    x_px, y_px, w_px, h_px = defect.bounding_box_px
                    w_um = (w_px / self.pixels_per_micron) if (w_px and self.pixels_per_micron) else None
                    h_um = (h_px / self.pixels_per_micron) if (h_px and self.pixels_per_micron) else None

                    writer.writerow({
                        "Defect_ID": defect.defect_id,
                        "Zone": defect.zone_name,
                        "Type": defect.defect_type,
                        "Centroid_X_px": cx_px, "Centroid_Y_px": cy_px,
                        "Centroid_X_um": f"{cx_um:.2f}" if cx_um is not None else "N/A",
                        "Centroid_Y_um": f"{cy_um:.2f}" if cy_um is not None else "N/A",
                        "BBox_X_px": x_px, "BBox_Y_px": y_px, "BBox_W_px": w_px, "BBox_H_px": h_px,
                        "BBox_W_um": f"{w_um:.2f}" if w_um is not None else "N/A",
                        "BBox_H_um": f"{h_um:.2f}" if h_um is not None else "N/A",
                        "Area_px": f"{defect.area.value_px:.2f}" if defect.area.value_px is not None else "N/A",
                        "Area_um": f"{defect.area.value_um:.2f}" if defect.area.value_um is not None else "N/A",
                        "Perimeter_px": f"{defect.perimeter.value_px:.2f}" if defect.perimeter.value_px is not None else "N/A",
                        "Perimeter_um": f"{defect.perimeter.value_um:.2f}" if defect.perimeter.value_um is not None else "N/A",
                        "Major_px": f"{defect.major_dimension.value_px:.2f}" if defect.major_dimension.value_px is not None else "N/A",
                        "Major_um": f"{defect.major_dimension.value_um:.2f}" if defect.major_dimension.value_um is not None else "N/A",
                        "Minor_px": f"{defect.minor_dimension.value_px:.2f}" if defect.minor_dimension.value_px is not None else "N/A",
                        "Minor_um": f"{defect.minor_dimension.value_um:.2f}" if defect.minor_dimension.value_um is not None else "N/A",
                        "Confidence": f"{defect.confidence_score:.3f}",
                        "Detection_Methods": "; ".join(defect.detection_methods)
                    })
            _log_message(f"Individual CSV report saved to {report_path}")
        except Exception as e:
            _log_message(f"Error saving per-image CSV report: {e}", level="ERROR")

    def _generate_annotated_image(self, original_bgr_image: np.ndarray, image_res: ImageResult) -> Optional[np.ndarray]:
        """
        Draws colored circles for zones and bounding boxes for each defect, plus text.
        Returns the annotated BGR image.
        """
        if original_bgr_image is None or not image_res.detected_zones:
            _log_message("Insufficient data to generate annotated image.", level="WARNING")
            return None

        annotated_image = original_bgr_image.copy()

        # Draw zone boundaries
        for zone_name, zone_info in image_res.detected_zones.items():
            color = next(
                (zd.color_bgr for zd in self.active_zone_definitions if zd.name == zone_name),
                (255, 255, 255)
            )
            cx, cy = zone_info.center_px
            r = int(zone_info.radius_px)
            cv2.circle(annotated_image, (cx, cy), r, color, thickness=2)

            cv2.putText(
                annotated_image,
                zone_name.upper(),
                (cx - r, cy - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.FONT_SCALE,
                color,
                self.config.LINE_THICKNESS
            )

        # Draw each defect
        for defect in image_res.defects:
            x, y, w, h = defect.bounding_box_px
            defect_color = self.config.DEFECT_COLORS.get(defect.defect_type, (0, 0, 255))
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), defect_color, self.config.LINE_THICKNESS)
            label = f"{defect.defect_type}:{defect_id}"
            cv2.putText(
                annotated_image,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.FONT_SCALE,
                defect_color,
                self.config.LINE_THICKNESS
            )

        # Add overall status and defect counts to the image
        cv2.putText(
            annotated_image,
            f"File: {image_res.filename}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.FONT_SCALE * 1.1,
            (230, 230, 230),
            self.config.LINE_THICKNESS
        )
        cv2.putText(
            annotated_image,
            f"Status: {image_res.stats.status}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.FONT_SCALE * 1.1,
            (230, 230, 230),
            self.config.LINE_THICKNESS
        )
        cv2.putText(
            annotated_image,
            f"Total Defects: {image_res.stats.total_defects}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.FONT_SCALE * 1.1,
            (230, 230, 230),
            self.config.LINE_THICKNESS
        )

        _log_message(f"Annotated image generated for {image_res.filename}.")
        return annotated_image

    def _generate_defect_histogram(self, image_res: ImageResult) -> Optional[plt.Figure]:
        """
        Generates a polar histogram (scatter) of defect distribution (angle vs radius).
        Returns a matplotlib Figure, or None if no data.
        """
        _log_message(f"Generating defect histogram for {image_res.filename}...")

        if not image_res.defects or not image_res.detected_zones.get("cladding"):
            _log_message("No defects or cladding center not found, skipping histogram.", level="WARNING")
            return None

        cladding_zone_info = image_res.detected_zones["cladding"]
        fiber_center_x, fiber_center_y = cladding_zone_info.center_px

        angles, radii, defect_plot_colors = [], [], []
        for defect in image_res.defects:
            dx = defect.centroid_px[0] - fiber_center_x
            dy = defect.centroid_px[1] - fiber_center_y
            angles.append(np.arctan2(dy, dx))
            radii.append(np.sqrt(dx ** 2 + dy ** 2))

            bgr_color = self.config.DEFECT_COLORS.get(defect.defect_type, (0, 0, 0))
            rgb_color = (bgr_color[2] / 255.0, bgr_color[1] / 255.0, bgr_color[0] / 255.0)
            defect_plot_colors.append(rgb_color)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        ax.scatter(angles, radii, c=defect_plot_colors, s=50, alpha=0.75, edgecolors='k')

        # Draw zone boundaries
        for zone_name, zone_info in image_res.detected_zones.items():
            zone_def = next((zd for zd in self.active_zone_definitions if zd.name == zone_name), None)
            if not zone_def:
                continue
            # Use radius_px for plotting
            r = zone_info.radius_px
            theta = np.linspace(0, 2 * np.pi, 360)
            ax.plot(theta, [r] * 360, color=np.array(zone_def.color_bgr)[::-1] / 255.0, linestyle='--')

        ax.set_title(f"Defect Distribution: {image_res.filename}", va='bottom')
        return fig

    def _save_batch_summary_report_csv(self):
        """
        Saves a summary CSV report for the entire batch, using self.batch_summary_results.
        """
        _log_message("Saving batch summary report...")
        if not self.batch_summary_results:
            _log_message("No batch results to save.", level="WARNING")
            return

        summary_path = self.output_dir_path / self.config.BATCH_SUMMARY_FILENAME
        try:
            summary_df = pd.DataFrame(self.batch_summary_results)
            summary_df.to_csv(summary_path, index=False, encoding='utf-8')
            _log_message(f"Batch summary report saved to {summary_path}")
        except Exception as e:
            _log_message(f"Error saving batch summary report: {e}", level="ERROR")

# --- Main Execution Function ---

def main():
    """Main function to drive the fiber inspection script."""
    print("=" * 70)
    print(" Advanced Automated Optical Fiber End Face Inspector")
    print("=" * 70)
    script_start_time = _start_timer()

    try:
        config = InspectorConfig()
        inspector = FiberInspector(config)
        inspector._get_user_specifications()

        image_paths = inspector._get_image_paths_from_user()
        if not image_paths:
            _log_message("No images to process. Exiting.", level="INFO")
            return

        inspector.process_image_batch(image_paths)
    except FileNotFoundError as fnf_error:
        _log_message(f"Error: {fnf_error}", level="CRITICAL")
    except ValueError as val_error:
        _log_message(f"Input Error: {val_error}", level="CRITICAL")
    except Exception as e:
        _log_message(f"An unexpected error occurred: {e}", level="CRITICAL")
        import traceback; traceback.print_exc()
    finally:
        _log_duration("Total Script Execution", script_start_time)
        print("=" * 70)
        print("Inspection Run Finished.")
        print("=" * 70)

# --- Script Entry Point ---
if __name__ == "__main__":
    main()
