#!/usr/bin/env python3
"""
Advanced Fiber Optic End Face Defect Detection System
=====================================================
This script implements a highly accurate, multi‐method approach to detecting defects
on fiber optic connector end faces. It combines DO2MR (Difference of Min‐Max Ranking)
for region‐based defects and LEI (Linear Enhancement Inspector) for scratch detection,
along with other CV techniques, and provides detailed reporting.

Author: Gemini AI
Date: June 4, 2025
Version: 1.2 (Corrected and Integrated)
"""

# ------------------------------------------------------------------------------
# Imports (all necessary libraries for image processing, numerical operations, and visualization)
# ------------------------------------------------------------------------------
import cv2                    # OpenCV for image processing
import numpy as np            # NumPy for efficient numerical computations
import matplotlib.pyplot as plt  # Matplotlib for generating plots and visualizations
import os                     # Operating system interface for file operations
import csv                    # CSV file handling
import json                   # JSON handling
import time                   # Time tracking for performance monitoring
import warnings               # Warning handling
from datetime import datetime # Datetime for timestamping
from pathlib import Path      # Path handling for cross‐platform compatibility
from typing import (
    Dict, List, Tuple,
    Optional, Union, Any
)                             # Type hints for better code clarity
from dataclasses import dataclass, field  # Dataclasses for structured data
import pandas as pd           # Pandas for easy CSV writing of batch summary

# Suppress non‐critical warnings (e.g., from OpenCV operations on empty arrays)
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# Part 1: Configuration, Data Structures, and Initial Setup
# ------------------------------------------------------------------------------

@dataclass
class FiberSpecifications:
    """
    Data structure to hold user‐provided (or default) fiber optic specifications.
    If the user provides core/cladding diameters, we will convert px→µm; otherwise we stay in PIXEL_ONLY mode.
    """
    core_diameter_um: Optional[float] = None       # Core diameter in microns
    cladding_diameter_um: Optional[float] = 125.0  # Cladding diameter in microns (default)
    ferrule_diameter_um: Optional[float] = 250.0   # Ferrule outer diameter in microns (approximate)
    fiber_type: str = "unknown"                    # e.g. "single‐mode", "multi‐mode"

@dataclass
class ZoneDefinition:
    """
    Data structure to define the “zones” on an end‐face:
      - name:           e.g. 'core', 'cladding', 'ferrule_contact', 'adhesive'
      - r_min_factor_or_um, r_max_factor_or_um: Either (unitless) factors (if PIXEL_ONLY/MICRON_INFERRED),
                                                or actual radii in µm (if MICRON_CALCULATED).
      - color_bgr:      BGR color for drawing this zone
      - max_defect_size_um: For PASS/FAIL decisions (not used in this demonstration, but stored)
      - defects_allowed:     True/False
    """
    name: str
    r_min_factor_or_um: float
    r_max_factor_or_um: float
    color_bgr: Tuple[int, int, int]
    max_defect_size_um: Optional[float] = None
    defects_allowed: bool = True

@dataclass
class DetectedZoneInfo:
    """
    Data structure to hold information about a detected zone in an image:
      - name: e.g. 'core'
      - center_px: (x,y) center in pixels (same for all zones, but stored for convenience)
      - radius_px: representative outer radius in pixels
      - radius_um: radius in µm (if known / if conversion applied)
      - mask:   np.ndarray mask of shape (H,W), uint8, with 255 inside the zone, 0 outside.
      - color_bgr: (B,G,R) tuple used when drawing this zone on the annotated image
    """
    name: str
    center_px: Tuple[int, int]
    radius_px: float
    radius_um: Optional[float]
    mask: Optional[np.ndarray]
    color_bgr: Tuple[int, int, int]  # <--- Newly added field

@dataclass
class DefectMeasurement:
    """Holds a single scalar measurement in pixels and µm (if conversion known)."""
    value_px: Optional[float] = None
    value_um: Optional[float] = None

@dataclass
class DefectInfo:
    """
    Data structure to hold detailed information about a detected defect:
      - defect_id:    unique ID
      - zone_name:    which zone (core/cladding/ferrule_contact/adhesive)
      - defect_type:  'Region' or 'Scratch'
      - centroid_px:  (x,y) in pixels
      - bounding_box_px: (x,y,width,height) in pixels
      - area:         DefectMeasurement (px and µm)
      - perimeter:    DefectMeasurement (px and µm)
      - major_dimension: DefectMeasurement (e.g. scratch length or pit diameter)
      - minor_dimension: DefectMeasurement (e.g. scratch width)
      - confidence_score: float 0.0–1.0
      - detection_methods: List[str], e.g. ['do2mr','lei']
    """
    defect_id: int
    zone_name: str
    defect_type: str
    centroid_px: Tuple[int, int]
    bounding_box_px: Tuple[int, int, int, int]
    area: DefectMeasurement = field(default_factory=DefectMeasurement)
    perimeter: DefectMeasurement = field(default_factory=DefectMeasurement)
    major_dimension: DefectMeasurement = field(default_factory=DefectMeasurement)
    minor_dimension: DefectMeasurement = field(default_factory=DefectMeasurement)
    confidence_score: float = 0.0
    detection_methods: List[str] = field(default_factory=list)
    contour: Optional[np.ndarray] = None

@dataclass
class ImageAnalysisStats:
    """
    Statistics for a single image analysis (for the summary, and status).
    """
    total_defects: int = 0
    core_defects: int = 0
    cladding_defects: int = 0
    ferrule_defects: int = 0
    adhesive_defects: int = 0
    processing_time_s: float = 0.0
    status: str = "Pending"   # Could be 'Pass'/'Fail'/'Review'
    microns_per_pixel: Optional[float] = None  # µm/px if known

@dataclass
class ImageResult:
    """
    Data structure to store _all_ results for a single processed image:
      - filename, timestamp, fiber_specs_used, operating_mode
      - detected_zones: Dict[name --> DetectedZoneInfo]
      - defects: List[DefectInfo]
      - stats: ImageAnalysisStats
      - annotated_image_path, report_csv_path, histogram_path
      - error_message: str if processing failed
      - intermediate_defect_maps: For debugging (not saved unless configured)
    """
    filename: str
    timestamp: datetime
    fiber_specs_used: FiberSpecifications
    operating_mode: str  # 'PIXEL_ONLY' or 'MICRON_CALCULATED' or 'MICRON_INFERRED'
    detected_zones: Dict[str, DetectedZoneInfo] = field(default_factory=dict)
    defects: List[DefectInfo] = field(default_factory=list)
    stats: ImageAnalysisStats = field(default_factory=ImageAnalysisStats)
    annotated_image_path: Optional[Path] = None
    report_csv_path: Optional[Path] = None
    histogram_path: Optional[Path] = None
    error_message: Optional[str] = None
    intermediate_defect_maps: Dict[str, np.ndarray] = field(default_factory=dict)
    timing_log: Dict[str, float] = field(default_factory=dict)

@dataclass
class InspectorConfig:
    """
    Holds _all_ configuration parameters.  You can adjust these for tuning.
    """
    # Output and Reporting
    OUTPUT_DIR_NAME: str = "fiber_inspection_output"
    BATCH_SUMMARY_FILENAME: str = "batch_inspection_summary.csv"
    DETAILED_REPORT_PER_IMAGE: bool = True
    SAVE_ANNOTATED_IMAGE: bool = True
    SAVE_DEFECT_MAPS: bool = False    # If True, saves intermediate masks for debugging
    SAVE_HISTOGRAM: bool = True

    # Minimal defect area (px)
    MIN_DEFECT_AREA_PX: int = 10

    # Calibration (not fully implemented here, but reserved)
    PERFORM_CALIBRATION: bool = False
    CALIBRATION_IMAGE_PATH: Optional[str] = None
    CALIBRATION_DOT_SPACING_UM: float = 10.0
    CALIBRATION_FILE_JSON: str = "calibration_data.json"

    # Default Zone Definitions (PIXEL_ONLY mode uses these as relative factors)
    DEFAULT_ZONES: List[ZoneDefinition] = field(default_factory=lambda: [
        ZoneDefinition(name="core", r_min_factor_or_um=0.0, r_max_factor_or_um=0.4,
                       color_bgr=(255, 0, 0), max_defect_size_um=5.0, defects_allowed=True),
        ZoneDefinition(name="cladding", r_min_factor_or_um=0.4, r_max_factor_or_um=1.0,
                       color_bgr=(0, 255, 0), max_defect_size_um=10.0, defects_allowed=True),
        ZoneDefinition(name="ferrule_contact", r_min_factor_or_um=1.0, r_max_factor_or_um=2.0,
                       color_bgr=(0, 0, 255), max_defect_size_um=25.0, defects_allowed=True),
        ZoneDefinition(name="adhesive", r_min_factor_or_um=2.0, r_max_factor_or_um=2.2,
                       color_bgr=(0, 255, 255), max_defect_size_um=50.0, defects_allowed=False)
    ])

    # Preprocessing
    GAUSSIAN_BLUR_KERNEL_SIZE: Tuple[int, int] = (7, 7)
    GAUSSIAN_BLUR_SIGMA: int = 2
    BILATERAL_FILTER_D: int = 9
    BILATERAL_FILTER_SIGMA_COLOR: int = 75
    BILATERAL_FILTER_SIGMA_SPACE: int = 75
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)

    # Hough Circle (zone detection)
    HOUGH_DP_VALUES: List[float] = field(default_factory=lambda: [1.0, 1.2, 1.5])
    HOUGH_MIN_DIST_FACTOR: float = 0.1
    HOUGH_PARAM1_VALUES: List[int] = field(default_factory=lambda: [50, 70, 100])
    HOUGH_PARAM2_VALUES: List[int] = field(default_factory=lambda: [25, 30, 40])
    HOUGH_MIN_RADIUS_FACTOR: float = 0.05
    HOUGH_MAX_RADIUS_FACTOR: float = 0.6
    CIRCLE_CONFIDENCE_THRESHOLD: float = 0.3  # Discard circles below this score

    # DO2MR (Region‐based detector)
    DO2MR_KERNEL_SIZES: List[Tuple[int, int]] = field(default_factory=lambda: [(5, 5), (9, 9), (13, 13)])
    DO2MR_GAMMA_VALUES: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0])
    DO2MR_MEDIAN_BLUR_KERNEL_SIZE: int = 5
    DO2MR_MORPH_OPEN_KERNEL_SIZE: Tuple[int, int] = (3, 3)

    # LEI (Scratch detector)
    LEI_KERNEL_LENGTHS: List[int] = field(default_factory=lambda: [11, 17, 23])
    LEI_ANGLE_STEP: int = 15  # degrees
    LEI_THRESHOLD_FACTOR: float = 2.0
    LEI_MORPH_CLOSE_KERNEL_SIZE: Tuple[int, int] = (5, 1)
    LEI_MIN_SCRATCH_AREA_PX: int = 15

    # Additional defect‐finding parameters (for fallback methods)
    CANNY_LOW_THRESHOLD: int = 50
    CANNY_HIGH_THRESHOLD: int = 150
    ADAPTIVE_THRESH_BLOCK_SIZE: int = 11
    ADAPTIVE_THRESH_C: int = 2

    # Ensemble / confidence
    MIN_METHODS_FOR_CONFIRMED_DEFECT: int = 2
    CONFIDENCE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "do2mr": 1.0,
        "lei": 1.0,
        "canny": 0.6,
        "adaptive_thresh": 0.7,
        "otsu_global": 0.5
    })

    # Visualization
    DEFECT_COLORS: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "Region": (0, 255, 255),       # Yellow
        "Scratch": (255, 0, 255),      # Magenta
        "Contamination": (255, 165, 0), # Orange
        "Pit": (0, 128, 255),          # Light Orange
        "Chip": (128, 0, 128),         # Purple
        "Linear Region": (255, 105, 180)  # Hot Pink
    })
    FONT_SCALE: float = 0.5
    LINE_THICKNESS: int = 1


# -------------------------------
# Utility & Logging Functions
# -------------------------------

def _log_message(message: str, level: str = "INFO"):
    """
    Prints a timestamped log message to the console.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{current_time}] [{level.upper()}] {message}")

def _start_timer() -> float:
    """
    Returns the current time (high‐resolution) to start a timer.
    """
    return time.perf_counter()

def _log_duration(operation_name: str, start_time: float, image_result: Optional[ImageResult] = None) -> float:
    """
    Logs the duration of an operation (and stores it in image_result.timing_log if provided).
    """
    duration = time.perf_counter() - start_time
    _log_message(f"Operation '{operation_name}' completed in {duration:.4f} seconds.")
    if image_result is not None:
        image_result.timing_log[operation_name] = duration
    return duration


# ------------------------------------------------------------------------------
# Part 2: Main Inspector Class (Combines config, setup, processing, detection, reporting)
# ------------------------------------------------------------------------------

class FiberInspector:
    """
    Main class to orchestrate the fiber optic end face inspection process.
    """

    def __init__(self, config: Optional[InspectorConfig] = None):
        """
        Initializes the FiberInspector instance with (optional) custom config.
        """
        # 1) Store configuration
        self.config = config if config else InspectorConfig()

        # 2) Default fiber specs and mode
        self.fiber_specs = FiberSpecifications()
        self.pixels_per_micron: Optional[float] = None
        self.operating_mode: str = "PIXEL_ONLY"

        # 3) Placeholder for the currently processed image’s results
        self.current_image_result: Optional[ImageResult] = None

        # 4) Will store summary dicts for the batch report
        self.batch_results_summary_list: List[Dict[str, Any]] = []

        # 5) Create main output directory (timestamped subfolder will be created later)
        self.output_dir_path: Path = Path(self.config.OUTPUT_DIR_NAME)
        self.output_dir_path.mkdir(parents=True, exist_ok=True)

        # 6) Prepare zone definitions (will be initialized properly in _initialize_zone_parameters)
        self.active_zone_definitions: List[ZoneDefinition] = []

        _log_message("FiberInspector initialized.", level="DEBUG")

        # 7) Initialize zone parameters based on default mode
        self._initialize_zone_parameters()


    # -------------------------
    # 1) USER‐SPECIFICATIONS
    # -------------------------
    def _get_user_specifications(self):
        """
        Prompts the user for fiber specifications (core/cladding diameters, fiber type).
        Updates self.fiber_specs and self.operating_mode accordingly.
        """
        start = _start_timer()
        _log_message("Prompting user for fiber specifications...")

        print("\n--- Fiber Optic Specifications ---")
        resp = input("Provide known fiber specifications in microns? (y/n, default=n): ").strip().lower()
        if resp == 'y':
            try:
                core_str = input(f"  Enter CORE diameter in µm (e.g. 9, 50, 62.5) [optional]: ").strip()
                if core_str:
                    self.fiber_specs.core_diameter_um = float(core_str)
                clad_str = input(f"  Enter CLADDING diameter in µm (default={self.fiber_specs.cladding_diameter_um}): ").strip()
                if clad_str:
                    self.fiber_specs.cladding_diameter_um = float(clad_str)
                ferrule_str = input(f"  Enter FERRULE outer diameter in µm (default={self.fiber_specs.ferrule_diameter_um}): ").strip()
                if ferrule_str:
                    self.fiber_specs.ferrule_diameter_um = float(ferrule_str)
                ftype_str = input("  Enter fiber type (e.g. single‐mode, multi‐mode) [optional]: ").strip()
                if ftype_str:
                    self.fiber_specs.fiber_type = ftype_str

                # Decide operating mode
                if (
                    self.fiber_specs.cladding_diameter_um is not None
                    and self.fiber_specs.cladding_diameter_um > 0
                ):
                    self.operating_mode = "MICRON_CALCULATED"
                    _log_message(f"Operating mode set to MICRON_CALCULATED "
                                 f"(Core={self.fiber_specs.core_diameter_um}, "
                                 f"Cladding={self.fiber_specs.cladding_diameter_um}, "
                                 f"Ferrule={self.fiber_specs.ferrule_diameter_um}, "
                                 f"Type={self.fiber_specs.fiber_type})")
                else:
                    self.operating_mode = "PIXEL_ONLY"
                    _log_message("Cladding diameter missing ⇒ PIXEL_ONLY mode", level="WARNING")

            except ValueError:
                _log_message("Invalid numeric input ⇒ falling back to PIXEL_ONLY mode", level="ERROR")
                self.operating_mode = "PIXEL_ONLY"
                self.fiber_specs = FiberSpecifications()  # reset
        else:
            # Default: PIXEL_ONLY
            _log_message("No specs provided ⇒ PIXEL_ONLY mode")

        _log_duration("User Specification Input", start)
        # Rebuild zone definitions based on chosen mode
        self._initialize_zone_parameters()


    # -------------------------
    # 2) INITIALIZE ZONE PARAMETERS
    # -------------------------
    def _initialize_zone_parameters(self):
        """
        Initializes self.active_zone_definitions based on current operating_mode and fiber_specs.
        If PIXEL_ONLY or MICRON_INFERRED, uses DEFAULT_ZONES (factors). If MICRON_CALCULATED, converts factors→µm→px.
        """
        _log_message("Initializing zone parameters...")

        if self.operating_mode == "MICRON_CALCULATED" and self.fiber_specs.cladding_diameter_um:
            # Compute radii in µm from diameters
            core_r_um = (self.fiber_specs.core_diameter_um or 0.0) / 2.0
            cladding_r_um = self.fiber_specs.cladding_diameter_um / 2.0
            ferrule_r_um = (self.fiber_specs.ferrule_diameter_um or (2.0 * cladding_r_um)) / 2.0
            adhesive_r_um = ferrule_r_um * 1.1  # 10% larger

            # Find default zone templates for colors and max_defect_size
            def_zones = {z.name: z for z in self.config.DEFAULT_ZONES}

            self.active_zone_definitions = [
                ZoneDefinition(
                    name="core",
                    r_min_factor_or_um=0.0,
                    r_max_factor_or_um=core_r_um,
                    color_bgr=def_zones['core'].color_bgr,
                    max_defect_size_um=def_zones['core'].max_defect_size_um
                ),
                ZoneDefinition(
                    name="cladding",
                    r_min_factor_or_um=core_r_um,
                    r_max_factor_or_um=cladding_r_um,
                    color_bgr=def_zones['cladding'].color_bgr,
                    max_defect_size_um=def_zones['cladding'].max_defect_size_um
                ),
                ZoneDefinition(
                    name="ferrule_contact",
                    r_min_factor_or_um=cladding_r_um,
                    r_max_factor_or_um=ferrule_r_um,
                    color_bgr=def_zones['ferrule_contact'].color_bgr,
                    max_defect_size_um=def_zones['ferrule_contact'].max_defect_size_um
                ),
                ZoneDefinition(
                    name="adhesive",
                    r_min_factor_or_um=ferrule_r_um,
                    r_max_factor_or_um=adhesive_r_um,
                    color_bgr=def_zones['adhesive'].color_bgr,
                    max_defect_size_um=def_zones['adhesive'].max_defect_size_um,
                    defects_allowed=def_zones['adhesive'].defects_allowed
                )
            ]
            _log_message(f"Zones for MICRON_CALCULATED: "
                         f"core_R={core_r_um}µm, clad_R={cladding_r_um}µm, ferrule_R={ferrule_r_um}µm")

        else:
            # For PIXEL_ONLY or MICRON_INFERRED, use DEFAULT_ZONES as "factors"
            self.active_zone_definitions = self.config.DEFAULT_ZONES.copy()
            _log_message(f"Zones set to default factors for {self.operating_mode}")



    # -------------------------
    # 3) GATHER IMAGE PATHS
    # -------------------------
    def _get_image_paths_from_user(self) -> List[Path]:
        """
        Prompts for a directory path, scans for all supported image files, and returns a list of Paths.
        """
        start = _start_timer()
        _log_message("Asking user to input image directory...")

        image_paths: List[Path] = []
        while True:
            dir_str = input("\nEnter the path to the directory containing fiber images: ").strip()
            image_dir = Path(dir_str)
            if not image_dir.is_dir():
                _log_message(f"'{image_dir}' is not a folder. Try again.", level="ERROR")
                continue

            # Supported extensions
            supported = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            for f in image_dir.iterdir():
                if f.is_file() and f.suffix.lower() in supported:
                    image_paths.append(f)

            if not image_paths:
                _log_message(f"No supported images in '{image_dir}'. Try another folder.", level="WARNING")
                # Loop again
            else:
                _log_message(f"Found {len(image_paths)} image(s) in '{image_dir}'")
                break

        _log_duration("Image Path Collection", start)
        return image_paths


    # -------------------------
    # 4) LOAD A SINGLE IMAGE
    # -------------------------
    def _load_single_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Loads a single image from disk with OpenCV. Converts BGRA→BGR or returns grayscale as needed.
        """
        _log_message(f"Loading image: '{image_path.name}'")
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                _log_message(f"Failed to load '{image_path.name}'", level="ERROR")
                return None

            # If BGRA, convert to BGR
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            _log_message(f"Image '{image_path.name}' loaded, shape={img.shape}")
            return img

        except Exception as e:
            _log_message(f"Exception in loading image '{image_path}': {e}", level="ERROR")
            return None


    # -------------------------
    # 5) PREPROCESSING
    # -------------------------
    def _preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Takes a BGR or grayscale image, returns a dict of processed grayscale images:
         - 'original_gray'
         - 'gaussian_blurred'
         - 'bilateral_filtered'
         - 'clahe_enhanced'
         - 'hist_equalized'
        """
        start = _start_timer()
        _log_message("Preprocessing image...")

        if image is None:
            _log_message("Cannot preprocess: image is None", level="ERROR")
            return {}

        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray = image.copy()
        else:
            _log_message(f"Unsupported image shape {image.shape}", level="ERROR")
            return {}

        processed = {}
        processed['original_gray'] = gray.copy()

        # 1) Gaussian Blur
        try:
            processed['gaussian_blurred'] = cv2.GaussianBlur(
                gray,
                self.config.GAUSSIAN_BLUR_KERNEL_SIZE,
                self.config.GAUSSIAN_BLUR_SIGMA
            )
        except Exception as e:
            _log_message(f"GaussianBlur failed: {e}", level="WARNING")
            processed['gaussian_blurred'] = gray.copy()

        # 2) Bilateral Filter
        try:
            processed['bilateral_filtered'] = cv2.bilateralFilter(
                gray,
                self.config.BILATERAL_FILTER_D,
                self.config.BILATERAL_FILTER_SIGMA_COLOR,
                self.config.BILATERAL_FILTER_SIGMA_SPACE
            )
        except Exception as e:
            _log_message(f"BilateralFilter failed: {e}", level="WARNING")
            processed['bilateral_filtered'] = gray.copy()

        # 3) CLAHE
        try:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.CLAHE_CLIP_LIMIT,
                tileGridSize=self.config.CLAHE_TILE_GRID_SIZE
            )
            processed['clahe_enhanced'] = clahe.apply(processed['bilateral_filtered'])
        except Exception as e:
            _log_message(f"CLAHE failed: {e}", level="WARNING")
            processed['clahe_enhanced'] = gray.copy()

        # 4) Histogram Equalization
        try:
            processed['hist_equalized'] = cv2.equalizeHist(gray)
        except Exception as e:
            _log_message(f"HistEqualize failed: {e}", level="WARNING")
            processed['hist_equalized'] = gray.copy()

        _log_duration("Image Preprocessing", start, self.current_image_result)
        return processed


    # -------------------------
    # 6) FIND FIBER CENTER & RADIUS
    # -------------------------
    def _find_fiber_center_and_radius(
        self,
        processed_images: Dict[str, np.ndarray]
    ) -> Optional[Tuple[Tuple[int,int], float]]:
        """
        Uses HoughCircles on the best processed images (Gaussian, Bilateral, CLAHE) with various parameters,
        collects all candidate circles, scores them by closeness to image center + radius size,
        and returns the circle with highest confidence (if above threshold).  Otherwise returns None.
        """
        start = _start_timer()
        _log_message("Running HoughCircles to detect cladding...")

        if 'original_gray' not in processed_images:
            _log_message("Missing 'original_gray' in processed_images", level="ERROR")
            _log_duration("Fiber Center Detection (fail)", start, self.current_image_result)
            return None

        h, w = processed_images['original_gray'].shape[:2]
        min_dist = int(min(h, w) * self.config.HOUGH_MIN_DIST_FACTOR)
        min_r = int(min(h, w) * self.config.HOUGH_MIN_RADIUS_FACTOR)
        max_r = int(min(h, w) * self.config.HOUGH_MAX_RADIUS_FACTOR)

        candidates: List[Tuple[int,int,int,float,str]] = []
        # Try each processing stage
        for key in ['gaussian_blurred', 'bilateral_filtered', 'clahe_enhanced']:
            if key not in processed_images:
                continue
            img = processed_images[key]
            for dp in self.config.HOUGH_DP_VALUES:
                for param1 in self.config.HOUGH_PARAM1_VALUES:
                    for param2 in self.config.HOUGH_PARAM2_VALUES:
                        try:
                            circles = cv2.HoughCircles(
                                img, cv2.HOUGH_GRADIENT,
                                dp=dp,
                                minDist=min_dist,
                                param1=param1,
                                param2=param2,
                                minRadius=min_r,
                                maxRadius=max_r
                            )
                            if circles is not None:
                                circles = np.uint16(np.around(circles))
                                for c in circles[0, :]:
                                    cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                                    # Score it: penalize distance from image center, but favor larger radius to a point
                                    dist_center = np.sqrt((cx - w//2)**2 + (cy - h//2)**2)
                                    norm_r = r / max_r if max_r > 0 else 0
                                    confidence = (param2 / max(self.config.HOUGH_PARAM2_VALUES)) * 0.5 \
                                               + 0.5 * norm_r \
                                               - 0.2 * (dist_center/(min(h,w)/2))
                                    confidence = max(0.0, min(1.0, confidence))
                                    candidates.append((cx, cy, r, confidence, key))
                        except Exception as e:
                            _log_message(f"HoughCircles error on {key} (dp={dp},p1={param1},p2={param2}): {e}", level="WARNING")

        if not candidates:
            _log_message("No circles found by HoughCircles.", level="WARNING")
            _log_duration("Fiber Center Detection (no candidates)", start, self.current_image_result)
            return None

        # Pick best candidate
        candidates.sort(key=lambda x: x[3], reverse=True)
        best_cx, best_cy, best_r, best_conf, best_key = candidates[0]
        if best_conf < self.config.CIRCLE_CONFIDENCE_THRESHOLD:
            _log_message(f"Best circle confidence ({best_conf:.2f}) below threshold ({self.config.CIRCLE_CONFIDENCE_THRESHOLD}).", level="WARNING")
            _log_duration("Fiber Center Detection (low confidence)", start, self.current_image_result)
            return None

        _log_message(f"Selected circle at ({best_cx},{best_cy}) radius={best_r}px, confidence={best_conf:.2f}, from '{best_key}'")
        _log_duration("Fiber Center Detection", start, self.current_image_result)
        return (best_cx, best_cy), float(best_r)


    # -------------------------
    # 7) CALCULATE PIXELS_PER_MICRON
    # -------------------------
    def _calculate_pixels_per_micron(self, detected_clad_r_px: float) -> Optional[float]:
        """
        Called only if operating_mode in [MICRON_CALCULATED, MICRON_INFERRED].
        It expects self.fiber_specs.cladding_diameter_um to be set.
        px_per_um = (2 * radius_px) / (cladding_diameter_um).
        """
        start = _start_timer()
        _log_message("Calculating pixels_per_micron ratio...")

        if self.operating_mode not in ["MICRON_CALCULATED", "MICRON_INFERRED"]:
            _log_message("Not in µm conversion mode, skipping calculation.", level="DEBUG")
            _log_duration("Pixels per Micron Calc (skipped)", start, self.current_image_result)
            return None

        if not self.fiber_specs.cladding_diameter_um or self.fiber_specs.cladding_diameter_um <= 0:
            _log_message("No valid cladding_diameter_um, cannot calculate px/µm.", level="WARNING")
            _log_duration("Pixels per Micron Calc (fail)", start, self.current_image_result)
            return None

        if detected_clad_r_px <= 0:
            _log_message("Detected cladding radius invalid (<=0).", level="WARNING")
            _log_duration("Pixels per Micron Calc (fail)", start, self.current_image_result)
            return None

        ppm = (2.0 * detected_clad_r_px) / self.fiber_specs.cladding_diameter_um
        self.pixels_per_micron = ppm
        if self.current_image_result:
            # Store microns_per_pixel = µm/px
            self.current_image_result.stats.microns_per_pixel = (1.0 / ppm) if ppm > 0 else None
        _log_message(f"Calculated px_per_µm = {ppm:.4f} px/µm (i.e. 1/ppm = {1.0/ppm:.4f} µm/px).")
        _log_duration("Pixels per Micron Calc", start, self.current_image_result)
        return ppm


    # -------------------------
    # 8) CREATE ZONE MASKS
    # -------------------------
    def _create_zone_masks(
        self,
        image_shape: Tuple[int,int],
        center_px: Tuple[int,int],
        detected_clad_r_px: float
    ) -> Dict[str, DetectedZoneInfo]:
        """
        For each self.active_zone_definitions, create a binary mask:
         - If PIXEL_ONLY or MICRON_INFERRED: r_min_px = factor * detected_clad_r_px; r_max_px = factor2 * detected_clad_r_px.
         - If MICRON_CALCULATED: r_min_um, r_max_um → r_min_px = r_min_um * px_per_µm, r_max_px likewise.
        Returns a dict: { zone_name: DetectedZoneInfo(...) }.
        """
        start = _start_timer()
        _log_message("Creating zone masks...")

        h, w = image_shape[:2]
        cx, cy = center_px
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2

        zone_info: Dict[str, DetectedZoneInfo] = {}
        for zd in self.active_zone_definitions:
            if self.operating_mode in ["PIXEL_ONLY", "MICRON_INFERRED"]:
                # r_min_px, r_max_px are factors × detected_clad_r_px
                rmin_px = zd.r_min_factor_or_um * detected_clad_r_px
                rmax_px = zd.r_max_factor_or_um * detected_clad_r_px
                rmin_um = None
                rmax_um = None
                if self.operating_mode == "MICRON_INFERRED" and self.pixels_per_micron:
                    rmin_um = rmin_px / self.pixels_per_micron
                    rmax_um = rmax_px / self.pixels_per_micron

            elif self.operating_mode == "MICRON_CALCULATED":
                # r_min_factor_or_um, r_max_factor_or_um are already in µm → convert to px
                if self.pixels_per_micron:
                    rmin_px = zd.r_min_factor_or_um * self.pixels_per_micron
                    rmax_px = zd.r_max_factor_or_um * self.pixels_per_micron
                    rmin_um = zd.r_min_factor_or_um
                    rmax_um = zd.r_max_factor_or_um
                else:
                    _log_message(f"Warning: px/µm not known for zone '{zd.name}'.  Defaulting to px‐factors.", level="WARNING")
                    rmin_px = zd.r_min_factor_or_um
                    rmax_px = zd.r_max_factor_or_um
                    rmin_um = None
                    rmax_um = None
            else:
                # Unrecognized mode: treat factors as px
                rmin_px = zd.r_min_factor_or_um * detected_clad_r_px
                rmax_px = zd.r_max_factor_or_um * detected_clad_r_px
                rmin_um = None
                rmax_um = None

            # Build binary mask
            mask = (((dist_sq >= (rmin_px**2)) & (dist_sq < (rmax_px**2))).astype(np.uint8)) * 255

            zone_info[zd.name] = DetectedZoneInfo(
                name=zd.name,
                center_px=center_px,
                radius_px=rmax_px,
                radius_um=rmax_um,
                mask=mask,
                color_bgr=zd.color_bgr      # <--- Pass the color from ZoneDefinition
            )
            _log_message(f"Zone '{zd.name}': r_min={rmin_px:.1f}px, r_max={rmax_px:.1f}px")

        _log_duration("Zone Mask Creation", start, self.current_image_result)
        return zone_info


    # -------------------------
    # 9) DETECT REGION DEFECTS (DO2MR)
    # -------------------------
    def _detect_region_defects_do2mr(
        self,
        gray: np.ndarray,
        zone_mask: np.ndarray,
        zone_name: str
    ) -> Optional[np.ndarray]:
        """
        Performs DO2MR region‐based detection on 'gray' restricted to 'zone_mask'.  Returns binary mask of defects.
        Steps (for each kernel_size in DO2MR_KERNEL_SIZES):
          1) min_filter (erosion), max_filter (dilation)
          2) residual = max_filtered - min_filtered
          3) median blur residual
          4) threshold = mean(residual[zone_mask>0]) + gamma * std(residual[zone_mask>0])  (for each gamma in DO2MR_GAMMA_VALUES)
          5) binarize, morphological open, accumulate “votes” in a float votemap
        Finally, pixels with ≥ (len(kernel_sizes)*len(gamma_values)/2) votes → final defect mask.
        """
        start = _start_timer()
        _log_message(f"DO2MR detection for zone '{zone_name}'")

        if gray is None or zone_mask is None:
            _log_message("DO2MR: gray or zone_mask is None", level="ERROR")
            return None

        # Restrict to zone
        masked_img = cv2.bitwise_and(gray, gray, mask=zone_mask)
        H, W = gray.shape
        vote_map = np.zeros((H, W), dtype=np.float32)

        total_passes = 0
        for ksz in self.config.DO2MR_KERNEL_SIZES:
            se = cv2.getStructuringElement(cv2.MORPH_RECT, ksz)
            min_f = cv2.erode(masked_img, se)
            max_f = cv2.dilate(masked_img, se)
            residual = cv2.subtract(max_f, min_f)
            if self.config.DO2MR_MEDIAN_BLUR_KERNEL_SIZE > 0:
                residual = cv2.medianBlur(residual, self.config.DO2MR_MEDIAN_BLUR_KERNEL_SIZE)

            for gamma in self.config.DO2MR_GAMMA_VALUES:
                # Compute threshold on residual values inside zone
                vals = residual[zone_mask > 0]
                if vals.size == 0:
                    continue
                m = float(np.mean(vals))
                s = float(np.std(vals))
                thr = np.clip(m + gamma * s, 0, 255)
                _, bin_mask = cv2.threshold(residual, thr, 255, cv2.THRESH_BINARY)
                bin_mask = cv2.bitwise_and(bin_mask, bin_mask, mask=zone_mask)
                # Morphological open to clean
                if self.config.DO2MR_MORPH_OPEN_KERNEL_SIZE[0] > 0:
                    mk = cv2.getStructuringElement(cv2.MORPH_RECT, self.config.DO2MR_MORPH_OPEN_KERNEL_SIZE)
                    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, mk)
                # Accumulate votes
                vote_map += (bin_mask.astype(np.float32) / 255.0)
                total_passes += 1

        if total_passes == 0:
            _log_message(f"DO2MR: No passes executed (zone {zone_name} may be empty)", level="WARNING")
            _log_duration("DO2MR (fail)", start, self.current_image_result)
            return None

        # Final defect if votes >= half of total parameter combinations
        threshold_votes = total_passes / 2.0
        final_mask = (vote_map >= threshold_votes).astype(np.uint8) * 255

        _log_duration("DO2MR Detection", start, self.current_image_result)
        return final_mask


    # -------------------------
    # 10) DETECT SCRATCH DEFECTS (LEI)
    # -------------------------
    def _detect_scratch_defects_lei(
        self,
        gray: np.ndarray,
        zone_mask: np.ndarray,
        zone_name: str
    ) -> Optional[np.ndarray]:
        """
        Performs LEI scratch detection on 'gray' within 'zone_mask'.
        Steps (for each kernel length L in LEI_KERNEL_LENGTHS):
          1) Equalize histogram on gray
          2) For each angle in 0, LEI_ANGLE_STEP, ..., 180:
             a) Create a linear kernel of size (L×1) rotated to that angle
             b) Filter the equalized image
             c) Accumulate in a "max_response" map
          3) Threshold max_response: thr = mean(zone pixels) + LEI_THRESHOLD_FACTOR * std(zone pixels)
          4) Binarize, morphological close (elongated kernel), restrict to zone_mask
        Returns a binary mask of scratch candidates.
        """
        start = _start_timer()
        _log_message(f"LEI scratch detection for zone '{zone_name}'")

        if gray is None or zone_mask is None:
            _log_message("LEI: gray or zone_mask is None", level="ERROR")
            return None

        # Step 1: Equalize grayscale
        eq = cv2.equalizeHist(gray)
        H, W = gray.shape
        max_response = np.zeros((H, W), dtype=np.float32)

        # Build kernels once
        for L in self.config.LEI_KERNEL_LENGTHS:
            # A vertical line of length L (L×1, with center at L//2)
            base_kernel = np.zeros((L, 1), dtype=np.uint8)
            base_kernel[:, 0] = 1
            # Filter at all angles
            for ang in range(0, 180, self.config.LEI_ANGLE_STEP):
                # Rotate kernel
                rotM = cv2.getRotationMatrix2D((0, (L - 1)/2), ang, 1.0)
                # We must place the rotated kernel in a bounding square
                bbox = cv2.warpAffine(base_kernel, rotM, (L, L))
                # Convert to float32 for filtering
                k = bbox.astype(np.float32)
                # Normalize kernel so that sum=1 (avoid scaling issues)
                if np.sum(k) != 0:
                    k = k / float(np.sum(k))
                else:
                    continue

                # Convolve
                try:
                    resp = cv2.filter2D(eq.astype(np.float32), cv2.CV_32F, k, borderType=cv2.BORDER_REPLICATE)
                    # Accumulate the maximum response
                    max_response = np.maximum(max_response, resp)
                except Exception as e:
                    _log_message(f"LEI: filter2D failed for L={L}, ang={ang}: {e}", level="WARNING")
                    continue

        # Step 2: Thresholding on the max_response within zone
        zone_vals = max_response[zone_mask > 0]
        if zone_vals.size == 0:
            _log_message(f"LEI: zone '{zone_name}' empty, skipping", level="WARNING")
            _log_duration("LEI (fail)", start, self.current_image_result)
            return None

        m = float(np.mean(zone_vals))
        s = float(np.std(zone_vals))
        thr = m + self.config.LEI_THRESHOLD_FACTOR * s
        thr = float(np.clip(thr, 0, 255))

        _, bin_scratch = cv2.threshold(max_response, thr, 255, cv2.THRESH_BINARY)
        bin_scratch = bin_scratch.astype(np.uint8)
        # Morphological close (elongated to fill scratch gaps)
        mc_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.config.LEI_MORPH_CLOSE_KERNEL_SIZE)
        bin_scratch = cv2.morphologyEx(bin_scratch, cv2.MORPH_CLOSE, mc_kernel)
        # Restrict to zone
        bin_scratch = cv2.bitwise_and(bin_scratch, bin_scratch, mask=zone_mask)
        # Enforce minimum scratch area
        nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_scratch, connectivity=8)
        final_mask = np.zeros_like(bin_scratch)
        for i in range(1, nb_components):
            area_i = stats[i, cv2.CC_STAT_AREA]
            if area_i >= self.config.LEI_MIN_SCRATCH_AREA_PX:
                final_mask[labels == i] = 255

        _log_duration("LEI Detection", start, self.current_image_result)
        return final_mask


    # -------------------------
    # 11) FALLBACK DEFECT DETECTION (Canny / Adaptive / Otsu)
    # -------------------------
    def _detect_fallback_contours(
        self,
        gray: np.ndarray,
        zone_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        A fallback “generic” detector that uses Canny + dilation or AdaptiveThreshold + contouring
        to catch anything missed by DO2MR/LEI.  Not heavily weighted; primarily to increase recall.
        Returns a binary mask of additional candidates (which will be fused later).
        """
        start = _start_timer()
        _log_message("Fallback defect detection (Canny + Adaptive + Otsu)")

        if gray is None or zone_mask is None:
            _log_message("Fallback: gray or zone_mask is None", level="ERROR")
            return None

        # Canny edges
        edges = cv2.Canny(gray, self.config.CANNY_LOW_THRESHOLD, self.config.CANNY_HIGH_THRESHOLD)
        # Dilate edges to thicken
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dil = cv2.dilate(edges, dil_kernel, iterations=1)
        edges_dil = cv2.bitwise_and(edges_dil, edges_dil, mask=zone_mask)

        # Adaptive threshold
        try:
            adapt = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.config.ADAPTIVE_THRESH_BLOCK_SIZE,
                self.config.ADAPTIVE_THRESH_C
            )
            adapt = cv2.bitwise_and(adapt, adapt, mask=zone_mask)
        except Exception as e:
            _log_message(f"AdaptiveThreshold failed: {e}", level="WARNING")
            adapt = np.zeros_like(gray)

        # Otsu
        try:
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            otsu = cv2.bitwise_and(otsu, otsu, mask=zone_mask)
        except Exception as e:
            _log_message(f"Otsu threshold failed: {e}", level="WARNING")
            otsu = np.zeros_like(gray)

        # Union of all fallback maps
        combined = cv2.bitwise_or(edges_dil, adapt)
        combined = cv2.bitwise_or(combined, otsu)

        _log_duration("Fallback Detection", start, self.current_image_result)
        return combined


    # -------------------------
    # 12) ANALYZE AND CLASSIFY DEFECT CONTOURS
    # -------------------------
    def _analyze_defects(
        self,
        region_mask: Optional[np.ndarray],
        scratch_mask: Optional[np.ndarray],
        fallback_mask: Optional[np.ndarray],
        zone_name: str,
        zone_info: DetectedZoneInfo,
        defect_list: List[DefectInfo]
    ):
        """
        Given three binary masks (region_mask, scratch_mask, fallback_mask) for this zone,
        we:
         1) Combine them into one final “candidate” mask (pixelwise OR).
         2) Find all contours in the final mask.
         3) For each contour:
            - Compute area_px, perimeter_px, boundingBox
            - Determine if it is present in region_mask and/or scratch_mask and/or fallback_mask (vote)
            - Assign defect_type = 'Region' if region_mask, 'Scratch' if scratch_mask, else 'Region' as fallback if fallback
            - Compute centroid_px
            - If self.pixels_per_micron: convert area_px → area_um, etc.
            - Append a new DefectInfo(...) to defect_list
         4) Also tally zone‐specific defect counts (increment in self.current_image_result.stats)
        """
        if self.current_image_result is None:
            return

        combined_mask = np.zeros_like(region_mask if region_mask is not None else scratch_mask)
        if region_mask is not None:
            combined_mask = cv2.bitwise_or(combined_mask, region_mask)
        if scratch_mask is not None:
            combined_mask = cv2.bitwise_or(combined_mask, scratch_mask)
        if fallback_mask is not None:
            combined_mask = cv2.bitwise_or(combined_mask, fallback_mask)

        # Remove small blobs
        nb, lbls, stats, _ = cv2.connectedComponentsWithStats(
            combined_mask, connectivity=8
        )
        final_mask = np.zeros_like(combined_mask)
        for i in range(1, nb):
            area_px = stats[i, cv2.CC_STAT_AREA]
            if area_px >= self.config.MIN_DEFECT_AREA_PX:
                final_mask[lbls == i] = 255

        # Find contours on final_mask
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # Compute area & perimeter
            area_px = cv2.contourArea(cnt)
            if area_px < self.config.MIN_DEFECT_AREA_PX:
                continue  # skip too small

            perimeter_px = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if abs(M["m00"]) < 1e-6:
                cx, cy = x + w//2, y + h//2
            else:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

            # Determine which methods detected it: do2mr, lei, or fallback
            area_mask = np.zeros_like(final_mask)
            cv2.drawContours(area_mask, [cnt], -1, 255, -1)
            methods = []
            if region_mask is not None and np.any(cv2.bitwise_and(area_mask, region_mask)):
                methods.append("do2mr")
            if scratch_mask is not None and np.any(cv2.bitwise_and(area_mask, scratch_mask)):
                methods.append("lei")
            if fallback_mask is not None and np.any(cv2.bitwise_and(area_mask, fallback_mask)):
                methods.append("fallback")

            # If fewer than MIN_METHODS_FOR_CONFIRMED_DEFECT, optionally skip
            if len(methods) < self.config.MIN_METHODS_FOR_CONFIRMED_DEFECT:
                # It’s a low‐confidence detection; but we’ll keep it anyway if fallback found it
                if "fallback" not in methods:
                    continue

            # Assign defect_type
            if "lei" in methods:
                defect_type = "Scratch"
            elif "do2mr" in methods:
                defect_type = "Region"
            else:
                defect_type = "Region"  # fallback

            # Compute area_um, perimeter_um if possible
            if self.pixels_per_micron:
                area_um = area_px / (self.pixels_per_micron ** 2)
                perimeter_um = perimeter_px / self.pixels_per_micron
            else:
                area_um = None
                perimeter_um = None

            # Major/minor dimension: For scratch, major = bounding box max dimension; For region, approximate equivalent diameter
            if defect_type == "Scratch":
                major_dim_px = max(w, h)
                minor_dim_px = min(w, h)
                major_dim_um = major_dim_px / self.pixels_per_micron if self.pixels_per_micron else None
                minor_dim_um = minor_dim_px / self.pixels_per_micron if self.pixels_per_micron else None
            else:
                # Equivalent diameter = sqrt(4*Area/π)
                if area_px > 0:
                    eq_d_px = np.sqrt(4.0 * area_px / np.pi)
                else:
                    eq_d_px = 0.0
                major_dim_px = eq_d_px
                minor_dim_px = eq_d_px
                eq_d_um = eq_d_px / self.pixels_per_micron if self.pixels_per_micron else None
                major_dim_um = eq_d_um
                minor_dim_um = eq_d_um

            # Build DefectInfo
            defect_id = len(defect_list) + 1
            di = DefectInfo(
                defect_id=defect_id,
                zone_name=zone_name,
                defect_type=defect_type,
                centroid_px=(cx, cy),
                bounding_box_px=(x, y, w, h),
                area=DefectMeasurement(value_px=area_px, value_um=area_um),
                perimeter=DefectMeasurement(value_px=perimeter_px, value_um=perimeter_um),
                major_dimension=DefectMeasurement(value_px=major_dim_px, value_um=major_dim_um),
                minor_dimension=DefectMeasurement(value_px=minor_dim_px, value_um=minor_dim_um),
                confidence_score= float(len(methods)) / 3.0,  # naive confidence
                detection_methods=methods,
                contour=cnt
            )
            defect_list.append(di)

            # Increment zone‐specific tally
            if zone_name == "core":
                self.current_image_result.stats.core_defects += 1
            elif zone_name == "cladding":
                self.current_image_result.stats.cladding_defects += 1
            elif zone_name == "ferrule_contact":
                self.current_image_result.stats.ferrule_defects += 1
            elif zone_name == "adhesive":
                self.current_image_result.stats.adhesive_defects += 1

        # Update total defects
        self.current_image_result.stats.total_defects = len(defect_list)
        _log_message(f"Zone '{zone_name}' contributed {len(contours)} candidate contours.")
        return


    # -------------------------
    # 13) ANNOTATE IMAGE
    # -------------------------
    def _annotate_image(self, orig_bgr: np.ndarray, image_res: ImageResult) -> np.ndarray:
        """
        Draws:
         - A large white overlay rectangle in top‐left with status text (filename, status, total defects).
         - Each zone’s circle in its color.
         - Bounding boxes & centroids for each defect (colored by defect_type).
         - Saves the annotated image if configured, and returns it.
        """
        start = _start_timer()
        _log_message(f"Annotating image '{image_res.filename}'...")

        annotated = orig_bgr.copy()
        H, W = annotated.shape[:2]

        # 1) Draw status box
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (W, 60), (0, 0, 0), thickness=-1)  # black background
        alpha = 0.5
        annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)

        cv2.putText(annotated, f"File: {image_res.filename}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, (230, 230, 230), self.config.LINE_THICKNESS)
        cv2.putText(annotated, f"Status: {image_res.stats.status}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, (230, 230, 230), self.config.LINE_THICKNESS)
        cv2.putText(annotated, f"Total Defects: {image_res.stats.total_defects}", (300, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, (230, 230, 230), self.config.LINE_THICKNESS)
        if image_res.stats.microns_per_pixel:
            cv2.putText(annotated, f"µm/px: {image_res.stats.microns_per_pixel:.4f}", (300, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, (230, 230, 230), self.config.LINE_THICKNESS)

        # 2) Draw zone circles
        for z in image_res.detected_zones.values():
            c_x, c_y = z.center_px
            # Outer radius_px as int
            r_px = int(round(z.radius_px))
            cv2.circle(annotated, (c_x, c_y), r_px, z.color_bgr, 1)

        # 3) Draw defects
        for d in image_res.defects:
            x, y, w, h = d.bounding_box_px
            col = self.config.DEFECT_COLORS.get(d.defect_type, (0, 0, 255))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), col, 1)
            cx, cy = d.centroid_px
            cv2.drawMarker(annotated, (cx, cy), col, markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)
            cv2.putText(annotated, f"{d.defect_type}:{d.defect_id}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        _log_duration("Image Annotation", start, self.current_image_result)
        return annotated


    # -------------------------
    # 14) GENERATE POLAR HISTOGRAM
    # -------------------------
    def _generate_defect_histogram(self, image_res: ImageResult) -> Optional[plt.Figure]:
        """
        Creates and saves a polar histogram (_histogram.png) of where all defects lie.
        Radial bin = zone (core/cladding/ferrule/adhesive), angular bin = angle from center→defect centroid.
        Returns the Figure (so it can be saved).
        """
        start = _start_timer()
        _log_message(f"Generating defect histogram for '{image_res.filename}'...")

        if not image_res.defects or "cladding" not in image_res.detected_zones:
            _log_message("No defects or no cladding zone ⇒ skipping histogram", level="WARNING")
            _log_duration("Histogram (skipped)", start, self.current_image_result)
            return None

        center = image_res.detected_zones['cladding'].center_px
        radius = image_res.detected_zones['cladding'].radius_px

        # Build lists
        angles = []
        rads = []
        for d in image_res.defects:
            cx, cy = d.centroid_px
            dx = cx - center[0]
            dy = cy - center[1]
            angle = np.arctan2(dy, dx)
            if angle < 0:
                angle += 2 * np.pi
            dist = np.sqrt(dx * dx + dy * dy)
            angles.append(angle)
            rads.append(dist)

        # Create polar plot
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, polar=True)
        ax.scatter(angles, rads, c='magenta', s=10, alpha=0.75)

        # Configure radial ticks at each zone radius
        zone_radii = []
        zone_labels = []
        for zname, zinfo in image_res.detected_zones.items():
            zone_radii.append(zinfo.radius_px)
            zone_labels.append(zname)
        zone_radii, zone_labels = zip(*sorted(zip(zone_radii, zone_labels)))
        ax.set_rgrids(zone_radii, labels=zone_labels, angle=22.5)  # place labels at 22.5°

        ax.set_title(f"Defect Distribution: {image_res.filename}", va='bottom')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        _log_duration("Histogram Generation", start, self.current_image_result)
        return fig


    # -------------------------
    # 15) SAVE CSV REPORT (ONE PER IMAGE)
    # -------------------------
    def _save_report_csv(self, image_res: ImageResult):
        """
        Saves a detailed CSV (filename_report.csv) listing every defect’s properties:
         defect_id, zone, type, centroid_x, centroid_y, bbox_x, bbox_y, bbox_w, bbox_h,
         area_px, area_um, perim_px, perim_um, major_px, major_um, minor_px, minor_um, confidence, methods.
        """
        start = _start_timer()
        _log_message(f"Saving detailed CSV for '{image_res.filename}' ...")

        if not self.config.DETAILED_REPORT_PER_IMAGE:
            _log_duration("Save CSV (skipped)", start, self.current_image_result)
            return

        out_csv = Path(image_res.filename).stem + "_report.csv"
        out_path = self.current_image_result.report_csv_path = self.output_dir_path / out_csv
        with open(out_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            header = [
                "defect_id",
                "zone_name",
                "defect_type",
                "centroid_x_px", "centroid_y_px",
                "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px",
                "area_px", "area_um",
                "perimeter_px", "perimeter_um",
                "major_px", "major_um",
                "minor_px", "minor_um",
                "confidence_score",
                "detection_methods"
            ]
            writer.writerow(header)
            for d in image_res.defects:
                writer.writerow([
                    d.defect_id,
                    d.zone_name,
                    d.defect_type,
                    d.centroid_px[0], d.centroid_px[1],
                    d.bounding_box_px[0], d.bounding_box_px[1],
                    d.bounding_box_px[2], d.bounding_box_px[3],
                    d.area.value_px, d.area.value_um,
                    d.perimeter.value_px, d.perimeter.value_um,
                    d.major_dimension.value_px, d.major_dimension.value_um,
                    d.minor_dimension.value_px, d.minor_dimension.value_um,
                    f"{d.confidence_score:.2f}",
                    "|".join(d.detection_methods)
                ])

        _log_duration("Save CSV", start, self.current_image_result)


    # -------------------------
    # 16) PROCESS A SINGLE IMAGE (full pipeline)
    # -------------------------
    def process_single_image(self, image_path: Path):
        """
        Executes the full pipeline on a single image:
         1) Create a new ImageResult
         2) Load image
         3) Preprocess → Hough → zone detection → pixel/µm conversion → zone masks
         4) For each zone: DO2MR, LEI, fallback, then analyze contours & append to defects
         5) Annotate image, save, histogram, save CSV, update stats
        """
        script_start = _start_timer()

        # 1) Setup image result
        image_res = ImageResult(
            filename=image_path.name,
            timestamp=datetime.now(),
            fiber_specs_used=self.fiber_specs,
            operating_mode=self.operating_mode
        )
        self.current_image_result = image_res

        try:
            # 2) Load
            orig_bgr = self._load_single_image(image_path)
            if orig_bgr is None:
                raise ValueError("Could not load image.")

            # 3) Preprocess
            processed = self._preprocess_image(orig_bgr)
            fc = self._find_fiber_center_and_radius(processed)
            if fc is None:
                raise ValueError("Fiber cladding circle not detected.")

            center_px, clad_r_px = fc
            
            # 4) If MICRO_CALCULATED or INFERRED, compute px/um
            if self.operating_mode in ["MICRON_CALCULATED", "MICRON_INFERRED"]:
                self._calculate_pixels_per_micron(detected_clad_r_px=clad_r_px)

            # 5) Create all zone masks
            all_zones = self._create_zone_masks(
                image_shape=orig_bgr.shape,
                center_px=center_px,
                detected_clad_r_px=clad_r_px
            )
            image_res.detected_zones = all_zones

            # 6) For each zone, run detectors and analyze contours
            gray_full = processed['original_gray']
            for zname, zinfo in all_zones.items():
                zone_mask = zinfo.mask
                if zone_mask is None:
                    continue

                region_mask = self._detect_region_defects_do2mr(gray_full, zone_mask, zname)
                scratch_mask = self._detect_scratch_defects_lei(gray_full, zone_mask, zname)
                fallback_mask = self._detect_fallback_contours(gray_full, zone_mask)

                # Optionally save intermediate masks for debugging
                if self.config.SAVE_DEFECT_MAPS:
                    image_res.intermediate_defect_maps[f"{zname}_region"] = region_mask
                    image_res.intermediate_defect_maps[f"{zname}_scratch"] = scratch_mask
                    image_res.intermediate_defect_maps[f"{zname}_fallback"] = fallback_mask

                # Analyze contours / classification
                self._analyze_defects(
                    region_mask=region_mask,
                    scratch_mask=scratch_mask,
                    fallback_mask=fallback_mask,
                    zone_name=zname,
                    zone_info=zinfo,
                    defect_list=image_res.defects
                )

            # 7) Set pass/fail status (example: if any adhesive defects, fail)
            if image_res.stats.adhesive_defects > 0:
                image_res.stats.status = "FAIL"
            else:
                image_res.stats.status = "PASS"

            # 8) Annotate and save image
            if self.config.SAVE_ANNOTATED_IMAGE:
                annotated = self._annotate_image(orig_bgr, image_res)
                out_annot = Path(image_path.stem + "_annotated.jpg")
                image_res.annotated_image_path = self.output_dir_path / out_annot
                cv2.imwrite(str(image_res.annotated_image_path), annotated)
                _log_message(f"Annotated image saved: {image_res.annotated_image_path}")

            # 9) Generate & save polar histogram
            if self.config.SAVE_HISTOGRAM:
                fig = self._generate_defect_histogram(image_res)
                if fig is not None:
                    out_hist = Path(image_path.stem + "_histogram.png")
                    image_res.histogram_path = self.output_dir_path / out_hist
                    fig.savefig(str(image_res.histogram_path), dpi=150)
                    plt.close(fig)
                    _log_message(f"Histogram saved: {image_res.histogram_path}")

            # 10) Save detailed CSV
            self._save_report_csv(image_res)

            # 11) Build summary dictionary
            summary = {
                "filename": image_res.filename,
                "total_defects": image_res.stats.total_defects,
                "core_defects": image_res.stats.core_defects,
                "cladding_defects": image_res.stats.cladding_defects,
                "ferrule_defects": image_res.stats.ferrule_defects,
                "adhesive_defects": image_res.stats.adhesive_defects,
                "status": image_res.stats.status,
                "processing_time_s": round(time.perf_counter() - script_start, 3),
                "µm_per_px": image_res.stats.microns_per_pixel
            }
            self.batch_results_summary_list.append(summary)

        except Exception as err:
            # Record any error in image_res and continue to next
            _log_message(f"Error processing '{image_path.name}': {err}", level="ERROR")
            import traceback; traceback.print_exc()
            image_res.error_message = str(err)
            image_res.stats.status = "ERROR"
            # Even on error, append summary row (with zeros)
            summary = {
                "filename": image_res.filename,
                "total_defects": 0,
                "core_defects": 0,
                "cladding_defects": 0,
                "ferrule_defects": 0,
                "adhesive_defects": 0,
                "status": image_res.stats.status,
                "processing_time_s": round(time.perf_counter() - script_start, 3),
                "µm_per_px": image_res.stats.microns_per_pixel
            }
            self.batch_results_summary_list.append(summary)

        finally:
            # Clear for next image
            self.current_image_result = None

        return


    # -------------------------
    # 17) PROCESS A BATCH OF IMAGES
    # -------------------------
    def process_image_batch(self, image_paths: List[Path]):
        """
        Orchestrates the entire batch:
          - Creates a timestamped subfolder inside OUTPUT_DIR_NAME
          - Loops over all image_paths, calling process_single_image(...)
          - At the end, writes batch_inspection_summary.csv
        """
        start = _start_timer()
        _log_message("Starting batch processing...")

        # Make a unique batch subfolder
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = self.output_dir_path / f"batch_{ts}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir_path = batch_dir  # redirect all outputs here

        for idx, imgp in enumerate(image_paths, start=1):
            _log_message(f"\n---- Processing image {idx}/{len(image_paths)}: '{imgp.name}' ----")
            self.process_single_image(imgp)

        # After all images, save batch summary CSV
        if self.batch_results_summary_list:
            df_summary = pd.DataFrame(self.batch_results_summary_list)
            batch_csv = self.output_dir_path / self.config.BATCH_SUMMARY_FILENAME
            df_summary.to_csv(batch_csv, index=False)
            _log_message(f"Batch summary saved: {batch_csv}")

        _log_duration("Batch Processing Complete", start)
        print("\n" + "=" * 70)
        print(f"Batch processing complete. Results in: {self.output_dir_path}")
        print("=" * 70 + "\n")
        return


# ------------------------------------------------------------------------------
# Part 3: Script Entry‐Point (main)
# ------------------------------------------------------------------------------

def main():
    """
    The script entry‐point.  It:
     1) Prints a brief header
     2) Instantiates FiberInspector
     3) Gets user specs (pixel‐only vs micron mode)
     4) Gets image paths from user
     5) Runs process_image_batch(...)
     6) Catches any top‐level exceptions
    """
    print("\n" + "="*30 + " Advanced Fiber Inspector " + "="*30 + "\n")
    _log_message("Starting advanced_fiber_inspector.py ...")

    script_start = _start_timer()
    try:
        inspector = FiberInspector()
        inspector._get_user_specifications()
        image_paths = inspector._get_image_paths_from_user()
        if not image_paths:
            _log_message("No images provided; exiting.", level="INFO")
            return
        inspector.process_image_batch(image_paths)

    except FileNotFoundError as fnf:
        _log_message(f"FileNotFoundError: {fnf}", level="CRITICAL")
    except ValueError as ve:
        _log_message(f"ValueError: {ve}", level="CRITICAL")
    except Exception as e:
        _log_message(f"Unexpected error: {e}", level="CRITICAL")
        import traceback; traceback.print_exc()
    finally:
        _log_duration("Total Script Execution", script_start)
        print("="*80)
        print("Inspection Run Finished.")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()