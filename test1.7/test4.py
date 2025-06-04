import cv2
import numpy as np
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import glob
import os
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
import warnings
warnings.filterwarnings('ignore')

class EnhancedFiberOpticInspector:
    """
    Enhanced Automated Fiber Optic End Face Defect Detection System
    Implements multiple OpenCV techniques for improved accuracy
    Based on DO2MR and LEI methods with additional enhancements
    """

    def __init__(self, 
                 calibration_file: str = "calibration.json",
                 core_diameter_um: Optional[float] = None,
                 cladding_diameter_um: Optional[float] = None,
                 ferrule_outer_diameter_um: Optional[float] = 250.0,
                 use_pixel_units: bool = False):
        """Initialize the inspector with calibration data and optional fiber dimensions."""
        
        self.use_pixel_units = use_pixel_units
        self.um_per_px = None
        self.calibration_file = calibration_file
        
        # Load calibration if not using pixel units
        if not self.use_pixel_units:
            try:
                self.calibration = self._load_calibration(calibration_file)
                self.um_per_px = self.calibration.get("um_per_px", None)
            except:
                print(f"Warning: Could not load calibration from {calibration_file}. Using pixel units.")
                self.use_pixel_units = True
        
        # Store fiber dimensions
        self.core_diameter_um = core_diameter_um
        self.cladding_diameter_um = cladding_diameter_um
        self.ferrule_outer_diameter_um = ferrule_outer_diameter_um
        
        # Zone definitions based on IEC 61300-3-35
        self._setup_zones()
        
        # Detection parameters (enhanced from paper)
        self.do2mr_params = {
            "kernel_sizes": [(5, 5), (9, 9), (15, 15)],  # Multiple kernel sizes
            "gamma_values": [2.0, 2.5, 3.0, 3.5],  # Multiple gamma values
            "min_area_px": 10,
            "morph_iterations": 2
        }
        
        self.lei_params = {
            "kernel_lengths": [9, 15, 21],  # Multiple lengths
            "angles": np.arange(0, 180, 5),  # Higher angular resolution
            "threshold_methods": ['otsu', 'adaptive', 'sigma'],  # Multiple thresholding
            "min_scratch_area": 10
        }
        
        # Hough circle detection parameters (enhanced)
        self.hough_params = {
            "dp_values": [1.0, 1.2, 1.5],
            "param1_values": [50, 70, 100],
            "param2_values": [30, 40, 50],
            "min_dist_factor": 0.125,
            "min_radius_factor": 0.1,
            "max_radius_factor": 0.5
        }
        
        # Additional processing methods
        self.processing_methods = {
            "canny": {"low": [50, 100], "high": [100, 200]},
            "gradient": {"ksize": [3, 5, 7]},
            "morphology": {"operations": ["gradient", "tophat", "blackhat"]}
        }

    def _load_calibration(self, filepath: str) -> Dict:
        """Load calibration data from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _setup_zones(self):
        """Setup zone definitions based on fiber dimensions"""
        if self.core_diameter_um and self.cladding_diameter_um:
            core_radius = self.core_diameter_um / 2.0
            cladding_radius = self.cladding_diameter_um / 2.0
            ferrule_radius = self.ferrule_outer_diameter_um / 2.0
        else:
            # Default values for typical single-mode fiber (9/125/250)
            core_radius = 4.5
            cladding_radius = 62.5
            ferrule_radius = 125.0
        
        self.zones_um = {
            "core": {
                "r_min": 0, 
                "r_max": core_radius, 
                "max_defect_um": 3,
                "defects_allowed": True,
                "criticality": "critical"
            },
            "cladding": {
                "r_min": core_radius, 
                "r_max": cladding_radius, 
                "max_defect_um": 10,
                "defects_allowed": True,
                "criticality": "high"
            },
            "ferrule_contact": {
                "r_min": cladding_radius, 
                "r_max": ferrule_radius, 
                "max_defect_um": 25,
                "defects_allowed": True,
                "criticality": "medium"
            },
            "adhesive": {
                "r_min": ferrule_radius, 
                "r_max": ferrule_radius + 15, 
                "max_defect_um": 50,
                "defects_allowed": True,
                "criticality": "low"
            }
        }

    def preprocess_image_multi(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply multiple preprocessing techniques"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        preprocessed = {}
        
        # 1. Gaussian blur
        preprocessed['gaussian'] = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Bilateral filter (edge-preserving)
        preprocessed['bilateral'] = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 3. Median filter (good for salt-and-pepper noise)
        preprocessed['median'] = cv2.medianBlur(gray, 5)
        
        # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        preprocessed['clahe'] = clahe.apply(gray)
        
        # 5. Histogram equalization
        preprocessed['hist_eq'] = cv2.equalizeHist(gray)
        
        # 6. Morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        preprocessed['morph_grad'] = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # 7. Top hat (bright features on dark background)
        preprocessed['tophat'] = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # 8. Black hat (dark features on bright background)
        preprocessed['blackhat'] = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        return preprocessed

    def find_fiber_center_enhanced(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """Enhanced fiber center detection using multiple methods"""
        
        # Try multiple preprocessing methods
        preprocessed = self.preprocess_image_multi(image)
        
        best_circle = None
        best_score = 0
        
        # Method 1: Hough Circle Transform with multiple parameters
        for prep_name, prep_img in preprocessed.items():
            if prep_name in ['morph_grad', 'tophat', 'blackhat']:
                continue  # Skip these for Hough
                
            edges = cv2.Canny(prep_img, 50, 150)
            
            for dp in self.hough_params['dp_values']:
                for param1 in self.hough_params['param1_values']:
                    for param2 in self.hough_params['param2_values']:
                        circles = cv2.HoughCircles(
                            edges,
                            cv2.HOUGH_GRADIENT,
                            dp=dp,
                            minDist=int(image.shape[0] * self.hough_params['min_dist_factor']),
                            param1=param1,
                            param2=param2,
                            minRadius=int(image.shape[0] * self.hough_params['min_radius_factor']),
                            maxRadius=int(image.shape[0] * self.hough_params['max_radius_factor'])
                        )
                        
                        if circles is not None:
                            circles = np.uint16(np.around(circles[0, :]))
                            for circle in circles:
                                score = self._score_circle(prep_img, circle)
                                if score > best_score:
                                    best_score = score
                                    best_circle = circle
        
        # Method 2: Contour-based detection as fallback
        if best_circle is None:
            best_circle = self._find_fiber_by_contours(image)
        
        if best_circle is not None:
            center = (int(best_circle[0]), int(best_circle[1]))
            radius = float(best_circle[2])
            return center, radius
        
        # Fallback to image center
        print("Warning: Could not detect fiber accurately. Using image center.")
        return (image.shape[1]//2, image.shape[0]//2), min(image.shape[0], image.shape[1])//4

    def _score_circle(self, image: np.ndarray, circle: np.ndarray) -> float:
        """Score a detected circle based on edge strength and circularity"""
        cx, cy, r = circle
        
        # Create circle mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, 2)
        
        # Calculate edge strength along circle
        edges = cv2.Canny(image, 50, 150)
        edge_strength = np.sum(edges & mask) / (2 * np.pi * r)
        
        # Check circularity by sampling points
        angles = np.linspace(0, 2*np.pi, 36)
        intensities = []
        for angle in angles:
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                intensities.append(image[y, x])
        
        # Lower std means more uniform circle
        intensity_std = np.std(intensities) if intensities else float('inf')
        
        # Combined score (higher is better)
        score = edge_strength / (1 + intensity_std)
        return score

    def _find_fiber_by_contours(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Find fiber using contour detection"""
        # Threshold image
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest circular contour
        best_contour = None
        best_circularity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip small contours
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > best_circularity and circularity > 0.6:
                best_circularity = circularity
                best_contour = contour
        
        if best_contour is not None:
            # Fit circle to contour
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            return np.array([x, y, radius])
        
        return None

    def detect_region_defects_enhanced(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Enhanced DO2MR with multiple kernel sizes and parameters"""
        
        all_masks = []
        
        # Try multiple kernel sizes
        for kernel_size in self.do2mr_params["kernel_sizes"]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            
            # Apply min-max filtering
            img_max = cv2.dilate(image, kernel)
            img_min = cv2.erode(image, kernel)
            residual = cv2.absdiff(img_max, img_min)
            
            # Apply median filter
            residual_filtered = cv2.medianBlur(residual, 5)
            
            # Try multiple gamma values for threshold
            for gamma in self.do2mr_params["gamma_values"]:
                mean_val = np.mean(residual_filtered)
                std_val = np.std(residual_filtered)
                threshold_val = mean_val + gamma * std_val
                
                _, binary_mask = cv2.threshold(residual_filtered, threshold_val, 255, cv2.THRESH_BINARY)
                
                # Morphological operations
                kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_morph, 
                                              iterations=self.do2mr_params["morph_iterations"])
                
                all_masks.append(binary_mask)
        
        # Combine masks using voting
        combined_mask = self._combine_masks_voting(all_masks)
        
        # Connected components
        n_labels, labeled = cv2.connectedComponents(combined_mask)
        
        return combined_mask, all_masks

    def detect_scratches_enhanced(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Enhanced LEI with multiple parameters and methods"""
        
        all_scratch_masks = []
        
        # Try multiple kernel lengths
        for kernel_length in self.lei_params["kernel_lengths"]:
            scratch_strength = np.zeros_like(image, dtype=np.float32)
            
            for angle in self.lei_params["angles"]:
                angle_rad = np.deg2rad(angle)
                
                # Create line kernel
                kernel_points = []
                for i in range(-kernel_length//2, kernel_length//2 + 1):
                    x = int(round(i * np.cos(angle_rad)))
                    y = int(round(i * np.sin(angle_rad)))
                    if (x, y) != (0, 0):
                        kernel_points.append((x, y))
                
                # Apply linear detector
                response = self._apply_linear_detector_enhanced(image, kernel_points)
                scratch_strength = np.maximum(scratch_strength, response)
            
            # Normalize scratch strength
            if scratch_strength.max() > 0:
                scratch_strength_norm = cv2.normalize(scratch_strength, None, 0, 255, 
                                                     cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                scratch_strength_norm = scratch_strength.astype(np.uint8)
            
            # Try multiple thresholding methods
            for method in self.lei_params["threshold_methods"]:
                if method == 'otsu':
                    _, mask = cv2.threshold(scratch_strength_norm, 0, 255, 
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                elif method == 'adaptive':
                    mask = cv2.adaptiveThreshold(scratch_strength_norm, 255, 
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
                elif method == 'sigma':
                    mean = np.mean(scratch_strength_norm)
                    std = np.std(scratch_strength_norm)
                    _, mask = cv2.threshold(scratch_strength_norm, mean + 2.5 * std, 255, 
                                          cv2.THRESH_BINARY)
                
                # Morphological refinement
                kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length//3, 1))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
                
                all_scratch_masks.append(mask)
        
        # Combine masks
        combined_mask = self._combine_masks_voting(all_scratch_masks, threshold=0.3)
        
        # Final refinement
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_final)
        
        return combined_mask, all_scratch_masks

    def _apply_linear_detector_enhanced(self, image: np.ndarray, kernel_points: List[Tuple[int, int]]) -> np.ndarray:
        """Enhanced linear detector with better edge handling"""
        height, width = image.shape
        response = np.zeros_like(image, dtype=np.float32)
        
        # Pad image
        max_offset = max(max(abs(dx), abs(dy)) for dx, dy in kernel_points) if kernel_points else 0
        padded = cv2.copyMakeBorder(image, max_offset, max_offset, max_offset, max_offset, 
                                    cv2.BORDER_REFLECT)
        
        # Apply detector
        for r in range(height):
            for c in range(width):
                # Get line pixels
                line_pixels = []
                for dx, dy in kernel_points:
                    px = padded[r + max_offset + dy, c + max_offset + dx]
                    line_pixels.append(float(px))
                
                if line_pixels:
                    # Center vs. surround analysis
                    center_idx = len(line_pixels) // 2
                    center_region = line_pixels[center_idx-2:center_idx+3] if len(line_pixels) > 5 else line_pixels
                    
                    if len(line_pixels) > 5:
                        surround = line_pixels[:center_idx-2] + line_pixels[center_idx+3:]
                        if surround:
                            # Response based on contrast
                            response[r, c] = max(0, abs(np.mean(center_region) - np.mean(surround)))
                    else:
                        # For short lines, use variance
                        response[r, c] = np.var(line_pixels)
        
        return response

    def _combine_masks_voting(self, masks: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Combine multiple masks using voting"""
        if not masks:
            return np.zeros((1, 1), dtype=np.uint8)
        
        # Convert to float for averaging
        masks_float = [mask.astype(np.float32) / 255.0 for mask in masks]
        
        # Average
        avg_mask = np.mean(masks_float, axis=0)
        
        # Threshold
        combined = (avg_mask >= threshold).astype(np.uint8) * 255
        
        return combined

    def create_zone_masks(self, image_shape: Tuple, center: Tuple[int, int], 
                         radius_px: Optional[float]) -> Dict[str, np.ndarray]:
        """Create masks for different zones"""
        masks = {}
        height, width = image_shape[:2]
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        
        # Determine scaling
        if self.use_pixel_units:
            # Use detected radius to scale zones proportionally
            if radius_px is None:
                radius_px = min(height, width) / 4  # Fallback
            
            # Assume detected radius is cladding radius
            cladding_radius_px = radius_px
            
            # Scale other zones based on typical fiber ratios
            if self.core_diameter_um and self.cladding_diameter_um:
                ratio = self.core_diameter_um / self.cladding_diameter_um
            else:
                ratio = 9.0 / 125.0  # Typical SMF ratio
            
            core_radius_px = cladding_radius_px * ratio
            ferrule_radius_px = cladding_radius_px * (250.0 / 125.0)  # Typical ratio
            adhesive_radius_px = ferrule_radius_px + 30  # Arbitrary extension
            
            zone_radii = {
                "core": (0, core_radius_px),
                "cladding": (core_radius_px, cladding_radius_px),
                "ferrule_contact": (cladding_radius_px, ferrule_radius_px),
                "adhesive": (ferrule_radius_px, adhesive_radius_px)
            }
        else:
            # Use micron values with um_per_px
            if self.um_per_px is None:
                print("Warning: No calibration available. Using pixel units.")
                return self.create_zone_masks(image_shape, center, radius_px)
            
            zone_radii = {}
            for zone_name, zone_params in self.zones_um.items():
                r_min_px = zone_params["r_min"] / self.um_per_px
                r_max_px = zone_params["r_max"] / self.um_per_px
                zone_radii[zone_name] = (r_min_px, r_max_px)
        
        # Create masks
        for zone_name, (r_min, r_max) in zone_radii.items():
            mask = ((dist_from_center >= r_min) & (dist_from_center < r_max)).astype(np.uint8) * 255
            masks[zone_name] = mask
        
        return masks

    def classify_defects_enhanced(self, region_masks: List[np.ndarray], scratch_masks: List[np.ndarray],
                                 zone_masks: Dict[str, np.ndarray], 
                                 combined_region_mask: np.ndarray,
                                 combined_scratch_mask: np.ndarray) -> pd.DataFrame:
        """Enhanced defect classification with multiple features"""
        defects = []
        
        # Process region defects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_region_mask, connectivity=8
        )
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.do2mr_params["min_area_px"]:
                continue
            
            defect_info = self._analyze_defect(
                labels == i, stats[i], centroids[i], zone_masks, "region"
            )
            
            if defect_info:
                defects.append(defect_info)
        
        # Process scratch defects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_scratch_mask, connectivity=8
        )
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.lei_params["min_scratch_area"]:
                continue
            
            defect_info = self._analyze_defect(
                labels == i, stats[i], centroids[i], zone_masks, "scratch"
            )
            
            if defect_info:
                defects.append(defect_info)
        
        return pd.DataFrame(defects)

    def _analyze_defect(self, defect_mask: np.ndarray, stats: np.ndarray, 
                       centroid: np.ndarray, zone_masks: Dict[str, np.ndarray], 
                       defect_type: str) -> Optional[Dict]:
        """Analyze individual defect"""
        cx, cy = int(centroid[0]), int(centroid[1])
        
        # Determine zone
        zone = "unknown"
        for zone_name, mask in zone_masks.items():
            if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                if mask[cy, cx] > 0:
                    zone = zone_name
                    break
        
        # Extract features
        area_px = stats[cv2.CC_STAT_AREA]
        x, y, w, h = stats[cv2.CC_STAT_LEFT], stats[cv2.CC_STAT_TOP], \
                     stats[cv2.CC_STAT_WIDTH], stats[cv2.CC_STAT_HEIGHT]
        
        # Calculate size in appropriate units
        if self.use_pixel_units or self.um_per_px is None:
            size_unit = "px"
            if defect_type == "scratch":
                # Find oriented bounding box for scratches
                contours, _ = cv2.findContours(defect_mask.astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    rect = cv2.minAreaRect(contours[0])
                    box_dims = sorted(rect[1])
                    width_val = box_dims[0]
                    length_val = box_dims[1]
                else:
                    width_val = min(w, h)
                    length_val = max(w, h)
            else:
                # For region defects, use equivalent diameter
                size_val = np.sqrt(4 * area_px / np.pi)
        else:
            size_unit = "um"
            if defect_type == "scratch":
                contours, _ = cv2.findContours(defect_mask.astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    rect = cv2.minAreaRect(contours[0])
                    box_dims = sorted(rect[1])
                    width_val = box_dims[0] * self.um_per_px
                    length_val = box_dims[1] * self.um_per_px
                else:
                    width_val = min(w, h) * self.um_per_px
                    length_val = max(w, h) * self.um_per_px
            else:
                area_um2 = area_px * (self.um_per_px ** 2)
                size_val = np.sqrt(4 * area_um2 / np.pi)
        
        # Create defect info
        defect_info = {
            "type": defect_type,
            "zone": zone,
            "area_px": area_px,
            "centroid_x": cx,
            "centroid_y": cy,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h
        }
        
        if defect_type == "scratch":
            defect_info[f"length_{size_unit}"] = round(length_val, 2)
            defect_info[f"width_{size_unit}"] = round(width_val, 2)
        else:
            defect_info[f"size_{size_unit}"] = round(size_val, 2)
        
        # Add shape features
        defect_info["aspect_ratio"] = round(w / h if h > 0 else 1, 2)
        
        # Add defect subtype classification
        if defect_type == "region":
            if defect_info["aspect_ratio"] > 3 or defect_info["aspect_ratio"] < 0.33:
                defect_info["subtype"] = "elongated"
            else:
                defect_info["subtype"] = "dig"
        else:
            defect_info["subtype"] = "linear"
        
        return defect_info

    def apply_pass_fail_criteria(self, defects_df: pd.DataFrame) -> Tuple[str, List[str]]:
        """Apply IEC 61300-3-35 pass/fail criteria"""
        if defects_df.empty:
            return "PASS", []
        
        status = "PASS"
        failure_reasons = []
        
        # Check each zone
        for zone_name, zone_criteria in self.zones_um.items():
            zone_defects = defects_df[defects_df["zone"] == zone_name]
            
            if zone_defects.empty:
                continue
            
            # Check if defects are allowed
            if not zone_criteria.get("defects_allowed", True):
                status = "FAIL"
                failure_reasons.append(f"{zone_name}: No defects allowed")
                continue
            
            # Check defect sizes
            max_allowed_size = zone_criteria.get("max_defect_um", float('inf'))
            
            for _, defect in zone_defects.iterrows():
                # Get defect size in microns
                if self.use_pixel_units and self.um_per_px is None:
                    # Can't check micron-based criteria
                    continue
                
                if defect["type"] == "scratch":
                    size_cols = [col for col in defect.index if col.startswith("length_")]
                    if size_cols:
                        defect_size = defect[size_cols[0]]
                        if "px" in size_cols[0] and self.um_per_px:
                            defect_size *= self.um_per_px
                    else:
                        continue
                else:
                    size_cols = [col for col in defect.index if col.startswith("size_")]
                    if size_cols:
                        defect_size = defect[size_cols[0]]
                        if "px" in size_cols[0] and self.um_per_px:
                            defect_size *= self.um_per_px
                    else:
                        continue
                
                if defect_size > max_allowed_size:
                    status = "FAIL"
                    failure_reasons.append(
                        f"{zone_name}: {defect['type']} exceeds size limit "
                        f"({defect_size:.1f}µm > {max_allowed_size}µm)"
                    )
        
        return status, list(set(failure_reasons))

    def inspect_fiber(self, image_path: str) -> Dict:
        """Main inspection function"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                "status": "ERROR",
                "error": f"Could not load image: {image_path}",
                "defect_count": 0,
                "defects": []
            }
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Find fiber center
        center, radius = self.find_fiber_center_enhanced(gray)
        
        if center is None:
            return {
                "status": "ERROR",
                "error": "Could not detect fiber center",
                "defect_count": 0,
                "defects": []
            }
        
        # Create zone masks
        zone_masks = self.create_zone_masks(gray.shape, center, radius)
        
        # Apply multiple preprocessing methods
        preprocessed_images = self.preprocess_image_multi(gray)
        
        # Detect defects using multiple methods
        all_region_masks = []
        all_scratch_masks = []
        
        # Apply DO2MR on different preprocessed images
        for prep_name, prep_img in preprocessed_images.items():
            if prep_name in ['morph_grad', 'tophat', 'blackhat']:
                # These are already defect-enhanced
                region_mask, masks = self.detect_region_defects_enhanced(prep_img)
                all_region_masks.extend(masks)
        
        # Apply LEI on original and some preprocessed images
        for prep_name in ['gaussian', 'bilateral', 'median']:
            if prep_name in preprocessed_images:
                scratch_mask, masks = self.detect_scratches_enhanced(preprocessed_images[prep_name])
                all_scratch_masks.extend(masks)
        
        # Combine all masks
        if all_region_masks:
            combined_region_mask = self._combine_masks_voting(all_region_masks, threshold=0.3)
        else:
            combined_region_mask = np.zeros_like(gray)
        
        if all_scratch_masks:
            combined_scratch_mask = self._combine_masks_voting(all_scratch_masks, threshold=0.3)
        else:
            combined_scratch_mask = np.zeros_like(gray)
        
        # Classify defects
        defects_df = self.classify_defects_enhanced(
            all_region_masks, all_scratch_masks, zone_masks,
            combined_region_mask, combined_scratch_mask
        )
        
        # Apply pass/fail criteria
        status, failure_reasons = self.apply_pass_fail_criteria(defects_df)
        
        # Prepare results
        results = {
            "image_path": image_path,
            "status": status,
            "failure_reasons": failure_reasons,
            "defect_count": len(defects_df),
            "defects": defects_df.to_dict('records') if not defects_df.empty else [],
            "fiber_center": center,
            "fiber_radius": radius,
            "processing_info": {
                "um_per_px": self.um_per_px if self.um_per_px else "N/A",
                "preprocessing_methods": len(preprocessed_images),
                "region_masks_generated": len(all_region_masks),
                "scratch_masks_generated": len(all_scratch_masks)
            },
            "masks": {
                "region": combined_region_mask,
                "scratch": combined_scratch_mask,
                "zones": zone_masks
            }
        }
        
        return results

    def visualize_results(self, image_path: str, results: Dict, save_path: Optional[str] = None):
        """Visualize inspection results"""
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image for visualization: {image_path}")
            return
        
        # Create visualization
        vis_image = image.copy()
        
        # Draw fiber center and zones
        center = results.get("fiber_center")
        radius = results.get("fiber_radius")
        
        if center and radius:
            # Draw detected fiber circle
            cv2.circle(vis_image, center, int(radius), (0, 255, 0), 2)
            cv2.circle(vis_image, center, 3, (0, 255, 0), -1)
            
            # Draw zone boundaries
            zone_colors = {
                "core": (255, 0, 0),       # Blue
                "cladding": (0, 255, 0),   # Green
                "ferrule_contact": (0, 0, 255),  # Red
                "adhesive": (255, 255, 0)  # Cyan
            }
            
            # Draw zones
            for zone_name, mask in results.get("masks", {}).get("zones", {}).items():
                if mask is None:
                    continue
                # Find contours of the zone
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color = zone_colors.get(zone_name, (128, 128, 128))
                cv2.drawContours(vis_image, contours, -1, color, 1)
        
        # Overlay defects
        region_mask = results.get("masks", {}).get("region")
        scratch_mask = results.get("masks", {}).get("scratch")
        
        # Create overlay for defects
        overlay = vis_image.copy()
        
        if region_mask is not None:
            # Yellow overlay for region defects
            overlay[region_mask > 0] = [0, 255, 255]
        
        if scratch_mask is not None:
            # Magenta overlay for scratches
            overlay[scratch_mask > 0] = [255, 0, 255]
        
        # Blend overlay with original
        alpha = 0.3
        vis_image = cv2.addWeighted(vis_image, 1-alpha, overlay, alpha, 0)
        
        # Draw defect bounding boxes
        for defect in results.get("defects", []):
            x = defect.get("bbox_x", 0)
            y = defect.get("bbox_y", 0)
            w = defect.get("bbox_w", 0)
            h = defect.get("bbox_h", 0)
            
            # Color based on defect type
            if defect.get("type") == "scratch":
                color = (255, 0, 255)  # Magenta
            else:
                color = (0, 165, 255)  # Orange
            
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
            
            # Add label
            label = f"{defect.get('subtype', defect.get('type', 'unknown'))}"
            cv2.putText(vis_image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add status text
        status = results.get("status", "UNKNOWN")
        status_color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
        cv2.putText(vis_image, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(vis_image, f"Defects: {results.get('defect_count', 0)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save or display
        if save_path:
            # Create detailed visualization with matplotlib
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f"Fiber Inspection: {Path(image_path).name}", fontsize=16)
            
            # Original image
            axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis('off')
            
            # Result visualization
            axes[0, 1].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title(f"Detection Result - {status}")
            axes[0, 1].axis('off')
            
            # Zone masks
            if results.get("masks", {}).get("zones"):
                zone_vis = np.zeros_like(image)
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                for i, (zone_name, mask) in enumerate(results["masks"]["zones"].items()):
                    if mask is not None and i < len(colors):
                        zone_vis[mask > 0] = colors[i]
                axes[0, 2].imshow(cv2.cvtColor(zone_vis, cv2.COLOR_BGR2RGB))
                axes[0, 2].set_title("Zone Segmentation")
                axes[0, 2].axis('off')
            
            # Region defects
            if region_mask is not None:
                axes[1, 0].imshow(region_mask, cmap='hot')
                axes[1, 0].set_title("Region Defects (DO2MR)")
                axes[1, 0].axis('off')
            
            # Scratch defects
            if scratch_mask is not None:
                axes[1, 1].imshow(scratch_mask, cmap='hot')
                axes[1, 1].set_title("Scratch Defects (LEI)")
                axes[1, 1].axis('off')
            
            # Defect table
            if results.get("defects"):
                defects_df = pd.DataFrame(results["defects"])
                # Select important columns
                cols = ["type", "subtype", "zone"]
                for col in defects_df.columns:
                    if col.endswith("_um") or col.endswith("_px"):
                        cols.append(col)
                
                table_data = defects_df[cols].head(10)
                
                # Create table
                axes[1, 2].axis('tight')
                axes[1, 2].axis('off')
                table = axes[1, 2].table(cellText=table_data.values,
                                        colLabels=table_data.columns,
                                        cellLoc='center',
                                        loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                axes[1, 2].set_title("Defect Details (Top 10)")
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved to: {save_path}")
        else:
            cv2.imshow("Inspection Result", vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def calibrate_system(calibration_image_path: str, 
                    dot_spacing_um: float = 10.0, 
                    output_path: str = "calibration.json") -> float:
    """Calibrate the system using a calibration target"""
    print(f"Calibrating with {calibration_image_path}...")
    
    # Load image
    image = cv2.imread(calibration_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load calibration image: {calibration_image_path}")
    
    # Enhance image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    # Find calibration dots
    # Use blob detector for more robust dot detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = True
    params.minConvexity = 0.8
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(enhanced)
    
    if len(keypoints) < 2:
        print("Not enough calibration dots found. Trying alternative method...")
        
        # Alternative: Use Hough circles
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=50
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles[0, :]))
            centroids = [(c[0], c[1]) for c in circles]
        else:
            raise ValueError("Could not detect calibration dots")
    else:
        centroids = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
    
    print(f"Found {len(centroids)} calibration dots")
    
    # Calculate distances between dots
    distances = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                          (centroids[i][1] - centroids[j][1])**2)
            distances.append(dist)
    
    # Find the mode of distances (most common spacing)
    hist, bins = np.histogram(distances, bins=50)
    mode_idx = np.argmax(hist)
    mode_distance = (bins[mode_idx] + bins[mode_idx + 1]) / 2
    
    # Filter distances near the mode
    filtered_distances = [d for d in distances if abs(d - mode_distance) < mode_distance * 0.1]
    
    if not filtered_distances:
        filtered_distances = distances
    
    avg_distance_px = np.mean(filtered_distances)
    um_per_px = dot_spacing_um / avg_distance_px
    
    # Save calibration
    calibration_data = {
        "um_per_px": um_per_px,
        "dot_spacing_um": dot_spacing_um,
        "avg_distance_px": avg_distance_px,
        "num_dots_detected": len(centroids),
        "calibration_image": calibration_image_path
    }
    
    with open(output_path, 'w') as f:
        json.dump(calibration_data, f, indent=4)
    
    print(f"Calibration complete: {um_per_px:.4f} µm/pixel")
    print(f"Calibration saved to: {output_path}")
    
    return um_per_px


def main():
    """Main execution function"""
    print("Enhanced Fiber Optic End Face Inspector")
    print("=" * 50)
    
    # Check for calibration
    calibration_path = "calibration.json"
    need_calibration = not Path(calibration_path).exists()
    
    if need_calibration:
        calibrate_choice = input("No calibration found. Calibrate now? (yes/no): ").strip().lower()
        if calibrate_choice == 'yes':
            cal_image = input("Enter path to calibration image: ").strip()
            dot_spacing = float(input("Enter dot spacing in microns (default 10): ").strip() or "10")
            try:
                calibrate_system(cal_image, dot_spacing)
            except Exception as e:
                print(f"Calibration failed: {e}")
                print("Continuing without calibration...")
    
    # Get operation mode
    use_pixel_units = input("Use pixel units? (yes/no, default: no): ").strip().lower() == 'yes'
    
    # Get fiber specifications
    core_diameter = None
    cladding_diameter = None
    
    if not use_pixel_units:
        print("\nEnter fiber specifications (press Enter to use defaults):")
        core_input = input("Core diameter (µm) [default: 9 for SMF]: ").strip()
        clad_input = input("Cladding diameter (µm) [default: 125]: ").strip()
        
        core_diameter = float(core_input) if core_input else None
        cladding_diameter = float(clad_input) if clad_input else None
    
    # Initialize inspector
    try:
        inspector = EnhancedFiberOpticInspector(
            calibration_file=calibration_path,
            core_diameter_um=core_diameter,
            cladding_diameter_um=cladding_diameter,
            use_pixel_units=use_pixel_units
        )
        print("Inspector initialized successfully!")
    except Exception as e:
        print(f"Error initializing inspector: {e}")
        return
    
    # Get images to process
    image_source = input("\nEnter image path or directory: ").strip()
    
    # Collect image paths
    image_paths = []
    if Path(image_source).is_file():
        image_paths.append(image_source)
    elif Path(image_source).is_dir():
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']:
            image_paths.extend(glob.glob(os.path.join(image_source, ext)))
    else:
        print(f"Error: '{image_source}' not found")
        return
    
    if not image_paths:
        print("No images found")
        return
    
    print(f"\nFound {len(image_paths)} images to process")
    
    # Create output directory
    output_dir = Path("inspection_results")
    output_dir.mkdir(exist_ok=True)
    
    # Process images
    all_results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing {i+1}/{len(image_paths)}: {Path(image_path).name}")
        
        try:
            # Inspect
            results = inspector.inspect_fiber(image_path)
            
            # Print summary
            print(f"  Status: {results['status']}")
            print(f"  Defects: {results['defect_count']}")
            
            if results['failure_reasons']:
                print("  Failures:")
                for reason in results['failure_reasons']:
                    print(f"    - {reason}")
            
            # Save visualization
            output_path = output_dir / f"{Path(image_path).stem}_inspected.png"
            inspector.visualize_results(image_path, results, str(output_path))
            
            # Store results
            all_results.append({
                "image": Path(image_path).name,
                "status": results['status'],
                "defect_count": results['defect_count'],
                "failures": "; ".join(results['failure_reasons'])
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            all_results.append({
                "image": Path(image_path).name,
                "status": "ERROR",
                "defect_count": -1,
                "failures": str(e)
            })
    
    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_path = output_dir / "inspection_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nInspection complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()