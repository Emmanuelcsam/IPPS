import cv2
import numpy as np
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
import glob
import os
from scipy import ndimage
from skimage import measure, morphology
import warnings
warnings.filterwarnings('ignore')

class EnhancedFiberOpticInspector:
    """
    Enhanced Automated Fiber Optic End Face Defect Detection System
    Using multiple OpenCV methods with ensemble validation for maximum accuracy
    """
    
    def __init__(self,
                 user_core_diameter_um: Optional[float] = None,
                 user_cladding_diameter_um: Optional[float] = None,
                 user_ferrule_outer_diameter_um: Optional[float] = 250.0,
                 calibration_file_path: str = "calibration.json"):
        
        # User-provided dimensions
        self.user_core_diameter_um = user_core_diameter_um
        self.user_cladding_diameter_um = user_cladding_diameter_um
        self.user_ferrule_outer_diameter_um = user_ferrule_outer_diameter_um
        
        # Calibration and scaling
        self.calibrated_um_per_px: Optional[float] = None
        self.effective_um_per_px: Optional[float] = None
        self.operating_mode: str = "PIXEL_ONLY"
        
        # Try to load calibration if available
        self._attempt_calibration_load(calibration_file_path)
        
        # Set operating mode based on available information
        self._determine_operating_mode()
        
        # Zone definitions (will be updated based on user input)
        self._initialize_zone_definitions()
        
        # Detection parameters for multiple methods
        self._initialize_detection_parameters()
        
        # Ensemble validation thresholds
        self.ensemble_config = {
            "min_methods_for_detection": 2,  # Minimum methods that must agree
            "confidence_weights": {
                "do2mr": 1.0,
                "lei": 1.0,
                "gradient": 0.8,
                "otsu": 0.7,
                "adaptive": 0.8,
                "watershed": 0.9,
                "canny": 0.7
            }
        }

    def _attempt_calibration_load(self, filepath: str):
        """Attempt to load calibration data"""
        if Path(filepath).exists():
            try:
                with open(filepath, 'r') as f:
                    cal_data = json.load(f)
                    self.calibrated_um_per_px = cal_data.get("um_per_px")
                    if self.calibrated_um_per_px and self.calibrated_um_per_px > 0:
                        self.effective_um_per_px = self.calibrated_um_per_px
                        print(f"Loaded calibration: {self.effective_um_per_px:.4f} µm/px")
            except Exception as e:
                print(f"Could not load calibration: {e}")

    def _determine_operating_mode(self):
        """Determine the operating mode based on available information"""
        if self.effective_um_per_px:
            self.operating_mode = "MICRON_CALIBRATED"
            print("Operating in MICRON_CALIBRATED mode")
        elif self.user_cladding_diameter_um:
            self.operating_mode = "MICRON_INFERRED"
            print("Operating in MICRON_INFERRED mode (will infer µm/px from detected features)")
        else:
            self.operating_mode = "PIXEL_ONLY"
            print("Operating in PIXEL_ONLY mode")

    def _initialize_zone_definitions(self):
        """Initialize zone definitions based on user input or defaults"""
        # Micron-based zone template
        self.zones_um_template = {
            "core": {"r_min": 0, "r_max": 4.5, "max_defect_um": 3, "defects_allowed": True},
            "cladding": {"r_min": 4.5, "r_max": 62.5, "max_defect_um": 10, "defects_allowed": True},
            "ferrule_contact": {"r_min": 62.5, "r_max": 125.0, "max_defect_um": 25, "defects_allowed": True},
            "adhesive_bond": {"r_min": 125.0, "r_max": 140.0, "max_defect_um": 50, "defects_allowed": True},
        }
        
        # Pixel-based zone template
        self.zones_px_template = {
            "core": {"r_min_px": 0, "r_max_px": 30, "max_defect_px": 5, "defects_allowed": True},
            "cladding": {"r_min_px": 30, "r_max_px": 80, "max_defect_px": 15, "defects_allowed": True},
            "ferrule_contact": {"r_min_px": 80, "r_max_px": 150, "max_defect_px": 25, "defects_allowed": True},
        }
        
        # Update templates if user provided dimensions
        if self.user_core_diameter_um and self.user_cladding_diameter_um:
            core_r = self.user_core_diameter_um / 2.0
            cladding_r = self.user_cladding_diameter_um / 2.0
            ferrule_r = (self.user_ferrule_outer_diameter_um or 250.0) / 2.0
            
            self.zones_um_template["core"]["r_max"] = core_r
            self.zones_um_template["cladding"]["r_min"] = core_r
            self.zones_um_template["cladding"]["r_max"] = cladding_r
            self.zones_um_template["ferrule_contact"]["r_min"] = cladding_r
            self.zones_um_template["ferrule_contact"]["r_max"] = ferrule_r
            print(f"Updated zone definitions with user dimensions: Core={self.user_core_diameter_um}µm, Cladding={self.user_cladding_diameter_um}µm")

    def _initialize_detection_parameters(self):
        """Initialize parameters for all detection methods"""
        # DO2MR parameters
        self.do2mr_params = {
            "kernel_sizes": [(11, 11), (15, 15), (19, 19)],  # Multiple kernel sizes
            "gamma_values": [2.5, 3.0, 3.5],  # Multiple sensitivity levels
            "min_area_px": 15
        }
        
        # LEI parameters
        self.lei_params = {
            "kernel_sizes": [11, 15, 19],  # Multiple lengths
            "angles": np.arange(0, 180, 5),  # Higher angular resolution
            "threshold_factors": [2.0, 2.5, 3.0]
        }
        
        # Gradient-based parameters
        self.gradient_params = {
            "sobel_ksize": [3, 5, 7],
            "gradient_threshold_factors": [1.5, 2.0, 2.5]
        }
        
        # Canny parameters
        self.canny_params = {
            "low_threshold_ratios": [0.1, 0.15, 0.2],
            "high_threshold_ratios": [0.3, 0.35, 0.4]
        }
        
        # Adaptive threshold parameters
        self.adaptive_params = {
            "block_sizes": [11, 21, 31],
            "C_values": [2, 5, 8]
        }

    def preprocess_image_multi(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply multiple preprocessing techniques"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        preprocessed = {
            "original": gray,
            "gaussian": cv2.GaussianBlur(gray, (5, 5), 0),
            "bilateral": cv2.bilateralFilter(gray, 9, 75, 75),
            "median": cv2.medianBlur(gray, 5),
            "nlmeans": cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        }
        
        # Add contrast enhancement variants
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        preprocessed["clahe"] = clahe.apply(preprocessed["bilateral"])
        
        # Histogram equalization
        preprocessed["histeq"] = cv2.equalizeHist(gray)
        
        return preprocessed

    def find_fiber_center_multi_method(self, preprocessed_images: Dict[str, np.ndarray]) -> Tuple[Tuple[int, int], float, float]:
        """Find fiber center using multiple methods and vote for best result"""
        candidates = []
        
        # Method 1: Hough Circles on different preprocessed images
        for img_type, img in preprocessed_images.items():
            if img_type in ["gaussian", "bilateral", "median"]:
                centers, radii = self._hough_circle_detection(img)
                candidates.extend([(c, r, "hough_" + img_type) for c, r in zip(centers, radii)])
        
        # Method 2: Contour-based detection
        for img_type in ["bilateral", "nlmeans"]:
            img = preprocessed_images[img_type]
            center, radius = self._contour_based_circle_detection(img)
            if center and radius:
                candidates.append((center, radius, "contour_" + img_type))
        
        # Method 3: Edge-based with ellipse fitting
        for img_type in ["clahe", "bilateral"]:
            img = preprocessed_images[img_type]
            center, radius = self._edge_based_circle_detection(img)
            if center and radius:
                candidates.append((center, radius, "edge_" + img_type))
        
        # Vote for best center and radius
        if not candidates:
            # Fallback to image center
            h, w = preprocessed_images["original"].shape
            return (w//2, h//2), min(w, h)//4, 0.0
        
        # Cluster centers and radii
        centers = np.array([c[0] for c in candidates])
        radii = np.array([c[1] for c in candidates])
        
        # Use median for robustness
        best_center = (int(np.median(centers[:, 0])), int(np.median(centers[:, 1])))
        best_radius = float(np.median(radii))
        confidence = len(candidates) / (len(preprocessed_images) * 3)  # Normalize by possible detections
        
        return best_center, best_radius, confidence

    def _hough_circle_detection(self, image: np.ndarray) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Detect circles using Hough transform with multiple parameters"""
        centers = []
        radii = []
        
        edges = cv2.Canny(image, 50, 150)
        
        for dp in [1.0, 1.2, 1.5]:
            for param2 in [30, 40, 50]:
                circles = cv2.HoughCircles(
                    edges, cv2.HOUGH_GRADIENT, dp=dp,
                    minDist=image.shape[0]//8,
                    param1=50, param2=param2,
                    minRadius=image.shape[0]//10,
                    maxRadius=image.shape[0]//2
                )
                
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0]:
                        centers.append((int(circle[0]), int(circle[1])))
                        radii.append(float(circle[2]))
        
        return centers, radii

    def _contour_based_circle_detection(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """Detect circle using contours and minimum enclosing circle"""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        return (int(x), int(y)), float(radius)

    def _edge_based_circle_detection(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """Detect circle using edges and ellipse fitting"""
        edges = cv2.Canny(image, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Merge all contours
        all_points = np.vstack(contours)
        
        if len(all_points) < 5:
            return None, None
        
        ellipse = cv2.fitEllipse(all_points)
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        radius = float(np.mean(ellipse[1]) / 2)  # Average of major and minor axes
        
        return center, radius

    def detect_defects_multi_method(self, preprocessed_images: Dict[str, np.ndarray], 
                                   zone_masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply multiple defect detection methods"""
        defect_maps = {}
        
        # 1. DO2MR with multiple parameters
        for img_type in ["clahe", "bilateral", "gaussian"]:
            for kernel_size in self.do2mr_params["kernel_sizes"]:
                for gamma in self.do2mr_params["gamma_values"]:
                    mask = self._apply_do2mr(preprocessed_images[img_type], kernel_size, gamma)
                    defect_maps[f"do2mr_{img_type}_{kernel_size[0]}_{gamma}"] = mask
        
        # 2. LEI for scratches
        for img_type in ["original", "gaussian"]:
            for kernel_size in self.lei_params["kernel_sizes"]:
                mask = self._apply_lei(preprocessed_images[img_type], kernel_size)
                defect_maps[f"lei_{img_type}_{kernel_size}"] = mask
        
        # 3. Gradient-based detection
        for img_type in ["clahe", "bilateral"]:
            for ksize in self.gradient_params["sobel_ksize"]:
                mask = self._gradient_based_detection(preprocessed_images[img_type], ksize)
                defect_maps[f"gradient_{img_type}_{ksize}"] = mask
        
        # 4. Otsu thresholding variants
        for img_type in ["clahe", "histeq", "bilateral"]:
            mask = self._otsu_based_detection(preprocessed_images[img_type])
            defect_maps[f"otsu_{img_type}"] = mask
        
        # 5. Adaptive thresholding
        for img_type in ["clahe", "bilateral"]:
            for block_size in self.adaptive_params["block_sizes"]:
                mask = self._adaptive_threshold_detection(preprocessed_images[img_type], block_size)
                defect_maps[f"adaptive_{img_type}_{block_size}"] = mask
        
        # 6. Watershed segmentation
        watershed_mask = self._watershed_detection(preprocessed_images["bilateral"])
        defect_maps["watershed"] = watershed_mask
        
        # 7. Canny edge-based
        for img_type in ["gaussian", "bilateral"]:
            for low_ratio in self.canny_params["low_threshold_ratios"]:
                mask = self._canny_based_detection(preprocessed_images[img_type], low_ratio)
                defect_maps[f"canny_{img_type}_{low_ratio}"] = mask
        
        return defect_maps

    def _apply_do2mr(self, image: np.ndarray, kernel_size: Tuple[int, int], gamma: float) -> np.ndarray:
        """Apply DO2MR algorithm"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        img_max = cv2.dilate(image, kernel)
        img_min = cv2.erode(image, kernel)
        residual = cv2.absdiff(img_max, img_min)
        
        residual_filtered = cv2.medianBlur(residual, 5)
        mean_val = np.mean(residual_filtered)
        std_val = np.std(residual_filtered)
        threshold_val = mean_val + gamma * std_val
        
        _, binary_mask = cv2.threshold(residual_filtered, threshold_val, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        
        return binary_mask

    def _apply_lei(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply LEI algorithm for scratch detection"""
        scratch_strength = np.zeros_like(image, dtype=np.float32)
        
        for angle in self.lei_params["angles"]:
            angle_rad = np.deg2rad(angle)
            
            # Create line kernel
            kernel_points = []
            for i in range(-kernel_size//2, kernel_size//2 + 1):
                if i != 0:
                    x = int(round(i * np.cos(angle_rad)))
                    y = int(round(i * np.sin(angle_rad)))
                    kernel_points.append((x, y))
            
            if kernel_points:
                response = self._apply_linear_detector(image, kernel_points)
                scratch_strength = np.maximum(scratch_strength, response)
        
        # Normalize and threshold
        if scratch_strength.max() > 0:
            scratch_strength_norm = cv2.normalize(scratch_strength, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            scratch_strength_norm = np.zeros_like(scratch_strength, dtype=np.uint8)
        
        _, scratch_mask = cv2.threshold(scratch_strength_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size//3, 3))
        scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, kernel_close)
        
        return scratch_mask

    def _apply_linear_detector(self, image: np.ndarray, kernel_points: List[Tuple[int, int]]) -> np.ndarray:
        """Apply linear detector for LEI"""
        height, width = image.shape
        response = np.zeros_like(image, dtype=np.float32)
        
        max_offset = max(max(abs(dx), abs(dy)) for dx, dy in kernel_points)
        padded = cv2.copyMakeBorder(image, max_offset, max_offset, max_offset, max_offset, cv2.BORDER_REFLECT)
        
        for r in range(height):
            for c in range(width):
                line_values = []
                for dx, dy in kernel_points:
                    line_values.append(float(padded[r + max_offset + dy, c + max_offset + dx]))
                
                if line_values:
                    center_val = float(padded[r + max_offset, c + max_offset])
                    # Detect both bright and dark scratches
                    bright_response = np.mean(line_values) - center_val
                    dark_response = center_val - np.mean(line_values)
                    response[r, c] = max(0, max(bright_response, dark_response))
        
        return response

    def _gradient_based_detection(self, image: np.ndarray, ksize: int) -> np.ndarray:
        """Detect defects using gradient methods"""
        # Sobel gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # Gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Threshold
        _, mask = cv2.threshold(magnitude_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return mask

    def _otsu_based_detection(self, image: np.ndarray) -> np.ndarray:
        """Detect defects using Otsu thresholding"""
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find defects as deviations from the binary pattern
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Defects are differences between operations
        defects = cv2.absdiff(opened, closed)
        
        return defects

    def _adaptive_threshold_detection(self, image: np.ndarray, block_size: int) -> np.ndarray:
        """Detect defects using adaptive thresholding"""
        # Apply adaptive threshold
        adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, 2)
        
        # Find small isolated regions as defects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        eroded = cv2.erode(adaptive, kernel, iterations=1)
        
        # Defects are removed by erosion
        defects = cv2.absdiff(adaptive, eroded)
        
        return defects

    def _watershed_detection(self, image: np.ndarray) -> np.ndarray:
        """Detect defects using watershed segmentation"""
        # Threshold
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Find peaks (sure foreground)
        _, sure_fg = cv2.threshold(dist, 0.3*dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Find unknown region
        kernel = np.ones((3,3), np.uint8)
        sure_bg = cv2.dilate(binary, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        
        # Defects are watershed boundaries
        defect_mask = np.zeros_like(image)
        defect_mask[markers == -1] = 255
        
        return defect_mask

    def _canny_based_detection(self, image: np.ndarray, low_ratio: float) -> np.ndarray:
        """Detect defects using Canny edge detection"""
        # Calculate thresholds based on image statistics
        v = np.median(image)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v * low_ratio))
        upper = int(min(255, (1.0 + sigma) * v * low_ratio * 3))
        
        # Apply Canny
        edges = cv2.Canny(image, lower, upper)
        
        # Close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return closed

    def ensemble_defect_validation(self, defect_maps: Dict[str, np.ndarray], 
                                  zone_masks: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Combine multiple detection methods using ensemble validation"""
        h, w = next(iter(defect_maps.values())).shape
        
        # Create voting maps
        region_vote_map = np.zeros((h, w), dtype=np.float32)
        scratch_vote_map = np.zeros((h, w), dtype=np.float32)
        
        # Accumulate weighted votes
        for method_name, mask in defect_maps.items():
            # Determine weight based on method type
            weight = 1.0
            for method_type, method_weight in self.ensemble_config["confidence_weights"].items():
                if method_type in method_name:
                    weight = method_weight
                    break
            
            # Separate scratches from regions based on method
            if "lei" in method_name or "canny" in method_name:
                scratch_vote_map += mask.astype(np.float32) * weight / 255.0
            else:
                region_vote_map += mask.astype(np.float32) * weight / 255.0
        
        # Normalize vote maps
        max_votes = len(defect_maps)
        region_vote_map /= max_votes
        scratch_vote_map /= max_votes
        
        # Apply ensemble threshold
        min_vote_ratio = self.ensemble_config["min_methods_for_detection"] / max_votes
        
        # Create final masks
        region_mask = (region_vote_map >= min_vote_ratio).astype(np.uint8) * 255
        scratch_mask = (scratch_vote_map >= min_vote_ratio).astype(np.uint8) * 255
        
        # Clean up masks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel)
        scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate confidence metrics
        confidence_metrics = {
            "region_avg_confidence": np.mean(region_vote_map[region_mask > 0]) if np.any(region_mask) else 0,
            "scratch_avg_confidence": np.mean(scratch_vote_map[scratch_mask > 0]) if np.any(scratch_mask) else 0,
            "total_methods_used": len(defect_maps),
            "consensus_regions": np.sum(region_vote_map >= min_vote_ratio),
            "consensus_scratches": np.sum(scratch_vote_map >= min_vote_ratio)
        }
        
        return region_mask, scratch_mask, confidence_metrics

    def refine_with_zone_context(self, region_mask: np.ndarray, scratch_mask: np.ndarray,
                                zone_masks: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Refine defects based on zone context"""
        # Different minimum sizes for different zones
        zone_min_areas = {
            "core": 10,
            "cladding": 15,
            "ferrule_contact": 20,
            "adhesive_bond": 25
        }
        
        # Process region defects
        refined_region = np.zeros_like(region_mask)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region_mask, connectivity=8)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            
            # Determine zone
            zone = "unknown"
            for zone_name, zone_mask in zone_masks.items():
                if zone_mask[y, x] > 0:
                    zone = zone_name
                    break
            
            # Apply zone-specific filtering
            min_area = zone_min_areas.get(zone, 20)
            if area >= min_area:
                refined_region[labels == i] = 255
        
        # Similar processing for scratches with length constraints
        refined_scratch = scratch_mask.copy()  # Scratches typically need less filtering
        
        return refined_region, refined_scratch

    def create_zone_masks(self, image_shape: Tuple, center: Tuple[int, int], 
                         detected_radius: float) -> Dict[str, np.ndarray]:
        """Create zone masks with smart scaling"""
        masks = {}
        height, width = image_shape[:2]
        Y, X = np.ogrid[:height, :width]
        dist_from_center_sq = (X - center[0])**2 + (Y - center[1])**2
        
        # Determine effective scaling
        if self.operating_mode == "MICRON_CALIBRATED":
            scale = self.effective_um_per_px
            zones = self.zones_um_template
            suffix = ""
        elif self.operating_mode == "MICRON_INFERRED" and self.user_cladding_diameter_um:
            # Infer scale from detected radius
            self.effective_um_per_px = (self.user_cladding_diameter_um / 2.0) / detected_radius
            scale = self.effective_um_per_px
            zones = self.zones_um_template
            suffix = ""
            print(f"Inferred scale: {scale:.4f} µm/px")
        else:
            # Pixel mode - dynamically adjust zones based on detected radius
            scale = 1.0
            zones = self.zones_px_template.copy()
            suffix = "_px"
            
            # Scale pixel zones based on detected radius
            if self.user_cladding_diameter_um and self.user_core_diameter_um:
                core_ratio = (self.user_core_diameter_um / 2.0) / (self.user_cladding_diameter_um / 2.0)
                zones["core"]["r_max_px"] = int(detected_radius * core_ratio)
                zones["cladding"]["r_min_px"] = zones["core"]["r_max_px"]
                zones["cladding"]["r_max_px"] = int(detected_radius)
                zones["ferrule_contact"]["r_min_px"] = int(detected_radius)
                zones["ferrule_contact"]["r_max_px"] = int(detected_radius * 2)
        
        # Create masks
        for zone_name, zone_params in zones.items():
            r_min = zone_params.get(f"r_min{suffix}", 0)
            r_max = zone_params.get(f"r_max{suffix}", float('inf'))
            
            r_min_px_sq = (r_min / scale) ** 2
            r_max_px_sq = (r_max / scale) ** 2
            
            zone_mask = (dist_from_center_sq >= r_min_px_sq) & (dist_from_center_sq < r_max_px_sq)
            masks[zone_name] = zone_mask.astype(np.uint8) * 255
        
        return masks

    def classify_defects_enhanced(self, region_mask: np.ndarray, scratch_mask: np.ndarray,
                                 zone_masks: Dict[str, np.ndarray], 
                                 confidence_metrics: Dict) -> pd.DataFrame:
        """Enhanced defect classification with confidence scores"""
        defects = []
        
        # Get active scale
        scale = self.effective_um_per_px if self.effective_um_per_px else 1.0
        unit = "um" if self.effective_um_per_px else "px"
        
        # Process regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(region_mask, connectivity=8)
        
        for i in range(1, num_labels):
            area_px = stats[i, cv2.CC_STAT_AREA]
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # Determine zone
            zone = "unknown"
            for zone_name, zone_mask in zone_masks.items():
                if 0 <= cy < zone_mask.shape[0] and 0 <= cx < zone_mask.shape[1]:
                    if zone_mask[cy, cx] > 0:
                        zone = zone_name
                        break
            
            # Calculate size
            diameter = np.sqrt(4 * area_px / np.pi) * scale
            
            # Determine defect type based on shape
            aspect_ratio = w / h if h > 0 else 1.0
            circularity = 4 * np.pi * area_px / (cv2.arcLength(
                cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True) ** 2)
            
            if circularity > 0.7:
                defect_type = "dig"
            elif aspect_ratio > 3 or aspect_ratio < 0.33:
                defect_type = "scratch-like"
            else:
                defect_type = "irregular"
            
            defects.append({
                "type": defect_type,
                "zone": zone,
                f"diameter_{unit}": round(diameter, 2),
                "area_px": area_px,
                "centroid_x": cx,
                "centroid_y": cy,
                "aspect_ratio": round(aspect_ratio, 2),
                "circularity": round(circularity, 2),
                "confidence": confidence_metrics["region_avg_confidence"],
                "source": "multi-method-region"
            })
        
        # Process scratches
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(scratch_mask, connectivity=8)
        
        for i in range(1, num_labels):
            area_px = stats[i, cv2.CC_STAT_AREA]
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            
            # Get oriented bounding box for better length measurement
            contour = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            if len(contour) >= 5:
                rect = cv2.minAreaRect(contour)
                (_, _), (width, height), angle = rect
                length = max(width, height) * scale
                width = min(width, height) * scale
            else:
                length = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) * scale
                width = min(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) * scale
            
            # Determine zone
            zone = "unknown"
            for zone_name, zone_mask in zone_masks.items():
                if 0 <= cy < zone_mask.shape[0] and 0 <= cx < zone_mask.shape[1]:
                    if zone_mask[cy, cx] > 0:
                        zone = zone_name
                        break
            
            defects.append({
                "type": "scratch",
                "zone": zone,
                f"length_{unit}": round(length, 2),
                f"width_{unit}": round(width, 2),
                "area_px": area_px,
                "centroid_x": cx,
                "centroid_y": cy,
                "confidence": confidence_metrics["scratch_avg_confidence"],
                "source": "multi-method-scratch"
            })
        
        return pd.DataFrame(defects)

    def apply_pass_fail_criteria_enhanced(self, defects_df: pd.DataFrame) -> Tuple[str, List[str], Dict]:
        """Enhanced pass/fail criteria with detailed metrics"""
        if self.operating_mode == "PIXEL_ONLY" and not self.effective_um_per_px:
            # Apply pixel-based criteria
            return self._apply_pixel_criteria(defects_df)
        else:
            # Apply micron-based criteria
            return self._apply_micron_criteria(defects_df)

    def _apply_micron_criteria(self, defects_df: pd.DataFrame) -> Tuple[str, List[str], Dict]:
        """Apply micron-based pass/fail criteria"""
        status = "PASS"
        failure_reasons = []
        metrics = {
            "total_defects": len(defects_df),
            "defects_by_zone": {},
            "largest_defect_um": 0,
            "total_defect_area_um2": 0
        }
        
        for zone_name, zone_criteria in self.zones_um_template.items():
            zone_defects = defects_df[defects_df["zone"] == zone_name]
            metrics["defects_by_zone"][zone_name] = len(zone_defects)
            
            if not zone_criteria.get("defects_allowed", True) and len(zone_defects) > 0:
                status = "FAIL"
                failure_reasons.append(f"{zone_name}: No defects allowed but {len(zone_defects)} found")
                continue
            
            max_allowed = zone_criteria.get("max_defect_um", float('inf'))
            
            for _, defect in zone_defects.iterrows():
                size_um = 0
                if defect["type"] in ["dig", "irregular", "scratch-like"]:
                    size_um = defect.get("diameter_um", 0)
                elif defect["type"] == "scratch":
                    size_um = defect.get("length_um", 0)
                
                metrics["largest_defect_um"] = max(metrics["largest_defect_um"], size_um)
                
                if size_um > max_allowed:
                    status = "FAIL"
                    failure_reasons.append(
                        f"{zone_name}: {defect['type']} ({size_um:.1f}µm) exceeds limit ({max_allowed}µm)"
                    )
            
            # Additional criteria
            if zone_name == "core" and len(zone_defects) > 5:
                status = "FAIL"
                failure_reasons.append(f"Core: Too many defects ({len(zone_defects)} > 5)")
        
        return status, failure_reasons, metrics

    def _apply_pixel_criteria(self, defects_df: pd.DataFrame) -> Tuple[str, List[str], Dict]:
        """Apply pixel-based pass/fail criteria"""
        status = "PASS"
        failure_reasons = []
        metrics = {
            "total_defects": len(defects_df),
            "defects_by_zone": {},
            "largest_defect_px": 0
        }
        
        # Use pixel-based thresholds
        for zone_name, zone_criteria in self.zones_px_template.items():
            zone_defects = defects_df[defects_df["zone"] == zone_name]
            metrics["defects_by_zone"][zone_name] = len(zone_defects)
            
            max_allowed_px = zone_criteria.get("max_defect_px", float('inf'))
            
            for _, defect in zone_defects.iterrows():
                size_px = 0
                if defect["type"] in ["dig", "irregular", "scratch-like"]:
                    size_px = defect.get("diameter_px", np.sqrt(defect.get("area_px", 0)))
                elif defect["type"] == "scratch":
                    size_px = defect.get("length_px", max(defect.get("area_px", 0) / 5, 0))
                
                metrics["largest_defect_px"] = max(metrics["largest_defect_px"], size_px)
                
                if size_px > max_allowed_px:
                    status = "FAIL"
                    failure_reasons.append(
                        f"{zone_name}: {defect['type']} ({size_px:.0f}px) exceeds limit ({max_allowed_px}px)"
                    )
        
        return status, failure_reasons, metrics

    def inspect_fiber_enhanced(self, image_path: str) -> Dict:
        """Main inspection function with enhanced multi-method approach"""
        print(f"\nInspecting: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                "image_path": image_path,
                "status": "ERROR",
                "failure_reasons": [f"Could not load image: {image_path}"],
                "defect_count": 0,
                "defects": []
            }
        
        # Multi-method preprocessing
        preprocessed = self.preprocess_image_multi(image)
        
        # Find fiber center using multiple methods
        center, radius, center_confidence = self.find_fiber_center_multi_method(preprocessed)
        print(f"  Fiber center: {center}, radius: {radius:.1f}px (confidence: {center_confidence:.2f})")
        
        # Create zone masks
        zone_masks = self.create_zone_masks(image.shape[:2], center, radius)
        
        # Apply multiple defect detection methods
        print("  Applying multiple detection methods...")
        defect_maps = self.detect_defects_multi_method(preprocessed, zone_masks)
        print(f"  Generated {len(defect_maps)} defect maps")
        
        # Ensemble validation
        print("  Performing ensemble validation...")
        region_mask, scratch_mask, confidence_metrics = self.ensemble_defect_validation(defect_maps, zone_masks)
        
        # Refine with zone context
        region_mask, scratch_mask = self.refine_with_zone_context(region_mask, scratch_mask, zone_masks)
        
        # Classify defects
        defects_df = self.classify_defects_enhanced(region_mask, scratch_mask, zone_masks, confidence_metrics)
        
        # Apply pass/fail criteria
        status, failure_reasons, metrics = self.apply_pass_fail_criteria_enhanced(defects_df)
        
        # Compile results
        results = {
            "image_path": image_path,
            "status": status,
            "failure_reasons": failure_reasons,
            "defect_count": len(defects_df),
            "defects": defects_df.to_dict('records') if not defects_df.empty else [],
            "fiber_center": center,
            "fiber_radius_px": radius,
            "center_confidence": center_confidence,
            "operating_mode": self.operating_mode,
            "effective_um_per_px": self.effective_um_per_px if self.effective_um_per_px else "N/A",
            "confidence_metrics": confidence_metrics,
            "analysis_metrics": metrics,
            "masks": {
                "region_defects": region_mask,
                "scratches": scratch_mask,
                "zone_masks": zone_masks
            }
        }
        
        return results

    def visualize_results_enhanced(self, image_path: str, results: Dict, save_path: Optional[str] = None):
        """Enhanced visualization with confidence information"""
        image = cv2.imread(image_path)
        if image is None:
            return
        
        # Create multi-panel visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Enhanced Inspection: {Path(image_path).name} - Status: {results['status']}", fontsize=16)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Zone visualization
        zone_vis = np.zeros_like(image)
        zone_colors = {
            "core": [255, 0, 0],
            "cladding": [0, 255, 0],
            "ferrule_contact": [0, 0, 255],
            "adhesive_bond": [255, 255, 0]
        }
        
        for zone_name, zone_mask in results["masks"]["zone_masks"].items():
            if zone_name in zone_colors:
                zone_vis[zone_mask > 0] = zone_colors[zone_name]
        
        axes[0, 1].imshow(zone_vis)
        axes[0, 1].set_title("Zone Map")
        axes[0, 1].axis('off')
        
        # Region defects
        axes[0, 2].imshow(results["masks"]["region_defects"], cmap='hot')
        axes[0, 2].set_title(f"Region Defects (Confidence: {results['confidence_metrics']['region_avg_confidence']:.2f})")
        axes[0, 2].axis('off')
        
        # Scratch defects
        axes[1, 0].imshow(results["masks"]["scratches"], cmap='hot')
        axes[1, 0].set_title(f"Scratches (Confidence: {results['confidence_metrics']['scratch_avg_confidence']:.2f})")
        axes[1, 0].axis('off')
        
        # Combined result
        vis_image = image.copy()
        center = results["fiber_center"]
        
        # Draw zone boundaries
        for zone_name, zone_mask in results["masks"]["zone_masks"].items():
            contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, zone_colors.get(zone_name, [128, 128, 128]), 2)
        
        # Overlay defects
        vis_image[results["masks"]["region_defects"] > 0] = [0, 255, 255]  # Yellow
        vis_image[results["masks"]["scratches"] > 0] = [255, 0, 255]  # Magenta
        
        # Add status text
        status_color = (0, 255, 0) if results["status"] == "PASS" else (0, 0, 255)
        cv2.putText(vis_image, f"Status: {results['status']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(vis_image, f"Defects: {results['defect_count']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        axes[1, 1].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Final Result")
        axes[1, 1].axis('off')
        
        # Metrics display
        metrics_text = f"Operating Mode: {results['operating_mode']}\n"
        metrics_text += f"Scale: {results['effective_um_per_px']}\n"
        metrics_text += f"Methods Used: {results['confidence_metrics']['total_methods_used']}\n"
        metrics_text += f"Center Confidence: {results['center_confidence']:.2f}\n\n"
        
        if results['analysis_metrics']:
            metrics_text += "Defects by Zone:\n"
            for zone, count in results['analysis_metrics']['defects_by_zone'].items():
                metrics_text += f"  {zone}: {count}\n"
        
        axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontsize=10, family='monospace')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def process_batch(inspector: EnhancedFiberOpticInspector, image_paths: List[str], output_dir: str):
    """Process a batch of images"""
    results_summary = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing {i+1}/{len(image_paths)}: {Path(image_path).name}")
        
        try:
            # Inspect image
            results = inspector.inspect_fiber_enhanced(image_path)
            
            # Save visualization
            vis_path = output_path / f"{Path(image_path).stem}_inspection.png"
            inspector.visualize_results_enhanced(image_path, results, str(vis_path))
            
            # Add to summary
            results_summary.append({
                "image": Path(image_path).name,
                "status": results["status"],
                "defect_count": results["defect_count"],
                "mode": results["operating_mode"],
                "scale": results["effective_um_per_px"],
                "center_confidence": results["center_confidence"],
                "region_confidence": results["confidence_metrics"]["region_avg_confidence"],
                "scratch_confidence": results["confidence_metrics"]["scratch_avg_confidence"],
                "failure_reasons": "; ".join(results["failure_reasons"])
            })
            
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            results_summary.append({
                "image": Path(image_path).name,
                "status": "ERROR",
                "error": str(e)
            })
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(output_path / "inspection_summary.csv", index=False)
    print(f"\nBatch processing complete. Results saved to {output_path}")

# Main execution
if __name__ == "__main__":
    print("Enhanced Fiber Optic Inspection System")
    print("=" * 50)
    
    # Get user input for dimensions
    print("\nFiber Specifications (press Enter to skip):")
    core_diameter = input("Core diameter (µm): ").strip()
    cladding_diameter = input("Cladding diameter (µm): ").strip()
    
    # Convert to float or None
    core_diameter = float(core_diameter) if core_diameter else None
    cladding_diameter = float(cladding_diameter) if cladding_diameter else None
    
    # Initialize inspector
    inspector = EnhancedFiberOpticInspector(
        user_core_diameter_um=core_diameter,
        user_cladding_diameter_um=cladding_diameter
    )
    
    # Get input path
    input_path = input("\nEnter image path or directory: ").strip()
    
    # Determine if single image or batch
    if Path(input_path).is_file():
        image_paths = [input_path]
    elif Path(input_path).is_dir():
        image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
            image_paths.extend(glob.glob(os.path.join(input_path, ext)))
    else:
        print(f"Error: '{input_path}' is not a valid file or directory")
        exit(1)
    
    if not image_paths:
        print(f"No images found in '{input_path}'")
        exit(1)
    
    print(f"\nFound {len(image_paths)} image(s) to process")
    
    # Process images
    output_dir = "./enhanced_inspection_results"
    process_batch(inspector, image_paths, output_dir)
