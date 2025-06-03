import cv2
import numpy as np
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import glob
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DefectDetectionParams:
    """Parameters for various defect detection algorithms"""
    # DO2MR parameters
    do2mr_kernel_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(5,5), (11,11), (15,15), (21,21)])
    do2mr_gamma_values: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0, 3.5])
    do2mr_min_area: int = 10
    
    # LEI parameters  
    lei_kernel_lengths: List[int] = field(default_factory=lambda: [11, 15, 19])
    lei_angle_steps: List[int] = field(default_factory=lambda: [5, 10, 15])
    lei_threshold_factors: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0])
    
    # Canny parameters
    canny_thresholds: List[Tuple[int, int]] = field(default_factory=lambda: [(50,150), (70,200), (100,250)])
    canny_apertures: List[int] = field(default_factory=lambda: [3, 5, 7])
    
    # Morphology parameters
    morph_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    morph_iterations: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # Threshold parameters
    adaptive_block_sizes: List[int] = field(default_factory=lambda: [11, 21, 31])
    adaptive_C_values: List[int] = field(default_factory=lambda: [2, 5, 8])

@dataclass 
class FiberZoneDefinition:
    """Definition of a fiber optic zone"""
    name: str
    r_min_um: float
    r_max_um: float
    max_defect_um: float
    defects_allowed: bool = True
    color_bgr: Tuple[int, int, int] = (255, 255, 255)

class AdvancedFiberOpticInspector:
    """
    Advanced Automated Fiber Optic End Face Defect Detection System
    Implements multiple OpenCV techniques for robust defect detection
    """
    
    def __init__(self,
                 user_core_diameter_um: Optional[float] = None,
                 user_cladding_diameter_um: Optional[float] = None,
                 user_ferrule_diameter_um: Optional[float] = 250.0,
                 calibration_file: Optional[str] = None,
                 params: Optional[DefectDetectionParams] = None):
        
        self.user_core_diameter_um = user_core_diameter_um
        self.user_cladding_diameter_um = user_cladding_diameter_um  
        self.user_ferrule_diameter_um = user_ferrule_diameter_um or 250.0
        
        self.params = params or DefectDetectionParams()
        
        # Calibration and scaling
        self.calibrated_um_per_px: Optional[float] = None
        self.effective_um_per_px: Optional[float] = None
        self.operating_mode: str = "PIXEL_ONLY"
        
        # Load calibration if available
        if calibration_file and Path(calibration_file).exists():
            self._load_calibration(calibration_file)
            
        # Define fiber zones
        self._setup_zones()
        
        # Hough circle detection parameters
        self.hough_params = {
            "dp_values": [1.0, 1.2, 1.5],
            "param1_values": [50, 70, 90],
            "param2_values": [30, 40, 50],
            "min_dist_factor": 0.1,
            "min_radius_factor": 0.05,
            "max_radius_factor": 0.5
        }
        
    def _load_calibration(self, filepath: str):
        """Load calibration data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                cal_data = json.load(f)
            self.calibrated_um_per_px = cal_data.get("um_per_px")
            if self.calibrated_um_per_px and self.calibrated_um_per_px > 0:
                self.effective_um_per_px = self.calibrated_um_per_px
                self.operating_mode = "MICRON_CALIBRATED"
                print(f"Loaded calibration: {self.effective_um_per_px:.4f} µm/px")
        except Exception as e:
            print(f"Warning: Could not load calibration - {e}")
            
    def _setup_zones(self):
        """Setup fiber optic zones based on user input or defaults"""
        if self.user_core_diameter_um and self.user_cladding_diameter_um:
            core_r = self.user_core_diameter_um / 2.0
            cladding_r = self.user_cladding_diameter_um / 2.0
            ferrule_r = self.user_ferrule_diameter_um / 2.0
            
            self.zones = [
                FiberZoneDefinition("core", 0, core_r, 3.0, True, (255, 0, 0)),
                FiberZoneDefinition("cladding", core_r, cladding_r, 10.0, True, (0, 255, 0)),
                FiberZoneDefinition("ferrule", cladding_r, ferrule_r, 25.0, True, (0, 0, 255)),
                FiberZoneDefinition("adhesive", ferrule_r, ferrule_r + 15, 50.0, True, (255, 255, 0))
            ]
            
            if not self.effective_um_per_px:
                self.operating_mode = "MICRON_INFERRED"
                print("Operating in MICRON_INFERRED mode")
        else:
            # Default pixel-based zones
            self.zones = [
                FiberZoneDefinition("core", 0, 30, 5, True, (255, 0, 0)),
                FiberZoneDefinition("cladding", 30, 80, 15, True, (0, 255, 0)),
                FiberZoneDefinition("ferrule", 80, 150, 25, True, (0, 0, 255))
            ]
            if not self.effective_um_per_px:
                self.operating_mode = "PIXEL_ONLY"
                print("Operating in PIXEL_ONLY mode")
    
    def preprocess_image_multi_technique(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply multiple preprocessing techniques"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        processed = {
            "original": gray.copy(),
            "bilateral": cv2.bilateralFilter(gray, 9, 75, 75),
            "gaussian": cv2.GaussianBlur(gray, (5, 5), 0),
            "median": cv2.medianBlur(gray, 5),
        }
        
        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed["clahe"] = clahe.apply(processed["bilateral"])
        
        # Histogram equalization
        processed["histeq"] = cv2.equalizeHist(gray)
        
        # Morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        processed["morph_gradient"] = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Top-hat and black-hat
        processed["tophat"] = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        processed["blackhat"] = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        return processed
    
    def find_fiber_center_multi_method(self, processed_images: Dict[str, np.ndarray]) -> Tuple[Optional[Tuple[int, int]], Optional[float], float]:
        """Find fiber center using multiple methods and vote for best result"""
        candidates = []
        confidences = []
        
        # Method 1: Hough Circle Transform on different preprocessed images
        for img_type, img in processed_images.items():
            if img_type in ["clahe", "bilateral", "gaussian"]:
                for dp in self.hough_params["dp_values"]:
                    for p1 in self.hough_params["param1_values"]:
                        for p2 in self.hough_params["param2_values"]:
                            center, radius, conf = self._hough_circle_detect(img, dp, p1, p2)
                            if center and radius:
                                candidates.append((center, radius, img_type))
                                confidences.append(conf)
        
        # Method 2: Contour-based detection
        for img_type in ["clahe", "bilateral"]:
            img = processed_images[img_type]
            center, radius, conf = self._contour_based_detect(img)
            if center and radius:
                candidates.append((center, radius, f"contour_{img_type}"))
                confidences.append(conf)
        
        # Method 3: Gradient-based center detection
        if "morph_gradient" in processed_images:
            center, radius, conf = self._gradient_based_detect(processed_images["morph_gradient"])
            if center and radius:
                candidates.append((center, radius, "gradient"))
                confidences.append(conf)
        
        # Vote for best result
        if not candidates:
            print("Warning: No fiber detected by any method")
            h, w = list(processed_images.values())[0].shape[:2]
            return (w//2, h//2), None, 0.0
            
        # Cluster similar results and choose the one with highest confidence
        best_idx = np.argmax(confidences)
        best_center, best_radius, best_method = candidates[best_idx]
        
        print(f"Best detection: {best_method} (confidence: {confidences[best_idx]:.2f})")
        return best_center, best_radius, confidences[best_idx]
    
    def _hough_circle_detect(self, image: np.ndarray, dp: float, param1: int, param2: int) -> Tuple[Optional[Tuple[int, int]], Optional[float], float]:
        """Detect circles using Hough transform"""
        edges = cv2.Canny(image, param1//2, param1)
        h, w = image.shape[:2]
        min_dim = min(h, w)
        
        circles = cv2.HoughCircles(
            edges, 
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=int(min_dim * self.hough_params["min_dist_factor"]),
            param1=param1,
            param2=param2,
            minRadius=int(min_dim * self.hough_params["min_radius_factor"]),
            maxRadius=int(min_dim * self.hough_params["max_radius_factor"])
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Choose the circle with maximum accumulator value (strongest)
            best_circle = circles[0][0]  # Take first circle (highest votes)
            center = (int(best_circle[0]), int(best_circle[1]))
            radius = float(best_circle[2])
            
            # Calculate confidence based on edge strength along the circle
            confidence = self._calculate_circle_confidence(edges, center, radius)
            return center, radius, confidence
            
        return None, None, 0.0
    
    def _contour_based_detect(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[float], float]:
        """Detect fiber using contour analysis"""
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 21, 5)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, 0.0
            
        # Find the most circular contour
        best_contour = None
        best_circularity = 0
        best_radius = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Skip small contours
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > best_circularity and circularity > 0.7:
                best_contour = contour
                best_circularity = circularity
                (x, y), radius = cv2.minEnclosingCircle(contour)
                best_radius = radius
                
        if best_contour is not None:
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy), best_radius, best_circularity
                
        return None, None, 0.0
    
    def _gradient_based_detect(self, gradient_image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[float], float]:
        """Detect fiber center using gradient information"""
        # Threshold the gradient image
        _, binary = cv2.threshold(gradient_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply Hough on the gradient
        return self._hough_circle_detect(gradient_image, 1.2, 70, 40)
    
    def _calculate_circle_confidence(self, edge_image: np.ndarray, center: Tuple[int, int], radius: float) -> float:
        """Calculate confidence score for detected circle"""
        if radius <= 0:
            return 0.0
            
        # Sample points along the circle
        num_samples = int(2 * np.pi * radius)
        num_samples = min(max(num_samples, 50), 360)
        
        edge_count = 0
        for i in range(num_samples):
            angle = 2 * np.pi * i / num_samples
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            
            if 0 <= x < edge_image.shape[1] and 0 <= y < edge_image.shape[0]:
                if edge_image[y, x] > 0:
                    edge_count += 1
                    
        return edge_count / num_samples
    
    def detect_defects_multi_algorithm(self, processed_images: Dict[str, np.ndarray], 
                                     center: Tuple[int, int], radius: float) -> Dict[str, np.ndarray]:
        """Apply multiple defect detection algorithms"""
        results = {}
        
        # 1. DO2MR with multiple parameters
        do2mr_combined = np.zeros_like(list(processed_images.values())[0], dtype=np.uint8)
        for img_type in ["clahe", "bilateral", "gaussian"]:
            if img_type in processed_images:
                for kernel_size in self.params.do2mr_kernel_sizes:
                    for gamma in self.params.do2mr_gamma_values:
                        mask = self._do2mr_detection(processed_images[img_type], kernel_size, gamma)
                        do2mr_combined = cv2.bitwise_or(do2mr_combined, mask)
        
        results["do2mr"] = self._clean_mask(do2mr_combined)
        
        # 2. LEI for scratches with multiple parameters
        lei_combined = np.zeros_like(list(processed_images.values())[0], dtype=np.uint8)
        for img_type in ["clahe", "bilateral"]:
            if img_type in processed_images:
                for kernel_len in self.params.lei_kernel_lengths:
                    for angle_step in self.params.lei_angle_steps:
                        mask = self._lei_detection(processed_images[img_type], kernel_len, angle_step)
                        lei_combined = cv2.bitwise_or(lei_combined, mask)
        
        results["lei"] = self._clean_mask(lei_combined)
        
        # 3. Canny edge-based detection
        canny_combined = np.zeros_like(list(processed_images.values())[0], dtype=np.uint8)
        for img_type in ["clahe", "gaussian"]:
            if img_type in processed_images:
                for (low, high) in self.params.canny_thresholds:
                    edges = cv2.Canny(processed_images[img_type], low, high)
                    # Close edges to form regions
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
                    canny_combined = cv2.bitwise_or(canny_combined, closed)
        
        results["canny"] = self._clean_mask(canny_combined)
        
        # 4. Adaptive threshold-based detection
        adaptive_combined = np.zeros_like(list(processed_images.values())[0], dtype=np.uint8)
        for img_type in ["clahe", "bilateral"]:
            if img_type in processed_images:
                for block_size in self.params.adaptive_block_sizes:
                    for C in self.params.adaptive_C_values:
                        binary = cv2.adaptiveThreshold(processed_images[img_type], 255,
                                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY_INV, block_size, C)
                        adaptive_combined = cv2.bitwise_or(adaptive_combined, binary)
        
        results["adaptive"] = self._clean_mask(adaptive_combined)
        
        # 5. Watershed segmentation
        if "clahe" in processed_images:
            watershed_mask = self._watershed_detection(processed_images["clahe"])
            results["watershed"] = watershed_mask
        
        # 6. Combine all methods with voting
        all_masks = list(results.values())
        vote_mask = np.zeros_like(all_masks[0], dtype=np.float32)
        
        for mask in all_masks:
            vote_mask += mask.astype(np.float32) / 255.0
            
        # Threshold based on minimum votes
        min_votes = len(all_masks) * 0.3  # At least 30% of methods must agree
        final_mask = (vote_mask >= min_votes).astype(np.uint8) * 255
        
        results["combined"] = self._clean_mask(final_mask)
        
        return results
    
    def _do2mr_detection(self, image: np.ndarray, kernel_size: Tuple[int, int], gamma: float) -> np.ndarray:
        """DO2MR detection with specific parameters"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        
        # Min-max filtering
        img_max = cv2.dilate(image, kernel)
        img_min = cv2.erode(image, kernel)
        
        # Residual
        residual = cv2.absdiff(img_max, img_min)
        
        # Adaptive thresholding
        residual_filtered = cv2.medianBlur(residual, 5)
        mean_val = np.mean(residual_filtered)
        std_val = np.std(residual_filtered)
        
        threshold = mean_val + gamma * std_val
        _, binary = cv2.threshold(residual_filtered, threshold, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def _lei_detection(self, image: np.ndarray, kernel_length: int, angle_step: int) -> np.ndarray:
        """LEI scratch detection with specific parameters"""
        scratch_strength = np.zeros_like(image, dtype=np.float32)
        
        for angle in range(0, 180, angle_step):
            angle_rad = np.deg2rad(angle)
            
            # Create linear kernel
            kernel_points = []
            for i in range(-kernel_length//2, kernel_length//2 + 1):
                if i == 0:
                    continue
                x = int(round(i * np.cos(angle_rad)))
                y = int(round(i * np.sin(angle_rad)))
                if (x, y) not in kernel_points:
                    kernel_points.append((x, y))
            
            if kernel_points:
                response = self._apply_linear_detector(image, kernel_points)
                scratch_strength = np.maximum(scratch_strength, response)
        
        # Normalize and threshold
        if scratch_strength.max() > 0:
            scratch_norm = cv2.normalize(scratch_strength, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            _, scratch_mask = cv2.threshold(scratch_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return scratch_mask
        
        return np.zeros_like(image, dtype=np.uint8)
    
    def _apply_linear_detector(self, image: np.ndarray, kernel_points: List[Tuple[int, int]]) -> np.ndarray:
        """Apply linear detector for scratch detection"""
        h, w = image.shape
        response = np.zeros_like(image, dtype=np.float32)
        
        # Pad image
        max_offset = max(max(abs(dx), abs(dy)) for dx, dy in kernel_points)
        padded = cv2.copyMakeBorder(image, max_offset, max_offset, max_offset, max_offset, cv2.BORDER_REFLECT)
        
        for y in range(h):
            for x in range(w):
                y_pad, x_pad = y + max_offset, x + max_offset
                
                # Calculate line average
                line_vals = []
                for dx, dy in kernel_points:
                    line_vals.append(float(padded[y_pad + dy, x_pad + dx]))
                
                if line_vals:
                    avg_line = np.mean(line_vals)
                    center_val = float(padded[y_pad, x_pad])
                    response[y, x] = max(0, avg_line - center_val)
        
        return response
    
    def _watershed_detection(self, image: np.ndarray) -> np.ndarray:
        """Watershed-based defect detection"""
        # Find sure background
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(image, kernel, iterations=3)
        
        # Find sure foreground
        dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        
        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Convert to 3-channel for watershed
        img_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_3ch, markers)
        
        # Create mask from markers
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[markers == -1] = 255
        
        return mask
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean up a binary mask"""
        # Remove small objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill small holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Remove tiny components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        
        min_area = self.params.do2mr_min_area
        cleaned_final = np.zeros_like(mask)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_final[labels == i] = 255
                
        return cleaned_final
    
    def create_zone_masks(self, image_shape: Tuple[int, int], center: Tuple[int, int], 
                         radius: Optional[float]) -> Dict[str, np.ndarray]:
        """Create masks for different fiber zones"""
        h, w = image_shape[:2]
        masks = {}
        
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        
        # Determine scale
        scale = 1.0
        if self.effective_um_per_px:
            scale = self.effective_um_per_px
        elif radius and self.operating_mode == "MICRON_INFERRED" and self.user_cladding_diameter_um:
            # Infer scale from detected radius
            scale = (self.user_cladding_diameter_um / 2.0) / radius
            self.effective_um_per_px = scale
            
        # Create masks for each zone
        for zone in self.zones:
            if scale == 1.0:  # Pixel mode
                r_min_px = zone.r_min_um
                r_max_px = zone.r_max_um
            else:  # Micron mode
                r_min_px = zone.r_min_um / scale
                r_max_px = zone.r_max_um / scale
                
            mask = ((dist_from_center >= r_min_px) & (dist_from_center < r_max_px)).astype(np.uint8) * 255
            masks[zone.name] = mask
            
        return masks
    
    def classify_defects(self, defect_masks: Dict[str, np.ndarray], 
                        zone_masks: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Classify detected defects by zone and type"""
        defects = []
        
        # Get combined defect mask
        combined_mask = defect_masks.get("combined", np.zeros_like(list(defect_masks.values())[0]))
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        
        for i in range(1, num_labels):
            area_px = stats[i, cv2.CC_STAT_AREA]
            if area_px < self.params.do2mr_min_area:
                continue
                
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # Determine zone
            zone_name = "unknown"
            for zone, mask in zone_masks.items():
                if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx] > 0:
                    zone_name = zone
                    break
            
            # Determine defect type based on shape
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Check if it's a scratch (elongated)
            is_scratch = False
            if defect_masks.get("lei") is not None:
                lei_region = defect_masks["lei"][y:y+h, x:x+w]
                label_region = (labels[y:y+h, x:x+w] == i)
                overlap = np.sum(lei_region > 0) / np.sum(label_region) if np.sum(label_region) > 0 else 0
                is_scratch = overlap > 0.5 or aspect_ratio > 3.0 or aspect_ratio < 0.33
            
            # Calculate size
            if self.effective_um_per_px:
                if is_scratch:
                    # For scratches, calculate length and width
                    contours, _ = cv2.findContours((labels == i).astype(np.uint8), 
                                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        rect = cv2.minAreaRect(contours[0])
                        width_px, length_px = sorted(rect[1])
                        length_um = length_px * self.effective_um_per_px
                        width_um = width_px * self.effective_um_per_px
                        size_info = {"length_um": round(length_um, 2), "width_um": round(width_um, 2)}
                else:
                    # For dig/pit, use equivalent diameter
                    diameter_px = np.sqrt(4 * area_px / np.pi)
                    diameter_um = diameter_px * self.effective_um_per_px
                    size_info = {"diameter_um": round(diameter_um, 2)}
            else:
                # Pixel mode
                if is_scratch:
                    size_info = {"length_px": max(w, h), "width_px": min(w, h)}
                else:
                    size_info = {"diameter_px": round(np.sqrt(4 * area_px / np.pi), 2)}
            
            defect_type = "scratch" if is_scratch else "dig"
            
            # Count votes from different methods
            votes = 0
            for method_name, method_mask in defect_masks.items():
                if method_name != "combined" and method_mask[cy, cx] > 0:
                    votes += 1
            
            defects.append({
                "type": defect_type,
                "zone": zone_name,
                "area_px": area_px,
                "cx_px": cx,
                "cy_px": cy,
                "bbox": (x, y, w, h),
                "aspect_ratio": round(aspect_ratio, 2),
                "detection_votes": votes,
                **size_info
            })
        
        return pd.DataFrame(defects)
    
    def apply_pass_fail_criteria(self, defects_df: pd.DataFrame) -> Tuple[str, List[str]]:
        """Apply pass/fail criteria based on zone rules"""
        if not self.effective_um_per_px:
            # Pixel mode - can't apply micron-based rules
            if len(defects_df) == 0:
                return "PASS", []
            else:
                return "REVIEW", [f"Found {len(defects_df)} defects - manual review needed (pixel mode)"]
        
        status = "PASS"
        failures = []
        
        for zone in self.zones:
            zone_defects = defects_df[defects_df["zone"] == zone.name]
            
            if not zone.defects_allowed and len(zone_defects) > 0:
                status = "FAIL"
                failures.append(f"{zone.name}: No defects allowed, found {len(zone_defects)}")
                continue
                
            for _, defect in zone_defects.iterrows():
                # Get defect size
                if defect["type"] == "scratch":
                    size = defect.get("length_um", 0)
                else:
                    size = defect.get("diameter_um", 0)
                    
                if size > zone.max_defect_um:
                    status = "FAIL"
                    failures.append(f"{zone.name}: {defect['type']} size {size:.1f}µm exceeds limit {zone.max_defect_um}µm")
        
        return status, failures
    
    def inspect_single_image(self, image_path: str) -> Dict:
        """Inspect a single fiber optic image"""
        # Reset per-image values
        if self.operating_mode == "MICRON_INFERRED":
            self.effective_um_per_px = None
            
        # Load image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return {
                "image_path": image_path,
                "status": "ERROR",
                "failure_reasons": [f"Failed to load image: {image_path}"],
                "defect_count": 0
            }
        
        # Preprocess with multiple techniques
        processed = self.preprocess_image_multi_technique(image_bgr)
        
        # Find fiber center using multiple methods
        center, radius, confidence = self.find_fiber_center_multi_method(processed)
        
        if center is None:
            return {
                "image_path": image_path,
                "status": "ERROR", 
                "failure_reasons": ["Failed to detect fiber center"],
                "defect_count": 0
            }
        
        # Create zone masks
        zone_masks = self.create_zone_masks(image_bgr.shape[:2], center, radius)
        
        # Detect defects using multiple algorithms
        defect_masks = self.detect_defects_multi_algorithm(processed, center, radius)
        
        # Classify defects
        defects_df = self.classify_defects(defect_masks, zone_masks)
        
        # Apply pass/fail criteria
        status, failures = self.apply_pass_fail_criteria(defects_df)
        
        return {
            "image_path": image_path,
            "status": status,
            "failure_reasons": failures,
            "defect_count": len(defects_df),
            "defects": defects_df.to_dict('records') if not defects_df.empty else [],
            "fiber_center": center,
            "fiber_radius": radius,
            "detection_confidence": confidence,
            "operating_mode": self.operating_mode,
            "effective_um_per_px": self.effective_um_per_px,
            "defect_masks": defect_masks,
            "zone_masks": zone_masks
        }
    
    def inspect_batch(self, image_paths: List[str], max_workers: int = 4) -> pd.DataFrame:
        """Inspect multiple images in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(self.inspect_single_image, path): path 
                            for path in image_paths}
            
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Completed: {Path(path).name} - Status: {result['status']}")
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    results.append({
                        "image_path": path,
                        "status": "ERROR",
                        "failure_reasons": [str(e)],
                        "defect_count": -1
                    })
        
        return pd.DataFrame(results)
    
    def visualize_results(self, image_path: str, results: Dict, save_path: Optional[str] = None):
        """Visualize inspection results"""
        image = cv2.imread(image_path)
        if image is None:
            return
            
        vis = image.copy()
        
        # Draw zone circles
        center = results.get("fiber_center")
        zone_masks = results.get("zone_masks", {})
        
        if center and zone_masks:
            # Draw zone boundaries
            for zone in self.zones:
                if zone.name in zone_masks:
                    # Find zone boundary
                    mask = zone_masks[zone.name]
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cv2.drawContours(vis, contours, -1, zone.color_bgr, 2)
        
        # Overlay defect masks
        defect_masks = results.get("defect_masks", {})
        if "combined" in defect_masks:
            mask = defect_masks["combined"]
            # Create colored overlay - properly handle channel mismatch
            overlay = np.zeros_like(vis)
            overlay[mask > 0] = [0, 255, 255]  # Yellow for defects
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        # Draw defect bounding boxes
        for defect in results.get("defects", []):
            x, y, w, h = defect["bbox"]
            color = (0, 0, 255) if defect["type"] == "scratch" else (255, 0, 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"{defect['type'][:3]}"
            cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add status text
        status = results.get("status", "N/A")
        color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
        cv2.putText(vis, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(vis, f"Defects: {results.get('defect_count', 0)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add mode info
        mode = results.get("operating_mode", "Unknown")
        cv2.putText(vis, f"Mode: {mode}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if save_path:
            # Create comparison figure
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f"{Path(image_path).name} - Status: {status}", fontsize=16)
            
            # Original image
            axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title("Original")
            axes[0, 0].axis('off')
            
            # Result visualization
            axes[0, 1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title("Detected Defects")
            axes[0, 1].axis('off')
            
            # Combined defect mask
            if "combined" in defect_masks:
                axes[0, 2].imshow(defect_masks["combined"], cmap='gray')
                axes[0, 2].set_title("Combined Defect Mask")
                axes[0, 2].axis('off')
            
            # Individual method results
            methods = ["do2mr", "lei", "canny"]
            for i, method in enumerate(methods):
                if method in defect_masks:
                    axes[1, i].imshow(defect_masks[method], cmap='gray')
                    axes[1, i].set_title(f"{method.upper()} Detection")
                    axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            cv2.imshow("Inspection Result", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main():
    """Main execution function"""
    print("Advanced Fiber Optic End Face Inspector")
    print("=" * 50)
    
    # Get user input for fiber specifications
    user_core_diameter_um = None
    user_cladding_diameter_um = None
    
    if input("Provide fiber specifications? (y/n, default: n): ").lower() == 'y':
        try:
            core_input = input("Core diameter (µm, e.g., 9 for single-mode, 50/62.5 for multi-mode): ").strip()
            if core_input:
                user_core_diameter_um = float(core_input)
                
            clad_input = input("Cladding diameter (µm, typically 125): ").strip()
            if clad_input:
                user_cladding_diameter_um = float(clad_input)
        except ValueError:
            print("Invalid input. Proceeding without specifications.")
    
    # Check for calibration file
    calibration_file = None
    if Path("calibration.json").exists():
        if input("Use existing calibration file? (y/n, default: y): ").lower() != 'n':
            calibration_file = "calibration.json"
    
    # Initialize inspector
    inspector = AdvancedFiberOpticInspector(
        user_core_diameter_um=user_core_diameter_um,
        user_cladding_diameter_um=user_cladding_diameter_um,
        calibration_file=calibration_file
    )
    
    # Get input path
    input_path = input("Enter image directory or file path: ").strip()
    if not input_path:
        print("No input provided.")
        return
    
    # Collect image paths
    image_paths = []
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_file():
        image_paths.append(str(input_path_obj))
    elif input_path_obj.is_dir():
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
            image_paths.extend(glob.glob(str(input_path_obj / ext)))
    else:
        print(f"Invalid path: {input_path}")
        return
    
    if not image_paths:
        print("No images found.")
        return
    
    print(f"\nFound {len(image_paths)} image(s)")
    
    # Create output directory
    output_dir = Path("inspection_results")
    output_dir.mkdir(exist_ok=True)
    
    # Process images
    print("\nProcessing images...")
    results_df = inspector.inspect_batch(image_paths, max_workers=4)
    
    # Save results
    summary_path = output_dir / "inspection_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    for _, row in results_df.iterrows():
        if row["status"] != "ERROR":
            image_path = row["image_path"]
            image_name = Path(image_path).stem
            
            # Recreate results dict for visualization
            results = {
                "fiber_center": row.get("fiber_center"),
                "fiber_radius": row.get("fiber_radius"),
                "status": row["status"],
                "defect_count": row["defect_count"],
                "defects": row.get("defects", []),
                "operating_mode": row.get("operating_mode"),
                "defect_masks": row.get("defect_masks", {}),
                "zone_masks": row.get("zone_masks", {})
            }
            
            vis_path = output_dir / f"{image_name}_result.png"
            
            # Re-run inspection to get masks (since they're not stored in DataFrame)
            full_results = inspector.inspect_single_image(image_path)
            inspector.visualize_results(image_path, full_results, str(vis_path))
            
    print("\nInspection complete!")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total images: {len(results_df)}")
    print(f"Passed: {len(results_df[results_df['status'] == 'PASS'])}")
    print(f"Failed: {len(results_df[results_df['status'] == 'FAIL'])}")
    print(f"Errors: {len(results_df[results_df['status'] == 'ERROR'])}")
    
    # Show defect statistics
    all_defects = []
    for _, row in results_df.iterrows():
        if isinstance(row.get("defects"), list):
            all_defects.extend(row["defects"])
    
    if all_defects:
        defects_df = pd.DataFrame(all_defects)
        print(f"\nTotal defects found: {len(defects_df)}")
        print("\nDefects by type:")
        print(defects_df["type"].value_counts())
        print("\nDefects by zone:")
        print(defects_df["zone"].value_counts())

if __name__ == "__main__":
    main()