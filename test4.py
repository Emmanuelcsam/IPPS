import cv2
import numpy as np
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import glob
import os

class FiberOpticInspector:
    """
    Automated Fiber Optic End Face Defect Detection System
    Improved with user-defined dimensions and batch processing.
    """

    def __init__(self, calibration_file: str = "calibration.json",
                 core_diameter_um: Optional[float] = None,
                 cladding_diameter_um: Optional[float] = None,
                 ferrule_outer_diameter_um: Optional[float] = 250.0, # Typical ferrule contact zone for IEC
                 use_pixel_units: bool = False):
        """Initialize the inspector with calibration data and optional fiber dimensions."""
        self.use_pixel_units = use_pixel_units
        self.um_per_px = 0.0 # Will be loaded or calculated

        if not self.use_pixel_units:
            self.calibration = self._load_calibration(calibration_file)
            self.um_per_px = self.calibration.get("um_per_px")
            if not self.um_per_px:
                print("Warning: um_per_px not found in calibration. Pixel units will be assumed for some calculations if not overridden by user input.")
                # Fallback to a default if not in file, or prompt user, or raise error
                # For this example, we'll allow it to be None and handle it in create_zone_masks
        else:
            print("Operating in pixel units mode. 'um_per_px' will not be used for zone definitions.")
            self.calibration = {} # No calibration needed for pixel mode zone definition

        # Zone definitions (will be updated based on input)
        # Default values based on common SMF like 9/125/250
        # Core: 0 to ~4.5µm radius (9µm diameter)
        # Cladding: ~4.5µm to ~62.5µm radius (125µm diameter)
        # Ferrule/Contact Zone: ~62.5µm to ~125µm radius (250µm diameter - IEC contact zone A)
        self.zones_um = {
            "core": {"r_min": 0, "r_max": 4.5, "max_defect_um": 3, "defects_allowed": True},
            "cladding": {"r_min": 4.5, "r_max": 62.5, "max_defect_um": 10, "defects_allowed": True},
            "ferrule_contact": {"r_min": 62.5, "r_max": 125.0, "max_defect_um": 25, "defects_allowed": True}, # IEC Zone A (Contact) for cleanliness
            "adhesive_bond": {"r_min": 125.0, "r_max": 140.0, "max_defect_um": 50, "defects_allowed": True}, # Example, often less critical
            # Add other zones as per IEC 61300-3-35 specific to fiber type if needed
        }
        self.zones_px = {} # To be populated if using pixel units directly for zones

        if not self.use_pixel_units and core_diameter_um is not None and cladding_diameter_um is not None:
            print(f"Using user-defined dimensions: Core Dia={core_diameter_um}µm, Cladding Dia={cladding_diameter_um}µm")
            core_radius_um = core_diameter_um / 2.0
            cladding_radius_um = cladding_diameter_um / 2.0
            ferrule_contact_radius_um = ferrule_outer_diameter_um / 2.0 # Assuming this is the edge of Zone A (Contact)

            self.zones_um["core"]["r_max"] = core_radius_um
            self.zones_um["cladding"]["r_min"] = core_radius_um
            self.zones_um["cladding"]["r_max"] = cladding_radius_um
            self.zones_um["ferrule_contact"]["r_min"] = cladding_radius_um
            self.zones_um["ferrule_contact"]["r_max"] = ferrule_contact_radius_um
            # Adjust adhesive bond or other outer zones as needed
            self.zones_um["adhesive_bond"]["r_min"] = ferrule_contact_radius_um
            self.zones_um["adhesive_bond"]["r_max"] = ferrule_contact_radius_um + 15 # Example for adhesive
        elif self.use_pixel_units:
            # If using pixel units, user might need to provide these if not relying on defaults
            # For now, let's use placeholder pixel radii if `use_pixel_units` is true and no specific pixel inputs are handled here.
            # A more robust implementation would ask for pixel radii for zones if use_pixel_units.
            # Or, it could attempt to find the fiber and then define zones as fractions of the detected fiber radius.
            print("Zone definitions will rely on pixel values. Ensure 'find_fiber_center_and_radii' is robust.")
            # These would ideally be set after `find_fiber_center_and_radii` is called
            self.zones_px = {
                "core": {"r_min_px": 0, "r_max_px": 30, "max_defect_px": 5, "defects_allowed": True}, # Example pixel values
                "cladding": {"r_min_px": 30, "r_max_px": 80, "max_defect_px": 15, "defects_allowed": True},
                "ferrule_contact": {"r_min_px": 80, "r_max_px": 150, "max_defect_px": 25, "defects_allowed": True},
            }


        # Detection parameters (can be tuned further)
        self.do2mr_params = {
            "kernel_size": (15, 15),
            "gamma": 3.0,
            "min_area_px": 20 # Reduced min area for potentially smaller defects
        }

        self.lei_params = {
            "kernel_size": 15,
            "angles": np.arange(0, 180, 10), # Increased angular resolution
            "threshold_factor": 2.5 # Adjusted for sensitivity
        }
        self.hough_params = { # Tunable parameters for HoughCircles
            "dp": 1.2, # Inverse ratio of accumulator resolution
            "param1": 70, # Higher threshold for Canny edge detector
            "param2": 40, # Accumulator threshold for circle centers
            "minDistFactor": 1/8.0, # min_dist = image.shape[0] * minDistFactor
            "minRadiusFactor": 1/10.0, # min_radius = image.shape[0] * minRadiusFactor
            "maxRadiusFactor": 1/2.0   # max_radius = image.shape[0] * maxRadiusFactor
        }

    def _load_calibration(self, filepath: str) -> Dict:
        """Load calibration data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                cal_data = json.load(f)
                if "um_per_px" not in cal_data or cal_data["um_per_px"] == 0:
                    print(f"Error: 'um_per_px' is missing or zero in {filepath}.")
                    raise ValueError("Invalid um_per_px in calibration file.")
                return cal_data
        except FileNotFoundError:
            print(f"Calibration file '{filepath}' not found. Please calibrate or provide correct path.")
            raise
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {filepath}.")
            raise
        except ValueError as e:
            print(e)
            raise
            
    def update_pixel_zones(self, fiber_radius_px: float):
        """Update zones_px based on a detected fiber radius if operating in pixel_units mode."""
        if self.use_pixel_units:
            # Example: Define core and cladding as fractions of the detected radius
            # These fractions would ideally come from typical fiber geometry knowledge
            # For a 9/125 fiber, core_radius / cladding_radius is roughly 4.5 / 62.5 = 0.072
            # cladding_radius is the reference here.
            self.zones_px["core"]["r_min_px"] = 0
            self.zones_px["core"]["r_max_px"] = int(fiber_radius_px * 0.072 * (62.5 / (62.5)) ) # Placeholder, needs robust logic if cladding is not fiber_radius_px
            self.zones_px["cladding"]["r_min_px"] = self.zones_px["core"]["r_max_px"]
            self.zones_px["cladding"]["r_max_px"] = int(fiber_radius_px) # Assuming detected circle is cladding
            self.zones_px["ferrule_contact"]["r_min_px"] = int(fiber_radius_px)
            self.zones_px["ferrule_contact"]["r_max_px"] = int(fiber_radius_px * (125.0/62.5)) # Outer ferrule contact based on 125um fiber

            print(f"Updated pixel zones based on detected fiber radius ({fiber_radius_px:.2f}px):")
            for zone_name, params in self.zones_px.items():
                 print(f"  {zone_name}: min_px={params['r_min_px']}, max_px={params['r_max_px']}")


    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhanced Denoising - consider bilateral filter for edge preservation
        # denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # CLAHE for localized contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(denoised)
        
        return enhanced_gray, denoised # Return enhanced for some detections, original gray/denoised for others

    def find_fiber_center_and_radius(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """Find the center and radius of the fiber using Hough Circle Transform with tunable params."""
        # Apply edge detection - consider adaptive Canny or different parameters
        # edges = cv2.Canny(image, 50, 150) # Original
        edges = cv2.Canny(image, self.hough_params['param1'] / 2, self.hough_params['param1'])


        min_img_dim = min(image.shape[0], image.shape[1])
        min_dist = int(min_img_dim * self.hough_params['minDistFactor'])
        min_radius = int(min_img_dim * self.hough_params['minRadiusFactor'])
        max_radius = int(min_img_dim * self.hough_params['maxRadiusFactor'])
        
        # Ensure minRadius is at least 1
        min_radius = max(1, min_radius)
        if max_radius <= min_radius: # Ensure maxRadius is greater than minRadius
            max_radius = min_radius + 10 # or some other logic

        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_params['dp'],
            minDist=min_dist,
            param1=self.hough_params['param1'],
            param2=self.hough_params['param2'],
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Select the circle with the largest radius, or one that's most central, etc.
            # For simplicity, taking the first one, assuming it's the most prominent.
            # A more robust selection might involve checking circularity or proximity to image center.
            best_circle = circles[0][np.argmax(circles[0, :, 2])] # Largest radius
            center = (int(best_circle[0]), int(best_circle[1]))
            radius = int(best_circle[2])
            
            # Sanity check: radius should be plausible
            if radius < 5: # Or some other threshold
                 print(f"Warning: Detected fiber radius {radius}px is very small. Check Hough parameters or image quality.")
                 # Fallback or raise error
            
            return center, radius
        else:
            print("Warning: No fiber outline detected by HoughCircles. Falling back to image center.")
            # Fallback: Use image center, but radius is unknown. This needs careful handling.
            # For a fallback, we might try to estimate radius differently or use predefined pixel zones.
            return (image.shape[1]//2, image.shape[0]//2), None


    def create_zone_masks(self, image_shape: Tuple, center: Tuple[int, int], fiber_radius_px: Optional[int]) -> Dict[str, np.ndarray]:
        """Create masks for different zones (core, cladding, ferrule)."""
        masks = {}
        height, width = image_shape[:2]
        Y, X = np.ogrid[:height, :width]
        dist_from_center_sq = (X - center[0])**2 + (Y - center[1])**2 # Use squared distance for efficiency

        if self.use_pixel_units:
            if fiber_radius_px is None and not self.zones_px: # Need a radius if dynamically setting pixel zones
                print("Error: Operating in pixel units but no fiber radius detected and no predefined pixel zones.")
                # Create a dummy full mask to avoid crashing, but this is not ideal
                # You might want to raise an error or have a very robust fallback.
                full_mask = np.ones(image_shape[:2], dtype=np.uint8) * 255
                return {"error_zone": full_mask}

            if fiber_radius_px is not None and not self.zones_px.get("core", {}).get("r_max_px"): # Check if zones_px needs update
                 self.update_pixel_zones(fiber_radius_px)

            current_zones_def = self.zones_px
            unit_suffix = "_px"
            scale = 1.0 # Already in pixels
        else:
            if self.um_per_px is None or self.um_per_px == 0:
                print("Error: um_per_px is not available for micron-based zone creation.")
                # Fallback to trying pixel units if possible, or raise an error
                # This state should ideally be caught earlier.
                full_mask = np.ones(image_shape[:2], dtype=np.uint8) * 255
                return {"error_zone_um_per_px": full_mask}
            current_zones_def = self.zones_um
            unit_suffix = ""
            scale = self.um_per_px

        for zone_name, zone_params in current_zones_def.items():
            r_min_val = zone_params.get(f"r_min{unit_suffix}")
            r_max_val = zone_params.get(f"r_max{unit_suffix}")

            if r_min_val is None or r_max_val is None:
                print(f"Warning: Zone {zone_name} is missing r_min{unit_suffix} or r_max{unit_suffix}. Skipping.")
                continue

            r_min_px_sq = (r_min_val / scale if not self.use_pixel_units else r_min_val)**2
            r_max_px_sq = (r_max_val / scale if not self.use_pixel_units else r_max_val)**2
            
            zone_mask = (dist_from_center_sq >= r_min_px_sq) & (dist_from_center_sq < r_max_px_sq)
            masks[zone_name] = zone_mask.astype(np.uint8) * 255 # Binary mask 0 or 255
        
        return masks

    def detect_region_defects_do2mr(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.do2mr_params["kernel_size"])
        img_max = cv2.dilate(image, kernel)
        img_min = cv2.erode(image, kernel)
        residual = cv2.absdiff(img_max, img_min) # Using absdiff for robustness
        
        residual_filtered = cv2.medianBlur(residual, 5) # Slightly larger median filter
        
        # Try Otsu's thresholding on the residual image for more adaptivity
        # _, binary_mask_otsu = cv2.threshold(residual_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Or stick with sigma-based if it works better after tuning
        mean_val = np.mean(residual_filtered)
        std_val = np.std(residual_filtered)
        threshold_val = mean_val + self.do2mr_params["gamma"] * std_val
        _, binary_mask_sigma = cv2.threshold(residual_filtered, threshold_val, 255, cv2.THRESH_BINARY)

        binary_mask = binary_mask_sigma # Choose the best or combine

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) # Elliptical kernel
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open, iterations=2) # More opening
        
        n_labels, labeled = cv2.connectedComponents(binary_mask)
        return binary_mask, labeled

    def detect_scratches_lei(self, image: np.ndarray) -> np.ndarray:
        # Original image for LEI, not CLAHE enhanced, as LEI enhances lines itself
        # Enhanced contrast might make non-scratches appear linear
        # gray_for_lei = cv2.GaussianBlur(image, (3,3),0) # Minimal blur
        gray_for_lei = image # Or use the original preprocessed gray without CLAHE

        scratch_strength = np.zeros_like(gray_for_lei, dtype=np.float32)
        kernel_length = self.lei_params["kernel_size"]

        for angle in self.lei_params["angles"]:
            angle_rad = np.deg2rad(angle)
            kernel_points = []
            for i in range(-kernel_length//2, kernel_length//2 + 1):
                if i == 0 : continue # Avoid zero offset point if it complicates kernel logic
                x = int(round(i * np.cos(angle_rad))) # Round for better pixel alignment
                y = int(round(i * np.sin(angle_rad)))
                kernel_points.append((x, y))
            
            if not kernel_points: continue

            response = self._apply_linear_detector_refined(gray_for_lei, kernel_points)
            scratch_strength = np.maximum(scratch_strength, response)
        
        # Normalize scratch_strength before thresholding if responses vary widely
        if scratch_strength.max() > 0:
             scratch_strength_norm = cv2.normalize(scratch_strength, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            scratch_strength_norm = scratch_strength.astype(np.uint8)

        # Adaptive thresholding or Otsu on the strength map
        _, scratch_mask_otsu = cv2.threshold(scratch_strength_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Original sigma based thresholding
        # mean_strength = np.mean(scratch_strength_norm)
        # std_strength = np.std(scratch_strength_norm)
        # threshold = mean_strength + self.lei_params["threshold_factor"] * std_strength
        # _, scratch_mask_sigma = cv2.threshold(scratch_strength_norm, threshold, 255, cv2.THRESH_BINARY)
        
        scratch_mask = scratch_mask_otsu # Choose or combine

        # Morphological operations to refine scratches: close gaps, remove noise
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length // 2, 1)) # Line-like kernel
        # Rotate kernel or apply for multiple orientations if needed, but LEI already does orientation
        scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
        
        return scratch_mask

    def _apply_linear_detector_refined(self, image: np.ndarray, kernel_points: List[Tuple[int, int]]) -> np.ndarray:
        """Refined linear detector application."""
        height, width = image.shape
        response = np.zeros_like(image, dtype=np.float32)
        
        # Determine max offset for padding
        max_offset = 0
        if kernel_points:
            max_offset = max(max(abs(dx), abs(dy)) for dx, dy in kernel_points)
        
        padded = cv2.copyMakeBorder(image, max_offset, max_offset, max_offset, max_offset, cv2.BORDER_REFLECT)

        for r in range(height):
            for c in range(width):
                center_pixel_val = float(padded[r + max_offset, c + max_offset])
                
                # For scratches, expect them to be darker or lighter than immediate surroundings
                # The original LEI paper implies scratches are brighter due to diffuse reflection
                # The provided code's _apply_linear_detector implied darker centers.
                # Let's make it more generic: difference from local average along the line
                
                line_pixels = []
                for dx, dy in kernel_points:
                    line_pixels.append(float(padded[r + max_offset + dy, c + max_offset + dx]))
                
                if not line_pixels: continue

                mean_line_val = np.mean(line_pixels)
                # A scratch might be consistently different from its local line neighborhood mean
                # Or the center pixel is different from the line mean
                # If scratch is a dark line on light background: center_pixel_val < mean_line_val -> mean_line_val - center_pixel_val
                # If scratch is a light line on dark background: center_pixel_val > mean_line_val -> center_pixel_val - mean_line_val
                # The research paper's LEI: "Scratch strength is large if the filter is aligned within a scratch"
                # s_theta(x,y) = 2 * f_theta_red(x,y) - f_theta_gray(x,y)
                # where red is the central part of the line, gray is surrounding.
                # Let's simplify to: contrast of center pixel to its linear neighborhood.
                # For simplicity, assuming scratches are darker (can be inverted if they are brighter)
                
                # A simple approach: Sum of absolute differences from center pixel
                # current_response = sum(abs(center_pixel_val - lp_val) for lp_val in line_pixels)
                
                # Alternative: how much the center differs from the mean of its linear extension
                # response[r, c] = abs(center_pixel_val - mean_line_val)

                # Let's try to implement a simplified version of center vs surround along the line
                # Assuming kernel_points are ordered from one end of the line to the other
                # Central part of the kernel vs outer parts
                center_kernel_region_len = len(kernel_points) // 3
                center_kp = kernel_points[len(kernel_points)//2 - center_kernel_region_len//2 : len(kernel_points)//2 + center_kernel_region_len//2 +1]
                surround_kp = [kp for i, kp in enumerate(kernel_points) if not (len(kernel_points)//2 - center_kernel_region_len//2 <= i <= len(kernel_points)//2 + center_kernel_region_len//2) ]

                center_vals = [float(padded[r + max_offset + dy, c + max_offset + dx]) for dx,dy in center_kp]
                surround_vals = [float(padded[r + max_offset + dy, c + max_offset + dx]) for dx,dy in surround_kp]

                if center_vals and surround_vals:
                    # If scratches are darker, center_mean < surround_mean
                    # response[r,c] = max(0, np.mean(surround_vals) - np.mean(center_vals))
                    # If scratches are brighter (as per Mei et al. LEI description of original paper for their s_theta)
                    response[r,c] = max(0, np.mean(center_vals) - np.mean(surround_vals))
                elif center_vals: # Only center points (short line)
                     response[r,c] = max(0, np.mean(center_vals) - image[r,c]) # if line is brighter
        return response

    def classify_defects(self, labeled_image: np.ndarray, scratch_mask: np.ndarray,
                         zone_masks: Dict[str, np.ndarray]) -> pd.DataFrame:
        defects = []
        props = ["label", "area", "centroid_x", "centroid_y", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "major_axis_length", "minor_axis_length", "orientation"] # For regionprops if used

        # Process region-based defects (from DO2MR)
        num_labels_region, labels_region, stats_region, centroids_region = cv2.connectedComponentsWithStats(
            (labeled_image > 0).astype(np.uint8), connectivity=8
        )

        for i in range(1, num_labels_region): # Skip background label 0
            area_px = stats_region[i, cv2.CC_STAT_AREA]
            if area_px < self.do2mr_params["min_area_px"]:
                continue

            centroid_x_px = int(centroids_region[i][0])
            centroid_y_px = int(centroids_region[i][1])
            x, y, w, h = stats_region[i, cv2.CC_STAT_LEFT], stats_region[i, cv2.CC_STAT_TOP], \
                         stats_region[i, cv2.CC_STAT_WIDTH], stats_region[i, cv2.CC_STAT_HEIGHT]

            zone = "unknown"
            for zone_name, zm in zone_masks.items():
                if zm[centroid_y_px, centroid_x_px] > 0 : # Check if centroid is in the mask
                    zone = zone_name
                    break
            
            # Use um_per_px if available and not in pixel_units mode
            size_val = 0
            if not self.use_pixel_units and self.um_per_px:
                area_um2 = area_px * (self.um_per_px ** 2)
                # Equivalent diameter for digs
                size_val = np.sqrt(4 * area_um2 / np.pi)
                size_unit = "um"
            else:
                size_val = np.sqrt(area_px) # Characteristic size in pixels (e.g. sqrt(area))
                size_unit = "px"

            # More robust dig/scratch classification can be added here
            # For now, assuming DO2MR finds mostly "digs" or non-linear defects
            aspect_ratio = w / h if h > 0 else (w / 0.1 if w > 0 else 1) # Avoid div by zero, handle flat lines

            defect_type = "dig"
            if aspect_ratio > 4 or aspect_ratio < 0.25: # Heuristic for scratch-like from DO2MR output
                 defect_type = "region_scratch_like"


            defects.append({
                "type": defect_type, "zone": zone, f"size ({size_unit})": round(size_val,2),
                "area_px": area_px, "centroid_x_px": centroid_x_px, "centroid_y_px": centroid_y_px,
                "bbox_x": x, "bbox_y":y, "bbox_w":w, "bbox_h":h, "aspect_ratio": round(aspect_ratio,2),
                "source_algo": "DO2MR"
            })

        # Process scratches (from LEI)
        num_labels_scratch, labels_scratch, stats_scratch, centroids_scratch = cv2.connectedComponentsWithStats(
            scratch_mask, connectivity=8
        )
        for i in range(1, num_labels_scratch): # Skip background
            area_px = stats_scratch[i, cv2.CC_STAT_AREA]
            if area_px < 10:  # Min scratch area
                continue

            centroid_x_px = int(centroids_scratch[i][0])
            centroid_y_px = int(centroids_scratch[i][1])
            x, y, w, h = stats_scratch[i, cv2.CC_STAT_LEFT], stats_scratch[i, cv2.CC_STAT_TOP], \
                         stats_scratch[i, cv2.CC_STAT_WIDTH], stats_scratch[i, cv2.CC_STAT_HEIGHT]


            zone = "unknown"
            for zone_name, zm in zone_masks.items():
                 if zm[centroid_y_px, centroid_x_px] > 0:
                    zone = zone_name
                    break
            
            # For scratches, length is more relevant than area-based diameter
            # Fit a rotated bounding box to get length and width
            contour_mask = (labels_scratch == i).astype(np.uint8)
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            length_val = 0
            width_val = 0
            
            if contours:
                rect = cv2.minAreaRect(contours[0]) # ((cx,cy), (width, height), angle)
                box_dims = sorted(rect[1]) # width, height (sorted smallest to largest)
                scratch_width_px = box_dims[0]
                scratch_length_px = box_dims[1]

                if not self.use_pixel_units and self.um_per_px:
                    length_val = scratch_length_px * self.um_per_px
                    width_val = scratch_width_px * self.um_per_px
                    size_unit = "um"
                else:
                    length_val = scratch_length_px
                    width_val = scratch_width_px
                    size_unit = "px"
            else: # Fallback if minAreaRect fails (should not happen if area_px > 0)
                length_val = max(w,h) # Use bbox dim as length
                width_val = min(w,h)
                size_unit = "px"
                if not self.use_pixel_units and self.um_per_px:
                    length_val *= self.um_per_px
                    width_val *= self.um_per_px
                    size_unit = "um"


            defects.append({
                "type": "scratch", "zone": zone, f"length ({size_unit})": round(length_val,2), f"width ({size_unit})": round(width_val,2),
                "area_px": area_px, "centroid_x_px": centroid_x_px, "centroid_y_px": centroid_y_px,
                 "bbox_x": x, "bbox_y":y, "bbox_w":w, "bbox_h":h,
                "source_algo": "LEI"
            })
            
        return pd.DataFrame(defects)

    def apply_pass_fail_criteria(self, defects_df: pd.DataFrame) -> Tuple[str, List[str]]:
        status = "PASS"
        failure_reasons = []

        # Use zones_um for criteria, convert defect sizes if necessary
        # This part needs to be robust if operating in pixel_units mode without um_per_px
        # For now, assume criteria are always in µm.
        
        active_zones_criteria = self.zones_um # Criteria are typically in microns

        for zone_name, zone_criteria in active_zones_criteria.items():
            if not zone_criteria.get("defects_allowed", True):
                if not defects_df[defects_df["zone"] == zone_name].empty:
                    status = "FAIL"
                    failure_reasons.append(f"{zone_name}: No defects allowed, but defects found.")
                continue

            zone_defects = defects_df[defects_df["zone"] == zone_name]
            if zone_defects.empty:
                continue

            max_allowed_defect_size_um = zone_criteria.get("max_defect_um", float('inf'))

            for _, defect in zone_defects.iterrows():
                defect_size_um = 0
                # Determine the relevant size of the defect in µm
                if self.use_pixel_units and not self.um_per_px:
                    # Cannot reliably compare to µm criteria if no µm/px and in pixel mode
                    # This check should ideally happen before applying rules, or rules should adapt
                    print(f"Warning: Cannot apply µm-based criteria for zone {zone_name} in pixel_units mode without um_per_px. Skipping defect.")
                    continue
                
                if not self.use_pixel_units or (self.use_pixel_units and self.um_per_px):
                    current_um_per_px = self.um_per_px if self.um_per_px else 1.0 # Avoid error if um_per_px somehow None

                    if defect["type"] == "dig" or defect["type"] == "region_scratch_like":
                        # 'size (um)' or 'size (px)'
                        size_col_name_um = "size (um)"
                        size_col_name_px = "size (px)"
                        if size_col_name_um in defect:
                            defect_size_um = defect[size_col_name_um]
                        elif size_col_name_px in defect and self.um_per_px: # Convert from px if needed
                            defect_size_um = defect[size_col_name_px] * current_um_per_px
                        else: # Fallback or error
                            defect_size_um = np.sqrt(defect["area_px"]) * current_um_per_px


                    elif defect["type"] == "scratch":
                        # 'length (um)' or 'length (px)'
                        length_col_name_um = "length (um)"
                        length_col_name_px = "length (px)"
                        if length_col_name_um in defect:
                            defect_size_um = defect[length_col_name_um] # Use length for scratches
                        elif length_col_name_px in defect and self.um_per_px:
                            defect_size_um = defect[length_col_name_px] * current_um_per_px
                        else: # Fallback
                            defect_size_um = max(defect["bbox_w"], defect["bbox_h"]) * current_um_per_px


                    if defect_size_um > max_allowed_defect_size_um:
                        status = "FAIL"
                        failure_reasons.append(
                            f"{zone_name}: {defect['type']} size {defect_size_um:.2f}µm exceeds limit {max_allowed_defect_size_um}µm"
                        )
            
            # Example: Max number of defects in a zone (add to self.zones_um if needed)
            # max_defect_count_in_zone = zone_criteria.get("max_defect_count", float('inf'))
            # if len(zone_defects) > max_defect_count_in_zone:
            #     status = "FAIL"
            #     failure_reasons.append(f"{zone_name}: Too many defects ({len(zone_defects)} > {max_defect_count_in_zone})")
        
        return status, list(set(failure_reasons)) # Remove duplicate reasons


    def inspect_fiber(self, image_path: str) -> Dict:
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "ERROR", "failure_reasons": [f"Could not load image: {image_path}"], "defect_count": 0, "defects": []}
        
        original_gray, preprocessed_img = self.preprocess_image(image) # preprocessed_img is CLAHE + bilateral
        
        # Use a less aggressively processed image for fiber finding if CLAHE distorts edges
        # For example, use the 'denoised' output from original preprocess_image
        # Or use original_gray if blurring affects HoughCircles too much
        # For now, using the 'denoised' output (which is bilateral in this version)
        
        # find_fiber_center_and_radius now takes the image directly.
        # We need to decide which version of the preprocessed image is best for it.
        # Let's use 'original_gray' (just grayscaled) or 'denoised' (bilateral filtered) for stability
        # as strong contrast enhancement (CLAHE) might create false edges for Hough.
        gray_for_hough = cv2.GaussianBlur(original_gray, (5,5), 0) # A gentle blur for Hough

        center, fiber_radius_px = self.find_fiber_center_and_radius(gray_for_hough)

        if center is None: # HoughCircles failed completely
             return {"status": "ERROR", "failure_reasons": ["Could not determine fiber center."], "defect_count": 0, "defects": []}

        zone_masks = self.create_zone_masks(original_gray.shape, center, fiber_radius_px)
        if "error_zone" in zone_masks or "error_zone_um_per_px" in zone_masks :
            return {"status": "ERROR", "failure_reasons": ["Could not create zone masks due to missing radius or um_per_px."], "defect_count": 0, "defects": []}

        # Use preprocessed_img (CLAHE + bilateral) for DO2MR as it's for region defects sensitive to contrast
        region_mask, labeled_regions = self.detect_region_defects_do2mr(preprocessed_img)
        
        # Use original_gray (or minimally processed gray) for LEI as it has its own enhancement logic for lines
        scratch_mask = self.detect_scratches_lei(original_gray) 
        
        defects_df = self.classify_defects(labeled_regions, scratch_mask, zone_masks)
        status, failure_reasons = self.apply_pass_fail_criteria(defects_df)
        
        results = {
            "image_path": image_path,
            "status": status,
            "failure_reasons": failure_reasons,
            "defect_count": len(defects_df),
            "defects": defects_df.to_dict('records') if not defects_df.empty else [],
            "fiber_center_px": center,
            "fiber_radius_px": fiber_radius_px,
            "um_per_px": self.um_per_px if not self.use_pixel_units and self.um_per_px else "N/A (Pixel Mode or No Calib)",
            "masks_viz": { # Masks for visualization
                "region_defects": region_mask,
                "scratches": scratch_mask,
            }
        }
        return results

    def visualize_results(self, image_path: str, results: Dict, save_path: Optional[str] = None):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Could not load image for visualization: {image_path}")
            return

        vis_image = image_bgr.copy()
        center = results.get("fiber_center_px")
        
        # Draw zone boundaries
        # Use the actual zone definitions that were used (um or px)
        temp_zone_defs = self.zones_um
        current_scale = self.um_per_px
        if self.use_pixel_units or not self.um_per_px :
            temp_zone_defs = self.zones_px
            current_scale = 1.0 # radii are already in px
            unit_suffix = "_px"
        else:
            unit_suffix = ""


        if center:
            # Zone boundary colors
            zone_colors = {
                "core": (255, 0, 0),  # Blue
                "cladding": (0, 255, 0),  # Green
                "ferrule_contact": (0, 0, 255),  # Red
                "adhesive_bond": (255,255,0) # Cyan
            }
            for zone_name, zone_params in temp_zone_defs.items():
                r_max_val = zone_params.get(f"r_max{unit_suffix}")
                if r_max_val is not None and current_scale != 0:
                    radius_px_vis = int(r_max_val / current_scale if not self.use_pixel_units and self.um_per_px else r_max_val)
                    color = zone_colors.get(zone_name, (128, 128, 128)) # Default gray
                    cv2.circle(vis_image, center, radius_px_vis, color, 1) # Thinner lines for zones

        # Overlay defects masks
        region_mask_viz = results.get("masks_viz", {}).get("region_defects")
        scratch_mask_viz = results.get("masks_viz", {}).get("scratches")

        if region_mask_viz is not None:
            vis_image[region_mask_viz > 0] = cv2.addWeighted(vis_image[region_mask_viz > 0],0.5, np.array([0,255,255], dtype=np.uint8), 0.5,0) # Yellow overlay for digs
        if scratch_mask_viz is not None:
            vis_image[scratch_mask_viz > 0] = cv2.addWeighted(vis_image[scratch_mask_viz > 0],0.5, np.array([255,0,255],dtype=np.uint8),0.5,0) # Magenta overlay for scratches
        
        # Draw defect bounding boxes and labels
        for defect in results.get("defects", []):
            x,y,w,h = defect.get("bbox_x"), defect.get("bbox_y"), defect.get("bbox_w"), defect.get("bbox_h")
            if all(v is not None for v in [x,y,w,h]):
                 cv2.rectangle(vis_image, (x,y), (x+w, y+h), (0,165,255), 1) # Orange boxes
                 # label = f"{defect['type']}" # Simple label
                 # cv2.putText(vis_image, label, (x, y-5 if y-5 >5 else y+h+10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)


        status_text = results.get("status", "UNKNOWN")
        status_color = (0, 255, 0) if status_text == "PASS" else (0, 0, 255)
        cv2.putText(vis_image, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(vis_image, f"Defects: {results.get('defect_count',0)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if results.get("failure_reasons"):
            for i, reason in enumerate(results["failure_reasons"]):
                cv2.putText(vis_image, reason, (10, 90 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)


        # Displaying
        if save_path:
            output_dir = Path(save_path).parent
            output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            
            # Create figure with subplots for more detailed output
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))
            fig.suptitle(f"Inspection: {Path(image_path).name} - Status: {status_text}", fontsize=14)

            axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            axes[1].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            axes[1].set_title("Processed with Defects & Zones")
            axes[1].axis('off')
            
            # Add defect table if any
            if results.get("defect_count", 0) > 0:
                defects_df_vis = pd.DataFrame(results['defects'])
                cols_to_show = ["type", "zone", "source_algo"]
                # Add size columns based on what's available (um or px)
                if "size (um)" in defects_df_vis.columns: cols_to_show.append("size (um)")
                elif "size (px)" in defects_df_vis.columns: cols_to_show.append("size (px)")
                if "length (um)" in defects_df_vis.columns: cols_to_show.append("length (um)")
                elif "length (px)" in defects_df_vis.columns: cols_to_show.append("length (px)")

                table_data = defects_df_vis[cols_to_show].head(10) # Show first 10 defects
                
                # Add table below plots if space allows or as separate image
                # For simplicity, we'll just save the main viz plot
                # table_str = table_data.to_string(index=False)
                # fig.text(0.5, 0.01, table_str, ha='center', va='bottom', fontsize=8, family='monospace')
                # plt.subplots_adjust(bottom=0.2) # Adjust layout to make space for text


            plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust for suptitle
            
            try:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            except Exception as e:
                print(f"Error saving visualization: {e}")
            plt.close(fig) # Close the figure to free memory
        else:
            cv2.imshow("Inspection Result", vis_image)
            cv2.waitKey(0)
            cv2.destroyWindow("Inspection Result")


def calibrate_system(calibration_image_path: str, dot_spacing_um: float = 10.0, config_path: str = "calibration.json") -> float:
    image = cv2.imread(calibration_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load calibration image: {calibration_image_path}")
    
    # Enhanced dot finding:
    blurred = cv2.GaussianBlur(image, (7,7), 0) # More blur for noisy targets
    # Adaptive threshold can be better than Otsu for uneven illumination on target
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2) # Use INV if dots are dark

    # Filter contours by area and circularity
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < 20 or area > 5000: continue # Adjust min/max area based on target dot size
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        if 0.7 < circularity < 1.3: # Filter for circular shapes
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                centroids.append((cx, cy))
    
    if len(centroids) < 2:
        raise ValueError(f"Not enough distinct circular calibration dots found (found {len(centroids)}). Check image or parameters.")
    
    # Robust distance calculation (e.g. average of nearest neighbor distances if grid, or known pattern)
    # For simplicity, sort by x and assume horizontal alignment or use a more advanced grid fitting
    centroids = sorted(centroids, key=lambda c: (c[0], c[1])) # Sort by x, then y
    
    distances = []
    # This simple distance calc assumes a linear array or well-spaced grid
    # For a grid, one might find all pair-wise distances and look for a peak in histogram of distances
    for i in range(1, len(centroids)):
        # Calculate distance only to the closest relevant neighbor
        # This requires more complex logic for a 2D grid.
        # Simplified: assuming somewhat linear arrangement along X for this example
        dist = np.sqrt((centroids[i][0] - centroids[i-1][0])**2 + (centroids[i][1] - centroids[i-1][1])**2)
        # Filter out very large or very small distances if centroids are not perfectly on a grid
        if dist > 5 and dist < image.shape[1] / 2 : # Heuristic filter for plausible distances
            distances.append(dist)

    if not distances:
        raise ValueError("Could not calculate valid distances between calibration dots.")
        
    avg_distance_px = np.mean(distances)
    if avg_distance_px == 0:
        raise ValueError("Average pixel distance is zero, cannot calibrate.")

    um_per_px = dot_spacing_um / avg_distance_px
    
    calibration_data = {"um_per_px": um_per_px, "dot_spacing_um_used": dot_spacing_um, "avg_dot_dist_px": avg_distance_px}
    with open(config_path, "w") as f:
        json.dump(calibration_data, f, indent=4)
    
    print(f"Calibration complete: {um_per_px:.4f} µm/pixel. Saved to {config_path}")
    return um_per_px

# --- Main Execution ---
if __name__ == "__main__":
    # --- Calibration (Optional) ---
    # Uncomment to run calibration first:
    # try:
    #     calibrate_image = "path/to/your/calibration_target_image.png" # IMPORTANT: Provide path
    #     known_spacing = 10.0 # In micrometers
    #     if not Path(calibrate_image).exists():
    #         print(f"Calibration image {calibrate_image} not found. Skipping calibration.")
    #     else:
    #         print(f"Starting calibration with {calibrate_image} (dot spacing: {known_spacing}µm)...")
    #         calibrate_system(calibrate_image, known_spacing)
    # except ValueError as e:
    #     print(f"Calibration error: {e}")
    #     # Decide if to exit or continue with defaults/pixel mode
    #     # exit() 
    # except Exception as e:
    #     print(f"An unexpected error occurred during calibration: {e}")
    #     # exit()

    # --- User Input for Fiber Dimensions ---
    use_pixel_units_input = input("Use pixel units for zone definitions? (yes/no, default: no): ").strip().lower()
    use_pixels = True if use_pixel_units_input == 'yes' else False

    core_dia_um_in = None
    cladding_dia_um_in = None
    ferrule_dia_um_in = 250.0 # Default contact zone for IEC

    if not use_pixels:
        try:
            core_dia_um_str = input("Enter core diameter in µm (e.g., 9 for SMF, 50 or 62.5 for MMF, or press Enter to skip): ").strip()
            if core_dia_um_str:
                core_dia_um_in = float(core_dia_um_str)

            cladding_dia_um_str = input("Enter cladding diameter in µm (e.g., 125, or press Enter to skip): ").strip()
            if cladding_dia_um_str:
                cladding_dia_um_in = float(cladding_dia_um_str)
            
            # Optionally ask for ferrule/contact zone diameter if it's variable
            # ferrule_dia_um_str = input(f"Enter ferrule contact zone outer diameter in µm (default: {ferrule_dia_um_in}): ").strip()
            # if ferrule_dia_um_str:
            #    ferrule_dia_um_in = float(ferrule_dia_um_str)

            if core_dia_um_in is None or cladding_dia_um_in is None:
                print("Micron diameters not fully specified by user, will use defaults or rely on robust fiber finding if possible.")
            else:
                 print(f"Using specified core: {core_dia_um_in}µm, cladding: {cladding_dia_um_in}µm.")

        except ValueError:
            print("Invalid diameter input. Using default zone definitions or pixel units if chosen.")
            core_dia_um_in = None # Reset on error
            cladding_dia_um_in = None


    # --- Inspector Initialization ---
    try:
        inspector = FiberOpticInspector(
            core_diameter_um=core_dia_um_in,
            cladding_diameter_um=cladding_dia_um_in,
            ferrule_outer_diameter_um=ferrule_dia_um_in,
            use_pixel_units=use_pixels,
            calibration_file="calibration.json" # Ensure this file exists if not use_pixels
        )
    except Exception as e:
        print(f"Failed to initialize FiberOpticInspector: {e}")
        print("Please ensure 'calibration.json' exists and is valid if not using pixel units, or run calibration.")
        exit()

    # --- Batch Image Processing ---
    # Modify image_source to point to a directory of images or a list of paths
    # image_source = '/home/jarvis/Documents/GitHub/OpenCV-Practice/img5.jpg' # Single image
    image_directory = input("Enter directory containing fiber images (e.g., ./images) or path to single image: ").strip()
    
    image_paths = []
    if Path(image_directory).is_file():
        image_paths.append(image_directory)
    elif Path(image_directory).is_dir():
        # Get common image types
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            image_paths.extend(glob.glob(os.path.join(image_directory, ext)))
    else:
        print(f"Error: '{image_directory}' is not a valid file or directory.")
        exit()

    if not image_paths:
        print(f"No images found in '{image_directory}'.")
        exit()
        
    print(f"Found {len(image_paths)} images to process.")

    output_base_dir = Path("./inspection_results")
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    all_results_summary = []

    for img_path in image_paths:
        print(f"\n--- Inspecting: {img_path} ---")
        try:
            results = inspector.inspect_fiber(img_path)
            
            print(f"Status: {results['status']}")
            if results['failure_reasons']:
                print("Failure reasons:")
                for reason in results['failure_reasons']:
                    print(f"  - {reason}")
            print(f"Total defects found: {results['defect_count']}")
            
            if results['defect_count'] > 0:
                print("Defect details:")
                defects_df = pd.DataFrame(results['defects'])
                print(defects_df.to_string())
            
            # Generate a unique name for the output visualization
            img_filename = Path(img_path).stem
            output_viz_path = output_base_dir / f"{img_filename}_inspected.png"
            inspector.visualize_results(img_path, results, save_path=str(output_viz_path))
            
            all_results_summary.append({
                "image": Path(img_path).name,
                "status": results['status'],
                "defect_count": results['defect_count'],
                "failure_reasons": "; ".join(results['failure_reasons']) if results['failure_reasons'] else ""
            })

        except Exception as e:
            print(f"Error during inspection of {img_path}: {e}")
            all_results_summary.append({
                "image": Path(img_path).name, "status": "ERROR_PROCESSING",
                "defect_count": -1, "failure_reasons": str(e)
            })
            # import traceback
            # traceback.print_exc() # For debugging detailed errors

    # Save summary of all results
    if all_results_summary:
        summary_df = pd.DataFrame(all_results_summary)
        summary_csv_path = output_base_dir / "_inspection_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nInspection summary saved to: {summary_csv_path}")

    print("\nBatch inspection complete.")