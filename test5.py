import cv2
import numpy as np
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import glob
import os

class FiberOpticInspector:
    """
    Automated Fiber Optic End Face Defect Detection System
    Smarter handling of user-provided dimensions and calibration.
    """

    def __init__(self,
                 user_core_diameter_um: Optional[float] = None,
                 user_cladding_diameter_um: Optional[float] = None,
                 user_ferrule_outer_diameter_um: Optional[float] = 250.0,
                 use_calibration_file: bool = True, # Whether to attempt loading from calibration.json
                 calibration_file_path: str = "calibration.json"):

        self.user_core_diameter_um = user_core_diameter_um
        self.user_cladding_diameter_um = user_cladding_diameter_um
        self.user_ferrule_outer_diameter_um = user_ferrule_outer_diameter_um

        self.calibrated_um_per_px: Optional[float] = None
        self.inferred_um_per_px_current_image: Optional[float] = None # For MICRON_INFERRED mode, per image
        self.effective_um_per_px: Optional[float] = None # The scale to actually use for an image

        self.operating_mode: str = "PIXEL_ONLY" # Default

        if use_calibration_file:
            try:
                calibration_data = self._load_calibration(calibration_file_path)
                self.calibrated_um_per_px = calibration_data.get("um_per_px")
                if self.calibrated_um_per_px and self.calibrated_um_per_px > 0:
                    self.effective_um_per_px = self.calibrated_um_per_px
                    self.operating_mode = "MICRON_CALIBRATED"
                    print(f"Operating in MICRON_CALIBRATED mode (um_per_px: {self.effective_um_per_px:.4f} from {calibration_file_path}).")
                else:
                    print(f"Warning: Valid um_per_px not found in {calibration_file_path} despite use_calibration_file=True.")
                    self.calibrated_um_per_px = None # Ensure it's None if invalid
            except (FileNotFoundError, ValueError) as e:
                print(f"Calibration file warning: {e}. Will proceed based on other inputs.")

        if self.operating_mode != "MICRON_CALIBRATED" and self.user_cladding_diameter_um is not None:
            # If no calibrated scale, but user gave cladding diameter, we'll aim for inferred mode.
            # The actual inference happens per-image after cladding detection.
            self.operating_mode = "MICRON_INFERRED"
            print(f"Operating in MICRON_INFERRED mode (will attempt to infer um_per_px from detected cladding and user input: {self.user_cladding_diameter_um} um).")
        elif self.operating_mode != "MICRON_CALIBRATED": # Fallback to PIXEL_ONLY
            self.operating_mode = "PIXEL_ONLY"
            print("Operating in PIXEL_ONLY mode (no calibration and/or no user-provided cladding diameter for inference).")


        # Base zone definitions in microns (defaults can be overridden if user_core/cladding_diameter_um are provided)
        self.zones_um_template = {
            "core": {"r_min": 0, "r_max": 4.5, "max_defect_um": 3, "defects_allowed": True},
            "cladding": {"r_min": 4.5, "r_max": 62.5, "max_defect_um": 10, "defects_allowed": True},
            "ferrule_contact": {"r_min": 62.5, "r_max": 125.0, "max_defect_um": 25, "defects_allowed": True}, # IEC Zone A (Contact)
            "adhesive_bond": {"r_min": 125.0, "r_max": 140.0, "max_defect_um": 50, "defects_allowed": True},
        }
        # Default pixel zones (used if in PIXEL_ONLY mode and not dynamically updated)
        self.zones_px_definitions = {
            "core": {"r_min_px": 0, "r_max_px": 30, "max_defect_px": 5, "defects_allowed": True},
            "cladding": {"r_min_px": 30, "r_max_px": 80, "max_defect_px": 15, "defects_allowed": True},
            "ferrule_contact": {"r_min_px": 80, "r_max_px": 150, "max_defect_px": 25, "defects_allowed": True},
        }
        
        # Update um_template if user provided dimensions
        if self.user_core_diameter_um is not None and self.user_cladding_diameter_um is not None:
            core_r = self.user_core_diameter_um / 2.0
            cladding_r = self.user_cladding_diameter_um / 2.0
            ferrule_r = (self.user_ferrule_outer_diameter_um or 250.0) / 2.0 # Use provided or default
            
            self.zones_um_template["core"]["r_max"] = core_r
            self.zones_um_template["cladding"]["r_min"] = core_r
            self.zones_um_template["cladding"]["r_max"] = cladding_r
            self.zones_um_template["ferrule_contact"]["r_min"] = cladding_r
            self.zones_um_template["ferrule_contact"]["r_max"] = ferrule_r
            self.zones_um_template["adhesive_bond"]["r_min"] = ferrule_r
            self.zones_um_template["adhesive_bond"]["r_max"] = ferrule_r + 15 # Example offset
            print("Micron zone templates updated with user-provided core/cladding diameters.")


        self.do2mr_params = {"kernel_size": (15, 15), "gamma": 3.0, "min_area_px": 20}
        self.lei_params = {"kernel_size": 15, "angles": np.arange(0, 180, 10), "threshold_factor": 2.5}
        self.hough_params = {
            "dp": 1.2, "param1": 70, "param2": 40,
            "minDistFactor": 1/8.0, "minRadiusFactor": 1/10.0, "maxRadiusFactor": 1/2.0
        }

    def _load_calibration(self, filepath: str) -> Dict:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Calibration file '{filepath}' not found.")
        try:
            with open(filepath, 'r') as f: cal_data = json.load(f)
            if "um_per_px" not in cal_data or not isinstance(cal_data["um_per_px"], (float, int)) or cal_data["um_per_px"] <= 0:
                raise ValueError("'um_per_px' is missing, not a number, or non-positive in calibration file.")
            return cal_data
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from {filepath}.")

    def _update_pixel_zones_dynamically(self, detected_cladding_radius_px: float):
        """Dynamically update pixel zone definitions based on detected cladding radius."""
        if not detected_cladding_radius_px or detected_cladding_radius_px <=0: return

        # Use ratios from the (potentially user-updated) um_template
        um_core_r = self.zones_um_template["core"]["r_max"]
        um_cladding_r = self.zones_um_template["cladding"]["r_max"]
        um_ferrule_r = self.zones_um_template["ferrule_contact"]["r_max"]

        if um_cladding_r == 0: # Avoid division by zero if template is bad
            print("Warning: um_cladding_r from template is zero, cannot dynamically scale pixel zones.")
            return

        self.zones_px_definitions["core"]["r_max_px"] = int(detected_cladding_radius_px * (um_core_r / um_cladding_r))
        self.zones_px_definitions["cladding"]["r_min_px"] = self.zones_px_definitions["core"]["r_max_px"]
        self.zones_px_definitions["cladding"]["r_max_px"] = int(detected_cladding_radius_px)
        self.zones_px_definitions["ferrule_contact"]["r_min_px"] = int(detected_cladding_radius_px)
        self.zones_px_definitions["ferrule_contact"]["r_max_px"] = int(detected_cladding_radius_px * (um_ferrule_r / um_cladding_r))
        # Update max_defect_px too if desired (e.g. scale from um_defects if effective_um_per_px is known)
        print(f"Dynamically updated pixel zones based on detected cladding radius ({detected_cladding_radius_px:.2f}px)")

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # ... (same as your last version)
        if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else: gray = image.copy()
        original_gray = gray.copy()
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_clahe = clahe.apply(denoised)
        return original_gray, denoised, enhanced_clahe


    def find_fiber_center_and_radius(self, image_for_hough: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        # ... (same as your last version)
        edges = cv2.Canny(image_for_hough, self.hough_params['param1'] / 2, self.hough_params['param1'])
        min_img_dim = min(image_for_hough.shape[0], image_for_hough.shape[1])
        min_dist = int(min_img_dim * self.hough_params['minDistFactor'])
        min_radius_calc = int(min_img_dim * self.hough_params['minRadiusFactor'])
        max_radius_calc = int(min_img_dim * self.hough_params['maxRadiusFactor'])
        min_radius = max(10, min_radius_calc); max_radius = max(min_radius + 10, max_radius_calc)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=self.hough_params['dp'], minDist=min_dist,
                                   param1=self.hough_params['param1'], param2=self.hough_params['param2'],
                                   minRadius=min_radius, maxRadius=max_radius)
        if circles is not None:
            circles_uint = np.uint16(np.around(circles)); best_circle = circles_uint[0][np.argmax(circles_uint[0, :, 2])]
            center, radius = (int(best_circle[0]), int(best_circle[1])), int(best_circle[2])
            if radius < 5: print(f"Warning: Detected fiber radius {radius}px is very small.")
            return center, radius
        print("Warning: No fiber outline detected by HoughCircles. Fallback: image center, unknown radius.");
        return (image_for_hough.shape[1]//2, image_for_hough.shape[0]//2), None


    def _get_active_zone_definitions_and_scale(self) -> Tuple[Dict, Optional[float], str]:
        """Determines which zone definitions (um_template or px_definitions) and scale to use."""
        if self.effective_um_per_px and self.effective_um_per_px > 0:
            # This implies MICRON_CALIBRATED or MICRON_INFERRED (after inference)
            return self.zones_um_template, self.effective_um_per_px, "" # "" for micron suffix
        else: # PIXEL_ONLY or MICRON_INFERRED before inference
            return self.zones_px_definitions, 1.0, "_px" # 1.0 scale for pixels


    def create_zone_masks(self, image_shape: Tuple, center: Tuple[int, int]) -> Dict[str, np.ndarray]:
        masks = {}
        height, width = image_shape[:2]
        Y, X = np.ogrid[:height, :width]
        dist_from_center_sq = (X - center[0])**2 + (Y - center[1])**2

        active_zones, scale, suffix = self._get_active_zone_definitions_and_scale()

        if scale is None or scale <= 0: # Should not happen if logic is correct
            print("Error: Invalid scale for zone creation. Returning empty masks.")
            return {"error_invalid_scale_for_zones": np.zeros(image_shape[:2], dtype=np.uint8)}

        for zone_name, zone_params in active_zones.items():
            r_min_val = zone_params.get(f"r_min{suffix}")
            r_max_val = zone_params.get(f"r_max{suffix}")

            if r_min_val is None or r_max_val is None:
                print(f"Warning: Zone '{zone_name}' missing r_min/max{suffix}. Skipping.")
                continue
            
            r_min_px_sq = (r_min_val / scale)**2
            r_max_px_sq = (r_max_val / scale)**2
            
            zone_mask = (dist_from_center_sq >= r_min_px_sq) & (dist_from_center_sq < r_max_px_sq)
            masks[zone_name] = zone_mask.astype(np.uint8) * 255
        
        if not masks: print("Warning: No zone masks were created.")
        return masks

    # --- detect_region_defects_do2mr, detect_scratches_lei, _apply_linear_detector_refined ---
    # Keep these methods as they were in your last provided version,
    # as they primarily operate on pixel data and their internal logic doesn't
    # directly depend on the um_per_px scale until the classification/sizing stage.
    # For brevity, I'm omitting them here but assume they are present and correct.
    def detect_region_defects_do2mr(self, image_for_do2mr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # ... (Implementation from your last version) ...
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.do2mr_params["kernel_size"])
        img_max = cv2.dilate(image_for_do2mr, kernel)
        img_min = cv2.erode(image_for_do2mr, kernel)
        residual = cv2.absdiff(img_max, img_min)
        residual_filtered = cv2.medianBlur(residual, 5)
        mean_val = np.mean(residual_filtered); std_val = np.std(residual_filtered)
        threshold_val = mean_val + self.do2mr_params["gamma"] * std_val
        if std_val < 1: threshold_val = mean_val + 5
        _, binary_mask = cv2.threshold(residual_filtered, threshold_val, 255, cv2.THRESH_BINARY)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        n_labels, labeled = cv2.connectedComponents(binary_mask)
        return binary_mask, labeled

    def detect_scratches_lei(self, image_for_lei: np.ndarray) -> np.ndarray:
        # ... (Implementation from your last version) ...
        scratch_strength = np.zeros_like(image_for_lei, dtype=np.float32)
        kernel_length = self.lei_params["kernel_size"]
        for angle in self.lei_params["angles"]:
            angle_rad = np.deg2rad(angle); kernel_points = []
            for i in range(-kernel_length//2, kernel_length//2 + 1):
                if i == 0 : continue
                x = int(round(i * np.cos(angle_rad))); y = int(round(i * np.sin(angle_rad)))
                if (x,y) not in kernel_points: kernel_points.append((x,y))
            if not kernel_points: continue
            response = self._apply_linear_detector_refined(image_for_lei, kernel_points)
            scratch_strength = np.maximum(scratch_strength, response)
        scratch_strength_norm = np.zeros_like(scratch_strength, dtype=cv2.CV_8U)
        if scratch_strength.max() > 0:
             cv2.normalize(scratch_strength, scratch_strength_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, scratch_mask = cv2.threshold(scratch_strength_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_close_len = max(3, kernel_length // 3)
        scratch_mask_closed = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=1)
        scratch_mask_opened = cv2.morphologyEx(scratch_mask_closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
        return scratch_mask_opened

    def _apply_linear_detector_refined(self, image: np.ndarray, kernel_points: List[Tuple[int, int]]) -> np.ndarray:
        # ... (Implementation from your last version) ...
        height, width = image.shape; response = np.zeros_like(image, dtype=np.float32)
        max_offset = max(max(abs(dx), abs(dy)) for dx, dy in kernel_points) if kernel_points else 0
        padded = cv2.copyMakeBorder(image, max_offset, max_offset, max_offset, max_offset, cv2.BORDER_REFLECT)
        for r_idx in range(height):
            for c_idx in range(width):
                r_pad, c_pad = r_idx + max_offset, c_idx + max_offset
                line_values = [float(padded[r_pad + dy, c_pad + dx]) for dx, dy in kernel_points]
                if not line_values: continue
                avg_line_val = np.mean(line_values); center_pixel_val = float(padded[r_pad, c_pad])
                current_response = avg_line_val - center_pixel_val
                response[r_idx, c_idx] = max(0, current_response)
        return response

    def classify_defects(self, labeled_regions_map: np.ndarray, scratch_mask_map: np.ndarray,
                         zone_masks_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        defects = []
        active_scale = self.effective_um_per_px # This is now set per-image for inferred mode
        
        # Process region-based defects (from DO2MR)
        num_labels_region, labels_region, stats_region, centroids_region = cv2.connectedComponentsWithStats(
            (labeled_regions_map > 0).astype(np.uint8), connectivity=8)

        for i in range(1, num_labels_region):
            area_px = stats_region[i, cv2.CC_STAT_AREA]
            if area_px < self.do2mr_params["min_area_px"]: continue
            cx_px, cy_px = int(centroids_region[i][0]), int(centroids_region[i][1])
            x,y,w,h = stats_region[i, cv2.CC_STAT_LEFT], stats_region[i, cv2.CC_STAT_TOP], \
                      stats_region[i, cv2.CC_STAT_WIDTH], stats_region[i, cv2.CC_STAT_HEIGHT]
            zone = "unknown"
            for zn, zm in zone_masks_dict.items():
                if 0<=cy_px<zm.shape[0] and 0<=cx_px<zm.shape[1] and zm[cy_px,cx_px]>0: zone=zn; break
            
            size_val, size_unit = (np.sqrt(4*area_px*(active_scale**2)/np.pi), "um") if active_scale else (np.sqrt(area_px), "px")
            aspect_ratio = w/h if h>0 else (w/0.1 if w>0 else 1.0)
            defect_type = "dig" if aspect_ratio < 4.0 and aspect_ratio > 0.25 else "region_elongated"
            defects.append({"type": defect_type, "zone": zone, f"size({size_unit})": round(size_val,2),
                            "area_px": area_px, "cx_px": cx_px, "cy_px": cy_px,
                            "bb_x": x, "bb_y":y, "bb_w":w, "bb_h":h, "ar": round(aspect_ratio,2), "algo": "DO2MR"})

        # Process scratches (from LEI)
        num_labels_scratch, labels_scratch, stats_scratch, centroids_scratch = cv2.connectedComponentsWithStats(
            scratch_mask_map, connectivity=8)
        for i in range(1, num_labels_scratch):
            area_px = stats_scratch[i, cv2.CC_STAT_AREA]
            if area_px < 15: continue
            cx_px, cy_px = int(centroids_scratch[i][0]), int(centroids_scratch[i][1])
            x,y,w,h = stats_scratch[i, cv2.CC_STAT_LEFT], stats_scratch[i, cv2.CC_STAT_TOP], \
                      stats_scratch[i, cv2.CC_STAT_WIDTH], stats_scratch[i, cv2.CC_STAT_HEIGHT]
            zone = "unknown"
            for zn, zm in zone_masks_dict.items():
                if 0<=cy_px<zm.shape[0] and 0<=cx_px<zm.shape[1] and zm[cy_px,cx_px]>0: zone=zn; break
            
            contours_list, _ = cv2.findContours((labels_scratch == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            len_val, wid_val, size_unit = 0,0,"px"
            if contours_list and len(contours_list[0]) >= 5:
                rect = cv2.minAreaRect(contours_list[0]); box_dims = sorted(rect[1])
                wid_px, len_px = box_dims[0], box_dims[1]
            else: wid_px, len_px = min(w,h), max(w,h) # Fallback
            
            len_val, wid_val, size_unit = (len_px*active_scale, wid_px*active_scale, "um") if active_scale else (len_px, wid_px, "px")
            defects.append({"type": "scratch", "zone": zone, f"L({size_unit})": round(len_val,2), f"W({size_unit})": round(wid_val,2),
                            "area_px": area_px, "cx_px": cx_px, "cy_px": cy_px,
                            "bb_x": x, "bb_y":y, "bb_w":w, "bb_h":h, "algo": "LEI"})
        return pd.DataFrame(defects)

    def apply_pass_fail_criteria(self, defects_df: pd.DataFrame) -> Tuple[str, List[str]]:
        status = "PASS"
        failure_reasons = []

        # Rules are always defined in microns in self.zones_um_template
        # Defects are sized using self.effective_um_per_px if available
        
        if not (self.effective_um_per_px and self.effective_um_per_px > 0):
            # Cannot apply micron-based rules if no valid scale is available
            # This means we are in PIXEL_ONLY mode or inferred mode failed to get a scale
            failure_reasons.append(f"Cannot apply µm-based pass/fail criteria: No valid µm/pixel scale. Current mode: {self.operating_mode}.")
            # Optional: Implement pixel-based rules here from self.zones_px_definitions if desired
            # for zone_name, zone_criteria_px in self.zones_px_definitions.items():
            #    # ... apply pixel rules ...
            return "UNDEFINED", failure_reasons


        for zone_name, zone_criteria_um in self.zones_um_template.items(): # Using template with µm rules
            if not zone_criteria_um.get("defects_allowed", True):
                if not defects_df[defects_df["zone"] == zone_name].empty:
                    status = "FAIL"
                    failure_reasons.append(f"{zone_name}: No defects allowed, but defects found.")
                continue

            zone_defects = defects_df[defects_df["zone"] == zone_name]
            if zone_defects.empty: continue

            max_allowed_size_um = zone_criteria_um.get("max_defect_um", float('inf'))

            for _, defect in zone_defects.iterrows():
                defect_size_metric_um = 0
                # Get the size in microns if available
                if defect["type"] == "dig" or defect["type"] == "region_elongated":
                    defect_size_metric_um = defect.get("size(um)", 0) 
                elif defect["type"] == "scratch":
                    defect_size_metric_um = defect.get("L(um)", 0) # Using length for scratch

                if defect_size_metric_um == 0 and "size(px)" in defect: # Fallback if 'size(um)' wasn't populated due to no scale
                     print(f"Warning: Defect in zone {zone_name} has only pixel size, cannot compare to µm rule accurately.")
                     continue # Or handle differently

                if defect_size_metric_um > max_allowed_size_um:
                    status = "FAIL"
                    failure_reasons.append(
                        f"{zone_name}: {defect['type']} size {defect_size_metric_um:.2f}µm exceeds limit {max_allowed_size_um}µm"
                    )
        return status, list(set(failure_reasons))


    def inspect_fiber(self, image_path: str) -> Dict:
        self.inferred_um_per_px_current_image = None # Reset for each image
        self.effective_um_per_px = self.calibrated_um_per_px # Start with calibrated if available

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return {"image_path": image_path, "status": "ERROR", "failure_reasons": [f"Could not load image: {image_path}"], "defect_count": 0}
        
        original_gray, denoised_img, enhanced_clahe_img = self.preprocess_image(image_bgr)
        img_for_hough = cv2.GaussianBlur(denoised_img, (5,5), 0)
        center_px, detected_cladding_radius_px = self.find_fiber_center_and_radius(img_for_hough)

        if center_px is None:
             return {"image_path": image_path, "status": "ERROR", "failure_reasons": ["Could not determine fiber center."], "defect_count": 0}

        current_mode_for_report = self.operating_mode

        if self.operating_mode == "MICRON_INFERRED":
            if detected_cladding_radius_px and detected_cladding_radius_px > 0 and self.user_cladding_diameter_um and self.user_cladding_diameter_um > 0:
                self.inferred_um_per_px_current_image = (self.user_cladding_diameter_um / 2.0) / detected_cladding_radius_px
                self.effective_um_per_px = self.inferred_um_per_px_current_image
                print(f"  Image {Path(image_path).name}: Inferred um_per_px: {self.effective_um_per_px:.4f}")
            else:
                print(f"  Image {Path(image_path).name}: Could not infer um_per_px (missing detected radius or user cladding diameter). Falling back to PIXEL_ONLY for this image.")
                self.effective_um_per_px = None # Force pixel mode for this image if inference fails
                current_mode_for_report = "PIXEL_ONLY (Inference Failed)"
        
        if self.operating_mode == "PIXEL_ONLY" or (self.operating_mode == "MICRON_INFERRED" and not self.effective_um_per_px) :
            if detected_cladding_radius_px and detected_cladding_radius_px > 0:
                self._update_pixel_zones_dynamically(detected_cladding_radius_px)
            else:
                print(f"  Image {Path(image_path).name}: No cladding radius detected, cannot update pixel zones dynamically. Using default pixel zones.")


        zone_masks = self.create_zone_masks(original_gray.shape, center_px)
        if any(k.startswith("error_") for k in zone_masks) or not zone_masks:
            return {"image_path": image_path, "status": "ERROR", "failure_reasons": [zone_masks.get(next(iter(zone_masks.keys())),"Zone creation error or no zones defined")], "defect_count": 0}

        region_mask, labeled_regions = self.detect_region_defects_do2mr(enhanced_clahe_img)
        scratch_mask = self.detect_scratches_lei(denoised_img)
        
        defects_df = self.classify_defects(labeled_regions, scratch_mask, zone_masks)
        status, failure_reasons = self.apply_pass_fail_criteria(defects_df)
        
        return {
            "image_path": image_path, "status": status, "failure_reasons": failure_reasons,
            "defect_count": len(defects_df), "defects": defects_df.to_dict('records') if not defects_df.empty else [],
            "fiber_center_px": center_px, "detected_cladding_radius_px": detected_cladding_radius_px,
            "operating_mode": current_mode_for_report,
            "effective_um_per_px": self.effective_um_per_px if self.effective_um_per_px else "N/A",
            "masks_viz": {"region_defects": region_mask, "scratches": scratch_mask}
        }

    def visualize_results(self, image_path: str, results: Dict, save_path: Optional[str] = None):
        # ... (visualization code from previous response, ensure it uses results correctly) ...
        # Uses self._get_active_zone_definitions_and_scale() for drawing correct zone boundaries
        image_bgr = cv2.imread(image_path)
        if image_bgr is None: print(f"Vis Error: Could not load {image_path}"); return
        vis_image = image_bgr.copy()
        center = results.get("fiber_center_px")
        
        active_zones_vis, scale_vis, suffix_vis = self._get_active_zone_definitions_and_scale()

        if center and scale_vis and scale_vis > 0:
            zone_colors = {"core": (255,0,0), "cladding": (0,255,0), "ferrule_contact": (0,0,255), "adhesive_bond":(255,255,0)}
            for zn, zp in active_zones_vis.items():
                r_max_val = zp.get(f"r_max{suffix_vis}")
                if r_max_val is not None:
                    cv2.circle(vis_image, center, int(r_max_val/scale_vis), zone_colors.get(zn,(128,128,128)),1)
        
        if results.get("masks_viz",{}).get("region_defects") is not None:
            vis_image[results["masks_viz"]["region_defects"] > 0] = cv2.addWeighted(vis_image[results["masks_viz"]["region_defects"] > 0],0.5, np.array([0,255,255],dtype=np.uint8),0.5,0)
        if results.get("masks_viz",{}).get("scratches") is not None:
            vis_image[results["masks_viz"]["scratches"] > 0] = cv2.addWeighted(vis_image[results["masks_viz"]["scratches"] > 0],0.5, np.array([255,0,255],dtype=np.uint8),0.5,0)

        for defect in results.get("defects", []):
            x,y,w,h = defect.get("bb_x"), defect.get("bb_y"), defect.get("bb_w"), defect.get("bb_h")
            if all(v is not None for v in [x,y,w,h]): cv2.rectangle(vis_image, (x,y), (x+w,y+h), (0,165,255),1)

        status_text = results.get("status", "N/A"); color = (0,255,0) if status_text=="PASS" else ((0,165,255) if status_text=="UNDEFINED" else (0,0,255))
        cv2.putText(vis_image, f"Status: {status_text}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(vis_image, f"Mode: {results.get('operating_mode', 'N/A')}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220),1)
        cv2.putText(vis_image, f"Scale: {results.get('effective_um_per_px', 'N/A')}", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220),1)
        cv2.putText(vis_image, f"Defects: {results.get('defect_count',0)}", (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        y_off = 95
        for i, reason in enumerate(results.get("failure_reasons",[])):
            cv2.putText(vis_image, reason, (10, y_off + i*15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)

        if save_path:
            Path(save_path).parent.mkdir(parents=True,exist_ok=True)
            fig, axes = plt.subplots(1,2,figsize=(16,8)); fig.suptitle(f"{Path(image_path).name} - Status: {status_text}", fontsize=14)
            axes[0].imshow(cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)); axes[0].set_title("Original"); axes[0].axis('off')
            axes[1].imshow(cv2.cvtColor(vis_image,cv2.COLOR_BGR2RGB)); axes[1].set_title("Processed"); axes[1].axis('off')
            plt.tight_layout(rect=[0,0.03,1,0.95]); plt.savefig(save_path,dpi=150,bbox_inches='tight'); plt.close(fig)
            print(f"Visualization saved: {save_path}")
        else:
            cv2.imshow("Inspection Result", vis_image); cv2.waitKey(0); cv2.destroyAllWindows()


# --- Calibration Function (example, ensure you have a target image) ---
def calibrate_system(calibration_image_path: str, dot_spacing_um: float = 10.0, config_path: str = "calibration.json") -> Optional[float]:
    # ... (same as your last version) ...
    image = cv2.imread(calibration_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: print(f"Error: Could not load calib image: {calibration_image_path}"); return None
    blurred = cv2.GaussianBlur(image, (7,7),0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for contour in contours:
        area = cv2.contourArea(contour); perimeter = cv2.arcLength(contour,True)
        if area < 10 or area > 2000 or perimeter == 0: continue
        circularity = 4 * np.pi * (area/(perimeter**2))
        if 0.6 < circularity < 1.4:
            M = cv2.moments(contour)
            if M['m00'] != 0: centroids.append((M['m10']/M['m00'], M['m01']/M['m00']))
    if len(centroids) < 2: print(f"Warning: Not enough distinct circular calib dots found ({len(centroids)})."); return None
    centroids = sorted(centroids, key=lambda c: (c[0],c[1])); x_coords = sorted(list(set(c[0] for c in centroids)))
    x_spacings = np.diff(x_coords); plausible_x_spacings = x_spacings[(x_spacings > 5)&(x_spacings < image.shape[1]/4)]
    if len(plausible_x_spacings)>0: avg_dist_px = np.median(plausible_x_spacings)
    elif len(centroids)>1: avg_dist_px = np.sqrt((centroids[1][0]-centroids[0][0])**2 + (centroids[1][1]-centroids[0][1])**2)
    else: print("Warning: Could not determine avg dist between calib dots."); return None
    if avg_dist_px < 5: print("Warning: Calib dot spacing too small."); return None
    if avg_dist_px == 0: print("Warning: Avg px dist is zero."); return None
    um_per_px = dot_spacing_um / avg_dist_px
    cal_data = {"um_per_px":um_per_px, "dot_spacing_um_used":dot_spacing_um, "avg_dot_dist_px":avg_dist_px}
    with open(config_path,"w") as f: json.dump(cal_data,f,indent=4)
    print(f"Calibration successful: {um_per_px:.4f} µm/pixel. Saved to {config_path}"); return um_per_px

# --- Main Execution ---
if __name__ == "__main__":
    calibration_file = "calibration.json"
    
    # --- Optional Calibration Step ---
    # if input("Run calibration? (y/n): ").lower() == 'y':
    #     cal_img_path = input("Path to calibration image: ")
    #     dot_space = float(input("Dot spacing in um (e.g. 10.0): "))
    #     if Path(cal_img_path).exists():
    #         calibrate_system(cal_img_path, dot_space, calibration_file)
    #     else:
    #         print(f"Calibration image {cal_img_path} not found.")

    # --- Gather User Inputs ---
    user_core_dia_um, user_cladding_dia_um = None, None
    provide_dims = input("Do you know the core and cladding diameters in microns? (y/n, default: n): ").strip().lower()
    if provide_dims == 'y':
        try:
            core_str = input("Enter core diameter in µm (e.g., 9, or Enter to skip): ").strip()
            if core_str: user_core_dia_um = float(core_str)
            clad_str = input("Enter cladding diameter in µm (e.g., 125, or Enter to skip): ").strip()
            if clad_str: user_cladding_dia_um = float(clad_str)
        except ValueError:
            print("Invalid micron input. Proceeding without specific user diameters.")
            user_core_dia_um, user_cladding_dia_um = None, None
    
    attempt_calib_load = True
    if not Path(calibration_file).exists() and user_cladding_dia_um is None: # No calib file and no user cladding for inference
        print(f"Warning: '{calibration_file}' not found, and no cladding diameter provided for inference.")
        print("System will operate in PIXEL_ONLY mode.")
        attempt_calib_load = False
    elif not Path(calibration_file).exists() and user_cladding_dia_um is not None: # No calib file but can infer
         print(f"Warning: '{calibration_file}' not found. Will attempt MICRON_INFERRED mode using your cladding diameter.")
         attempt_calib_load = False # Don't try to load if it's not there
    elif Path(calibration_file).exists(): # Calib file exists
        use_cal = input(f"Use existing calibration file '{calibration_file}'? (y/n, default: y): ").strip().lower()
        if use_cal == 'n':
            attempt_calib_load = False
            print("Will not use calibration file. Mode will be MICRON_INFERRED (if cladding_um given) or PIXEL_ONLY.")
        # else, attempt_calib_load remains True

    # --- Initialize Inspector ---
    try:
        inspector = FiberOpticInspector(
            user_core_diameter_um=user_core_dia_um,
            user_cladding_diameter_um=user_cladding_dia_um,
            use_calibration_file=attempt_calib_load, # This now controls if it TRIES to load
            calibration_file_path=calibration_file
        )
    except Exception as e:
        print(f"FATAL: Failed to initialize FiberOpticInspector: {e}"); exit()

    # --- Batch Image Processing ---
    # ... (image path gathering logic from your last version - unchanged) ...
    image_directory = input("Enter directory with images or path to single image: ").strip()
    image_paths = []
    if Path(image_directory).is_file(): image_paths.append(image_directory)
    elif Path(image_directory).is_dir():
        for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"): image_paths.extend(glob.glob(os.path.join(image_directory,ext)))
    else: print(f"Error: '{image_directory}' invalid."); exit()
    if not image_paths: print(f"No images found in '{image_directory}'."); exit()
    print(f"Found {len(image_paths)} images.")

    output_base_dir = Path("./inspection_results"); output_base_dir.mkdir(parents=True,exist_ok=True)
    all_results_summary = []

    for img_path in image_paths:
        print(f"\n--- Inspecting: {Path(img_path).name} ---")
        try:
            results = inspector.inspect_fiber(img_path) # inspect_fiber now handles mode determination internally
            print(f"Status: {results['status']}")
            if results['failure_reasons']: print("Failure reasons:\n" + "\n".join([f"  - {r}" for r in results['failure_reasons']]))
            print(f"Total defects found: {results['defect_count']}")
            if results.get('defects') and results['defect_count'] > 0: print("Defect details:\n" + pd.DataFrame(results['defects']).to_string())
            
            img_filename = Path(img_path).stem
            output_viz_path = output_base_dir / f"{img_filename}_inspected.png"
            inspector.visualize_results(img_path, results, save_path=str(output_viz_path))
            all_results_summary.append({
                "image": Path(img_path).name, "status": results['status'], "defect_count": results['defect_count'],
                "mode": results.get("operating_mode"), "eff_um/px": results.get("effective_um_per_px"),
                "fail_reasons": "; ".join(results.get('failure_reasons',[]))
            })
        except Exception as e:
            print(f"Error inspecting {img_path}: {e}"); import traceback; traceback.print_exc()
            all_results_summary.append({"image":Path(img_path).name, "status":"ERROR_PROC", "defect_count":-1, "fail_reasons":str(e)})

    if all_results_summary:
        summary_df = pd.DataFrame(all_results_summary)
        summary_csv_path = output_base_dir / "_inspection_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSummary saved: {summary_csv_path}")
    print("\nBatch inspection complete.")