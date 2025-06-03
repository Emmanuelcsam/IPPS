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
        self.inferred_um_per_px_current_image: Optional[float] = None
        self.effective_um_per_px: Optional[float] = None

        self.operating_mode: str = "PIXEL_ONLY"

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
                    self.calibrated_um_per_px = None
            except (FileNotFoundError, ValueError) as e:
                print(f"Calibration file warning: {e}. Will proceed based on other inputs.")

        if self.operating_mode != "MICRON_CALIBRATED" and self.user_cladding_diameter_um is not None:
            self.operating_mode = "MICRON_INFERRED"
            print(f"Operating in MICRON_INFERRED mode (will attempt to infer um_per_px from detected cladding and user input: {self.user_cladding_diameter_um} um).")
        elif self.operating_mode != "MICRON_CALIBRATED":
            self.operating_mode = "PIXEL_ONLY"
            print("Operating in PIXEL_ONLY mode (no calibration and/or no user-provided cladding diameter for inference).")

        self.zones_um_template = {
            "core": {"r_min": 0, "r_max": 4.5, "max_defect_um": 3, "defects_allowed": True},
            "cladding": {"r_min": 4.5, "r_max": 62.5, "max_defect_um": 10, "defects_allowed": True},
            "ferrule_contact": {"r_min": 62.5, "r_max": 125.0, "max_defect_um": 25, "defects_allowed": True},
            "adhesive_bond": {"r_min": 125.0, "r_max": 140.0, "max_defect_um": 50, "defects_allowed": True},
        }
        self.zones_px_definitions = {
            "core": {"r_min_px": 0, "r_max_px": 30, "max_defect_px": 5, "defects_allowed": True},
            "cladding": {"r_min_px": 30, "r_max_px": 80, "max_defect_px": 15, "defects_allowed": True},
            "ferrule_contact": {"r_min_px": 80, "r_max_px": 150, "max_defect_px": 25, "defects_allowed": True},
        }
        
        if self.user_core_diameter_um is not None and self.user_cladding_diameter_um is not None:
            core_r = self.user_core_diameter_um / 2.0
            cladding_r = self.user_cladding_diameter_um / 2.0
            ferrule_r = (self.user_ferrule_outer_diameter_um or 250.0) / 2.0
            
            self.zones_um_template["core"]["r_max"] = core_r
            self.zones_um_template["cladding"]["r_min"] = core_r
            self.zones_um_template["cladding"]["r_max"] = cladding_r
            self.zones_um_template["ferrule_contact"]["r_min"] = cladding_r
            self.zones_um_template["ferrule_contact"]["r_max"] = ferrule_r
            self.zones_um_template["adhesive_bond"]["r_min"] = ferrule_r
            self.zones_um_template["adhesive_bond"]["r_max"] = ferrule_r + 15
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
        if not detected_cladding_radius_px or detected_cladding_radius_px <=0: return
        um_core_r = self.zones_um_template["core"]["r_max"]
        um_cladding_r = self.zones_um_template["cladding"]["r_max"]
        um_ferrule_r = self.zones_um_template["ferrule_contact"]["r_max"]
        if um_cladding_r == 0:
            print("Warning: um_cladding_r from template is zero, cannot dynamically scale pixel zones.")
            return
        self.zones_px_definitions["core"]["r_max_px"] = int(detected_cladding_radius_px * (um_core_r / um_cladding_r))
        self.zones_px_definitions["cladding"]["r_min_px"] = self.zones_px_definitions["core"]["r_max_px"]
        self.zones_px_definitions["cladding"]["r_max_px"] = int(detected_cladding_radius_px)
        self.zones_px_definitions["ferrule_contact"]["r_min_px"] = int(detected_cladding_radius_px)
        self.zones_px_definitions["ferrule_contact"]["r_max_px"] = int(detected_cladding_radius_px * (um_ferrule_r / um_cladding_r))
        print(f"Dynamically updated pixel zones based on detected cladding radius ({detected_cladding_radius_px:.2f}px)")

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else: gray = image.copy()
        original_gray = gray.copy()
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_clahe = clahe.apply(denoised)
        return original_gray, denoised, enhanced_clahe

    def find_fiber_center_and_radius(self, image_for_hough: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
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
        if self.effective_um_per_px and self.effective_um_per_px > 0:
            return self.zones_um_template, self.effective_um_per_px, ""
        else:
            return self.zones_px_definitions, 1.0, "_px"

    def create_zone_masks(self, image_shape: Tuple, center: Tuple[int, int]) -> Dict[str, np.ndarray]:
        masks = {}
        height, width = image_shape[:2]
        Y, X = np.ogrid[:height, :width]
        dist_from_center_sq = (X - center[0])**2 + (Y - center[1])**2
        active_zones, scale, suffix = self._get_active_zone_definitions_and_scale()
        if scale is None or scale <= 0:
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

    def detect_region_defects_do2mr(self, image_for_do2mr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        
        # FIX: Use np.uint8 for dtype with np.zeros_like when intending to use with OpenCV normalize/threshold
        scratch_strength_norm = np.zeros_like(scratch_strength, dtype=np.uint8) # Changed cv2.CV_8U to np.uint8
        
        if scratch_strength.max() > 0:
             cv2.normalize(scratch_strength, scratch_strength_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # cv2.CV_8U is OK for OpenCV functions
        
        _, scratch_mask = cv2.threshold(scratch_strength_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel_close_len = max(3, kernel_length // 3)
        scratch_mask_closed = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=1)
        scratch_mask_opened = cv2.morphologyEx(scratch_mask_closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
        return scratch_mask_opened

    def _apply_linear_detector_refined(self, image: np.ndarray, kernel_points: List[Tuple[int, int]]) -> np.ndarray:
        height, width = image.shape; response = np.zeros_like(image, dtype=np.float32)
        max_offset = max(max(abs(dx), abs(dy)) for dx, dy in kernel_points) if kernel_points else 0
        padded = cv2.copyMakeBorder(image, max_offset, max_offset, max_offset, max_offset, cv2.BORDER_REFLECT)
        for r_idx in range(height):
            for c_idx in range(width):
                r_pad, c_pad = r_idx + max_offset, c_idx + max_offset
                line_values = [float(padded[r_pad + dy, c_pad + dx]) for dx, dy in kernel_points]
                if not line_values: continue
                avg_line_val = np.mean(line_values); center_pixel_val = float(padded[r_pad, c_pad])
                current_response = avg_line_val - center_pixel_val # Assuming scratches are brighter lines
                response[r_idx, c_idx] = max(0, current_response)
        return response

    def classify_defects(self, labeled_regions_map: np.ndarray, scratch_mask_map: np.ndarray,
                         zone_masks_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        defects = []
        active_scale = self.effective_um_per_px
        
        num_labels_region, labels_region, stats_region, centroids_region = cv2.connectedComponentsWithStats(
            (labeled_regions_map > 0).astype(np.uint8), connectivity=8)
        for i in range(1, num_labels_region):
            area_px = stats_region[i, cv2.CC_STAT_AREA]
            if area_px < self.do2mr_params["min_area_px"]: continue
            cx_px, cy_px = int(centroids_region[i][0]), int(centroids_region[i][1])
            x,y,w,h = stats_region[i,cv2.CC_STAT_LEFT], stats_region[i,cv2.CC_STAT_TOP],stats_region[i,cv2.CC_STAT_WIDTH],stats_region[i,cv2.CC_STAT_HEIGHT]
            zone = "unknown"
            for zn, zm in zone_masks_dict.items():
                if 0<=cy_px<zm.shape[0] and 0<=cx_px<zm.shape[1] and zm[cy_px,cx_px]>0: zone=zn; break
            size_val, unit = (np.sqrt(4*area_px*(active_scale**2)/np.pi), "um") if active_scale and active_scale > 0 else (np.sqrt(area_px), "px")
            ar = w/h if h>0 else (w/0.1 if w > 0 else 1.0)
            def_type = "dig" if ar < 4.0 and ar > 0.25 else "region_elongated"
            defects.append({"type":def_type, "zone":zone, f"size({unit})":round(size_val,2), "area_px":area_px, "cx_px":cx_px, "cy_px":cy_px,
                            "bb_x":x,"bb_y":y,"bb_w":w,"bb_h":h, "ar":round(ar,2), "algo":"DO2MR"})

        num_labels_scratch, labels_scratch, stats_scratch, centroids_scratch = cv2.connectedComponentsWithStats(
            scratch_mask_map, connectivity=8)
        for i in range(1, num_labels_scratch):
            area_px = stats_scratch[i, cv2.CC_STAT_AREA]
            if area_px < 15: continue
            cx_px,cy_px = int(centroids_scratch[i][0]), int(centroids_scratch[i][1])
            x,y,w,h = stats_scratch[i,cv2.CC_STAT_LEFT], stats_scratch[i,cv2.CC_STAT_TOP],stats_scratch[i,cv2.CC_STAT_WIDTH],stats_scratch[i,cv2.CC_STAT_HEIGHT]
            zone = "unknown"
            for zn, zm in zone_masks_dict.items():
                if 0<=cy_px<zm.shape[0] and 0<=cx_px<zm.shape[1] and zm[cy_px,cx_px]>0: zone=zn; break
            contours_list, _ = cv2.findContours((labels_scratch == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            len_px, wid_px = max(w,h), min(w,h) # Fallback
            if contours_list and len(contours_list[0]) >= 5:
                rect = cv2.minAreaRect(contours_list[0]); wid_px, len_px = sorted(rect[1])[:2]
            
            len_val, wid_val, unit = (len_px*active_scale, wid_px*active_scale, "um") if active_scale and active_scale > 0 else (len_px, wid_px, "px")
            defects.append({"type":"scratch", "zone":zone, f"L({unit})":round(len_val,2), f"W({unit})":round(wid_val,2),
                            "area_px":area_px, "cx_px":cx_px, "cy_px":cy_px, "bb_x":x,"bb_y":y,"bb_w":w,"bb_h":h, "algo":"LEI"})
        return pd.DataFrame(defects)

    def apply_pass_fail_criteria(self, defects_df: pd.DataFrame) -> Tuple[str, List[str]]:
        status = "PASS"; failure_reasons = []
        if not (self.effective_um_per_px and self.effective_um_per_px > 0):
            mode_info = self.operating_mode
            if self.operating_mode == "MICRON_INFERRED" and not self.inferred_um_per_px_current_image:
                mode_info = "MICRON_INFERRED (inference failed)"

            failure_reasons.append(f"Cannot apply µm-based P/F rules: No valid µm/px scale. Mode: {mode_info}.")
            # Optionally, apply pixel-based rules from self.zones_px_definitions if they exist
            # For now, just return UNDEFINED if micron rules can't be applied
            return "UNDEFINED", failure_reasons

        for zone_name, crit_um in self.zones_um_template.items():
            if not crit_um.get("defects_allowed", True) and not defects_df[defects_df["zone"] == zone_name].empty:
                status = "FAIL"; failure_reasons.append(f"{zone_name}: No defects allowed, found some.")
                continue
            zone_defects = defects_df[defects_df["zone"] == zone_name]
            if zone_defects.empty: continue
            max_allowed_um = crit_um.get("max_defect_um", float('inf'))
            for _, defect in zone_defects.iterrows():
                size_um = 0
                if defect["type"] in ["dig", "region_elongated"]: size_um = defect.get("size(um)",0)
                elif defect["type"] == "scratch": size_um = defect.get("L(um)",0)
                if size_um > max_allowed_um:
                    status = "FAIL"; failure_reasons.append(f"{zone_name}: {defect['type']} size {size_um:.2f}µm > {max_allowed_um}µm")
        return status, list(set(failure_reasons))

    def inspect_fiber(self, image_path: str) -> Dict:
        self.inferred_um_per_px_current_image = None
        self.effective_um_per_px = self.calibrated_um_per_px

        image_bgr = cv2.imread(image_path)
        if image_bgr is None: return {"image_path": image_path, "status": "ERROR", "failure_reasons": [f"Load fail: {image_path}"], "defect_count": 0}
        
        original_gray, denoised_img, enhanced_clahe_img = self.preprocess_image(image_bgr)
        img_for_hough = cv2.GaussianBlur(denoised_img, (5,5), 0)
        center_px, detected_cladding_radius_px = self.find_fiber_center_and_radius(img_for_hough)

        if center_px is None: return {"image_path": image_path, "status": "ERROR", "failure_reasons":["Center find fail"], "defect_count":0}

        current_mode_report = self.operating_mode
        if self.operating_mode == "MICRON_INFERRED":
            if detected_cladding_radius_px and detected_cladding_radius_px > 0 and \
               self.user_cladding_diameter_um and self.user_cladding_diameter_um > 0:
                self.inferred_um_per_px_current_image = (self.user_cladding_diameter_um/2.0) / detected_cladding_radius_px
                self.effective_um_per_px = self.inferred_um_per_px_current_image
                print(f"  Image {Path(image_path).name}: Inferred um_per_px: {self.effective_um_per_px:.4f}")
            else:
                print(f"  Image {Path(image_path).name}: Could not infer um/px. Fallback to PIXEL_ONLY for this image.")
                self.effective_um_per_px = None; current_mode_report = "PIXEL_ONLY (Inference Fail)"
        
        if self.operating_mode == "PIXEL_ONLY" or not self.effective_um_per_px: # If pixel mode or inference failed
            if detected_cladding_radius_px and detected_cladding_radius_px > 0:
                self._update_pixel_zones_dynamically(detected_cladding_radius_px)
            else: print(f"  Image {Path(image_path).name}: No radius for dynamic pixel zones. Using defaults.")

        zone_masks = self.create_zone_masks(original_gray.shape, center_px)
        if any(k.startswith("error_") for k in zone_masks) or not zone_masks:
            err_key = next(iter(zone_masks.keys())) if zone_masks else "Unknown zone error"
            return {"image_path":image_path, "status":"ERROR", "failure_reasons":[zone_masks.get(err_key, "Zone creation error")], "defect_count":0}

        region_mask, labeled = self.detect_region_defects_do2mr(enhanced_clahe_img)
        scratch_mask = self.detect_scratches_lei(denoised_img)
        defects_df = self.classify_defects(labeled, scratch_mask, zone_masks)
        status, reasons = self.apply_pass_fail_criteria(defects_df)
        
        return {"image_path":image_path, "status":status, "failure_reasons":reasons,
                "defect_count":len(defects_df), "defects":defects_df.to_dict('records') if not defects_df.empty else [],
                "fiber_center_px":center_px, "detected_cladding_radius_px":detected_cladding_radius_px,
                "operating_mode":current_mode_report, "effective_um_per_px":self.effective_um_per_px if self.effective_um_per_px else "N/A",
                "masks_viz":{"region_defects":region_mask, "scratches":scratch_mask}}

    def visualize_results(self, image_path: str, results: Dict, save_path: Optional[str] = None):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None: print(f"Vis Error: Could not load {image_path}"); return
        vis = image_bgr.copy(); center = results.get("fiber_center_px")
        active_zones_vis, scale_vis, suffix_vis = self._get_active_zone_definitions_and_scale()
        if center and scale_vis and scale_vis > 0:
            colors = {"core":(255,0,0),"cladding":(0,255,0),"ferrule_contact":(0,0,255),"adhesive_bond":(255,255,0)}
            for zn,zp in active_zones_vis.items():
                r_max = zp.get(f"r_max{suffix_vis}")
                if r_max is not None: cv2.circle(vis,center,int(r_max/scale_vis),colors.get(zn,(128,128,128)),1)
        masks = results.get("masks_viz",{})
        if masks.get("region_defects") is not None: vis[masks["region_defects"]>0]=cv2.addWeighted(vis[masks["region_defects"]>0],0.5,np.array([0,255,255],dtype=np.uint8),0.5,0)
        if masks.get("scratches") is not None: vis[masks["scratches"]>0]=cv2.addWeighted(vis[masks["scratches"]>0],0.5,np.array([255,0,255],dtype=np.uint8),0.5,0)
        for d in results.get("defects",[]):
            x,y,w,h = d.get("bb_x"),d.get("bb_y"),d.get("bb_w"),d.get("bb_h")
            if all(v is not None for v in [x,y,w,h]): cv2.rectangle(vis,(x,y),(x+w,y+h),(0,165,255),1)
        stat=results.get("status","N/A");clr=(0,255,0) if stat=="PASS" else ((0,165,255) if stat=="UNDEFINED" else (0,0,255))
        cv2.putText(vis,f"Status: {stat}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,clr,2)
        cv2.putText(vis,f"Mode: {results.get('operating_mode','N/A')}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.4,(220,220,220),1)
        eff_scale = results.get('effective_um_per_px','N/A')
        cv2.putText(vis,f"Scale: {eff_scale if isinstance(eff_scale, str) else f'{eff_scale:.4f}'}",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.4,(220,220,220),1)
        cv2.putText(vis,f"Defects: {results.get('defect_count',0)}",(10,75),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        y_off=95
        for i,r in enumerate(results.get("failure_reasons",[])): cv2.putText(vis,r,(10,y_off+i*15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
        if save_path:
            Path(save_path).parent.mkdir(parents=True,exist_ok=True)
            fig,ax=plt.subplots(1,2,figsize=(16,8));fig.suptitle(f"{Path(image_path).name} - Status: {stat}",fontsize=14)
            ax[0].imshow(cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB));ax[0].set_title("Original");ax[0].axis('off')
            ax[1].imshow(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB));ax[1].set_title("Processed");ax[1].axis('off')
            plt.tight_layout(rect=[0,0.03,1,0.95]);plt.savefig(save_path,dpi=150,bbox_inches='tight');plt.close(fig)
            print(f"Visualization saved: {save_path}")
        else: cv2.imshow("Inspection Result",vis);cv2.waitKey(0);cv2.destroyAllWindows()

def calibrate_system(calibration_image_path: str, dot_spacing_um: float = 10.0, config_path: str = "calibration.json") -> Optional[float]:
    image = cv2.imread(calibration_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: print(f"Error: Could not load calib image: {calibration_image_path}"); return None
    blurred = cv2.GaussianBlur(image, (7,7),0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for contour in contours:
        area = cv2.contourArea(contour); perimeter = cv2.arcLength(contour,True)
        if area < 10 or area > 2000 or perimeter == 0: continue # Adjust area filter based on your target
        circularity = 4 * np.pi * (area/(perimeter**2))
        if 0.6 < circularity < 1.4: # Filter for dot-like shapes
            M = cv2.moments(contour)
            if M['m00'] != 0: centroids.append((M['m10']/M['m00'], M['m01']/M['m00']))
    if len(centroids) < 2: print(f"Warning: Not enough distinct circular calib dots found ({len(centroids)})."); return None
    
    # Simplified spacing calculation assuming somewhat regular grid or line. Robust grid analysis is more complex.
    centroids = sorted(centroids, key=lambda c: (c[0],c[1])) 
    distances = []
    if len(centroids) > 1: # More robust: calculate distances between all pairs, find mode or median of plausible distances
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + (centroids[i][1] - centroids[j][1])**2)
                # Filter for plausible distances based on dot_spacing_um and rough pixel estimate
                # This heuristic needs tuning based on expected pixel size of dots
                if dist > 5 : distances.append(dist) 
    
    if not distances: print("Warning: Could not determine any distances between calibration dots."); return None
    
    # Use median of plausible distances. This is still a simplification for a true grid.
    # For a real grid, one might expect distances to cluster around multiples of the dot spacing.
    # A more robust approach would be to find the smallest, most frequent distances.
    avg_dist_px = np.median(sorted(distances)[:min(len(distances), 10)]) # Median of smallest 10, if many dots

    if avg_dist_px < 5: print("Warning: Calib dot spacing in pixels is too small or dots are too close."); return None
    if avg_dist_px == 0: print("Warning: Avg px dist is zero."); return None
    um_per_px = dot_spacing_um / avg_dist_px
    cal_data = {"um_per_px":um_per_px, "dot_spacing_um_used":dot_spacing_um, "avg_dot_dist_px_calculated":avg_dist_px}
    with open(config_path,"w") as f: json.dump(cal_data,f,indent=4)
    print(f"Calibration successful: {um_per_px:.4f} µm/pixel. Saved to {config_path}"); return um_per_px

# --- Main Execution ---
if __name__ == "__main__":
    calibration_file = "calibration.json"
    user_core_dia_um, user_cladding_dia_um = None, None
    if input("Provide specific core/cladding diameters (µm)? (y/n, default: n): ").lower() == 'y':
        try:
            core_str = input("Core diameter (µm, e.g., 9): ").strip()
            if core_str: user_core_dia_um = float(core_str)
            clad_str = input("Cladding diameter (µm, e.g., 125): ").strip()
            if clad_str: user_cladding_dia_um = float(clad_str)
        except ValueError: print("Invalid µm input. Using defaults/pixel mode."); user_core_dia_um=None;user_cladding_dia_um=None

    attempt_calib_load = True
    if Path(calibration_file).exists():
        if input(f"Use existing '{calibration_file}'? (y/n, default: y): ").lower() == 'n':
            attempt_calib_load = False
            print("Calibration file will NOT be used.")
    else: # Calibration file does not exist
        print(f"'{calibration_file}' not found.")
        attempt_calib_load = False
        if user_cladding_dia_um is None: # No calib AND no user cladding for inference
             print("No calibration file and no user-provided cladding diameter. Will operate in PIXEL_ONLY mode.")
        else: # No calib BUT user cladding is available for inference
            print("Will attempt MICRON_INFERRED mode using your provided cladding diameter.")

    try:
        inspector = FiberOpticInspector(
            user_core_diameter_um=user_core_dia_um, user_cladding_diameter_um=user_cladding_dia_um,
            use_calibration_file=attempt_calib_load, calibration_file_path=calibration_file)
    except Exception as e: print(f"FATAL: Init Inspector failed: {e}"); exit()

    image_dir = input("Enter image directory or single image file path: ").strip()
    image_paths = []
    if Path(image_dir).is_file(): image_paths.append(image_dir)
    elif Path(image_dir).is_dir():
        for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"): image_paths.extend(glob.glob(os.path.join(image_dir,ext)))
    else: print(f"Error: '{image_dir}' invalid."); exit()
    if not image_paths: print(f"No images found in '{image_dir}'."); exit()
    print(f"Found {len(image_paths)} image(s).")

    output_dir = Path("./inspection_results"); output_dir.mkdir(parents=True,exist_ok=True)
    summary = []
    for img_path_str in image_paths:
        img_p = Path(img_path_str)
        print(f"\n--- Inspecting: {img_p.name} ---")
        try:
            results = inspector.inspect_fiber(img_path_str)
            print(f"  Status: {results['status']}")
            if results['failure_reasons']: print("  Failure reasons:\n" + "\n".join([f"    - {r}" for r in results['failure_reasons']]))
            print(f"  Total defects found: {results['defect_count']}")
            if results.get('defects') and results['defect_count'] > 0: print("  Defect details:\n" + pd.DataFrame(results['defects']).to_string())
            
            viz_path = output_dir / f"{img_p.stem}_inspected.png"
            inspector.visualize_results(img_path_str, results, save_path=str(viz_path))
            summary.append({"image":img_p.name, "status":results['status'], "defects":results['defect_count'],
                            "mode":results.get("operating_mode"), "eff_um/px":results.get("effective_um_per_px"),
                            "reasons":"; ".join(results.get('failure_reasons',[]))})
        except Exception as e:
            print(f"  ERROR inspecting {img_p.name}: {e}"); import traceback; traceback.print_exc()
            summary.append({"image":img_p.name, "status":"ERROR_PROCESSING", "defects":-1, "reasons":str(e)})
    if summary:
        pd.DataFrame(summary).to_csv(output_dir / "_inspection_summary.csv", index=False)
        print(f"\nSummary saved to: {output_dir / '_inspection_summary.csv'}")
    print("\nBatch inspection complete.")