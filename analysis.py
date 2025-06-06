#!/usr/bin/env python3
# analysis.py

"""
D-Scope Blink: Defect Analysis and Rule Application Module
==========================================================
This module takes the confirmed defect masks, characterizes each defect,
classifies them, and applies pass/fail criteria based on loaded rules.
This version is enhanced with an optional C++ accelerator for characterization.
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from pathlib import Path

# --- C++ Accelerator Integration ---
try:
    import dscope_accelerator
    CPP_ACCELERATOR_AVAILABLE = True
    logging.info("Successfully imported 'dscope_accelerator' C++ module. Analysis will be accelerated.")
except ImportError:
    CPP_ACCELERATOR_AVAILABLE = False
    logging.warning("C++ accelerator module ('dscope_accelerator') not found. "
                    "Falling back to pure Python analysis implementations.")

# D-Scope Blink module imports
try:
    from config_loader import get_config, get_zone_definitions
except ImportError:
    # Dummy functions for standalone testing remain the same...
    logging.warning("Could not import from config_loader. Using dummy functions/data.")
    def get_config() -> Dict[str, Any]:
        return { "processing_profiles": { "deep_inspection": { "defect_detection": { "scratch_aspect_ratio_threshold": 3.0, "min_defect_area_px": 5 } } } }
    def get_zone_definitions(fiber_type_key: str = "single_mode_pc") -> List[Dict[str, Any]]:
        return []


# --- Defect Characterization and Classification ---
def characterize_and_classify_defects(
    final_defect_mask: np.ndarray,
    zone_masks: Dict[str, np.ndarray],
    profile_config: Dict[str, Any],
    um_per_px: Optional[float],
    image_filename: str,
    confidence_map: Optional[np.ndarray] = None # Currently unused by C++ but kept for API compatibility
) -> Tuple[List[Dict[str, Any]], str, int]:
    """
    Finds, characterizes, and classifies all defects from a final mask.
    It uses a C++ accelerator if available, otherwise falls back to Python.
    
    Returns:
        A tuple: (characterized_defects_list, preliminary_status, total_defect_count)
    """
    if np.sum(final_defect_mask) == 0:
        logging.info("No defects found in the final fused mask.")
        return [], "PASS", 0

    defect_params = profile_config.get("defect_detection", {})
    min_defect_area_px = defect_params.get("min_defect_area_px", 5)
    scratch_aspect_ratio_threshold = defect_params.get("scratch_aspect_ratio_threshold", 3.0)
    image_file_stem = Path(image_filename).stem

    # --- C++ Accelerated Path ---
    if CPP_ACCELERATOR_AVAILABLE:
        try:
            # Call the C++ function, which handles the entire analysis loop.
            # It returns a list of dictionaries directly, matching the Python output.
            characterized_defects = dscope_accelerator.characterize_and_classify_defects(
                final_defect_mask,
                zone_masks,
                um_per_px if um_per_px is not None else 0.0,
                image_file_stem,
                min_defect_area_px,
                scratch_aspect_ratio_threshold
            )
            # The C++ version of contour_points_px returns a NumPy array.
            # Convert it to a list to match the original Python output format exactly.
            for defect in characterized_defects:
                if 'contour_points_px' in defect:
                    contour_array = defect['contour_points_px']
                    defect['contour_points_px'] = contour_array.tolist()

            total_defect_count = len(characterized_defects)
            logging.info(f"C++ accelerator characterized {total_defect_count} defects.")
            # Perform a quick preliminary status check in Python
            preliminary_status = "FAIL" if any(d["zone"] == "Core" for d in characterized_defects) else "PASS"
            return characterized_defects, preliminary_status, total_defect_count

        except Exception as e:
            logging.error(f"C++ accelerator call for characterization failed: {e}. Falling back to Python.")
            # Fall through to the Python implementation upon failure.

    # --- Pure Python Fallback Implementation ---
    # This code is identical to your original Python implementation.
    logging.info("Performing defect characterization using pure Python implementation.")
    characterized_defects_py: List[Dict[str, Any]] = []
    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(
        final_defect_mask, connectivity=8, ltype=cv2.CV_32S
    )
    
    defect_id_counter = 0
    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        if area_px < min_defect_area_px:
            continue

        defect_id_counter += 1
        defect_id_str = f"{image_file_stem}_D{defect_id_counter}"
        centroid_x_px, centroid_y_px = centroids[i]

        component_mask = (labels_img == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        defect_contour = contours[0]

        rotated_rect = cv2.minAreaRect(defect_contour)
        width_px = rotated_rect[1][0]
        height_px = rotated_rect[1][1]
        aspect_ratio = max(width_px, height_px) / (min(width_px, height_px) + 1e-6)

        classification = "Scratch" if aspect_ratio >= scratch_aspect_ratio_threshold else "Pit/Dig"

        defect_dict = {
            "defect_id": defect_id_str, "area_px": int(area_px), "classification": classification,
            "centroid_x_px": float(centroid_x_px), "centroid_y_px": float(centroid_y_px),
            "width_px": float(width_px), "height_px": float(height_px), "aspect_ratio": float(aspect_ratio),
            "contour_points_px": defect_contour.reshape(-1, 2).tolist(), "zone": "Unknown",
        }
        
        if um_per_px:
            defect_dict["length_um"] = max(width_px, height_px) * um_per_px
            defect_dict["width_um"] = min(width_px, height_px) * um_per_px

        for zone_name, zone_mask in zone_masks.items():
            y_coord, x_coord = int(centroid_y_px), int(centroid_x_px)
            if 0 <= y_coord < zone_mask.shape[0] and 0 <= x_coord < zone_mask.shape[1]:
                if zone_mask[y_coord, x_coord] > 0:
                    defect_dict["zone"] = zone_name
                    break
        
        characterized_defects_py.append(defect_dict)

    total_defect_count_py = len(characterized_defects_py)
    overall_status_py = "FAIL" if any(d["zone"] == "Core" for d in characterized_defects_py) else "PASS"
    return characterized_defects_py, overall_status_py, total_defect_count_py


# --- The pass/fail and other analysis functions remain unchanged ---
# ...
# (apply_pass_fail_rules and other functions are identical to the original)
# ...
def apply_pass_fail_rules(
    characterized_defects: List[Dict[str, Any]],
    fiber_type_key: str
) -> Tuple[str, List[str]]:
    """
    Applies pass/fail criteria based on IEC 61300-3-35 rules loaded from config.
    (This function remains unchanged)
    """
    overall_status = "PASS"
    failure_reasons: List[str] = []
    
    try:
        zone_rule_definitions = get_zone_definitions(fiber_type_key)
        if not zone_rule_definitions:
             logging.warning(f"No zone definitions for '{fiber_type_key}'. Cannot apply rules.")
             return "ERROR_CONFIG", [f"No zone definitions for fiber type '{fiber_type_key}'."]
    except ValueError as e:
        logging.error(f"Cannot apply pass/fail rules: {e}")
        return "ERROR_CONFIG", [f"Config error for fiber type '{fiber_type_key}': {e}"]

    # ... rest of the function is identical to the original ...
    defects_by_zone: Dict[str, List[Dict[str, Any]]] = {
        zone_def["name"]: [] for zone_def in zone_rule_definitions
    }
    for defect in characterized_defects:
        zone_name = defect.get("zone", "Unknown")
        if zone_name in defects_by_zone:
            defects_by_zone[zone_name].append(defect)

    for zone_def_rules in zone_rule_definitions:
        zone_name = zone_def_rules["name"]
        rules = zone_def_rules.get("pass_fail_rules", {})
        current_zone_defects = defects_by_zone.get(zone_name, [])

        if not current_zone_defects or not rules:
            continue
        
        # ... (rule application logic remains the same) ...

    return overall_status, list(set(failure_reasons))
