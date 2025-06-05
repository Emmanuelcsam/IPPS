#!/usr/bin/env python3
# analysis.py

"""
D-Scope Blink: Defect Analysis and Rule Application Module
==========================================================
This module takes the confirmed defect masks from image_processing.py,
characterizes each defect (size, shape, location), classifies them,
and applies pass/fail criteria based on loaded IEC 61300-3-35 rules
from the configuration.
"""

import cv2 # OpenCV for image processing, especially for contour analysis and minAreaRect.
import numpy as np # NumPy for numerical operations and array manipulation.
from typing import Dict, Any, Optional, List, Tuple, Union # Standard library for type hinting.
import logging # Standard library for logging events.
from pathlib import Path # Standard library for object-oriented path manipulation.

# Attempt to import functions from other D-Scope Blink modules.
try:
    # Assuming config_loader.py is in the same directory or Python path.
    from config_loader import get_config, get_zone_definitions # Functions to access global config and zone rules.
except ImportError:
    # Fallback for standalone testing if config_loader is not directly available.
    logging.warning("Could not import from config_loader. Using dummy functions/data for standalone testing.")
    def get_config() -> Dict[str, Any]: # Define a dummy get_config for standalone testing.
        """Returns a dummy configuration for standalone testing."""
        return { # Simplified dummy config.
            "processing_profiles": {
                "deep_inspection": { # Example profile.
                    "defect_detection": {
                        "scratch_aspect_ratio_threshold": 3.0,
                        "min_defect_area_px": 5
                    }
                }
            },
            "zone_definitions_iec61300_3_35": { # Dummy zone definitions.
                "single_mode_pc": [
                    {"name": "Core", "pass_fail_rules": {"max_scratches": 0, "max_defects": 0, "max_defect_size_um": 3}},
                    {"name": "Cladding", "pass_fail_rules": {"max_scratches": 5, "max_defect_size_um": 10}},
                    {"name": "Adhesive", "pass_fail_rules": {"max_defects": "unlimited", "max_defect_size_um": 50}},
                    {"name": "Contact", "pass_fail_rules": {"max_defects": "unlimited", "max_defect_size_um": 100}}
                ]
            }
        }
    def get_zone_definitions(fiber_type_key: str = "single_mode_pc") -> List[Dict[str, Any]]: # Dummy get_zone_definitions.
        """Returns dummy zone definitions."""
        return get_config()["zone_definitions_iec61300_3_35"].get(fiber_type_key, [])


# --- Defect Characterization and Classification ---
def characterize_and_classify_defects(
    final_defect_mask: np.ndarray,
    zone_masks: Dict[str, np.ndarray],
    profile_config: Dict[str, Any],
    um_per_px: Optional[float],
    image_filename: str,
    confidence_map: Optional[np.ndarray] = None  # Add this parameter
) -> List[Dict[str, Any]]:
    """
    Analyzes connected components in the final defect mask to characterize and classify each defect.

    Args:
        final_defect_mask: Binary mask of confirmed defects from the fusion process.
        zone_masks: Dictionary of binary masks for each inspection zone.
        profile_config: The specific processing profile sub-dictionary from the main config.
        um_per_px: The microns-per-pixel scale for the current image, if available.
        image_filename: The filename of the image being processed.

    Returns:
        A list of dictionaries, where each dictionary represents a characterized defect.
    """
    characterized_defects: List[Dict[str, Any]] = [] # Initialize list to store defect details.
    if np.sum(final_defect_mask) == 0: # Check if there are any defect pixels.
        logging.info("No defects found in the final fused mask.")
        return characterized_defects # Return empty list if no defects.

    # Find connected components (individual defects) in the final mask.
    # stats: [left, top, width, height, area] for each component.
    # centroids: (x, y) for each component.
    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(
        final_defect_mask, connectivity=8, ltype=cv2.CV_32S
    ) #

    logging.info(f"Found {num_labels - 1} potential defect components from fused mask.")

    # Get parameters from config.
    defect_params = profile_config.get("defect_detection", {}) # Get defect detection parameters.
    min_defect_area_px = defect_params.get("min_defect_area_px", 5) # Minimum defect area in pixels.
    scratch_aspect_ratio_threshold = defect_params.get("scratch_aspect_ratio_threshold", 3.0) # Threshold for classifying scratches.

    defect_id_counter = 0 # Initialize defect ID counter.

    # Iterate through each detected component (label 0 is the background).
    for i in range(1, num_labels): # Start from 1 to skip background.
        area_px = stats[i, cv2.CC_STAT_AREA] # Get area of the component.

        if area_px < min_defect_area_px: # Filter out very small components based on config.
            logging.debug(f"Skipping defect component {i} due to small area: {area_px}px < {min_defect_area_px}px.")
            continue # Skip to next component.

        defect_id_counter += 1 # Increment defect ID.
        defect_id_str = f"{Path(image_filename).stem}_D{defect_id_counter}" # Create unique defect ID.

        # Get basic properties from stats.
        x_bbox = stats[i, cv2.CC_STAT_LEFT] # Bounding box left coordinate.
        y_bbox = stats[i, cv2.CC_STAT_TOP] # Bounding box top coordinate.
        w_bbox = stats[i, cv2.CC_STAT_WIDTH] # Bounding box width.
        h_bbox = stats[i, cv2.CC_STAT_HEIGHT] # Bounding box height.
        centroid_x_px, centroid_y_px = centroids[i] # Centroid coordinates.

        # Create a mask for the individual defect component.
        component_mask = (labels_img == i).astype(np.uint8) * 255 # Create mask for current component.
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours of component.
        
        if not contours: # If no contours found for component (should not happen if area > 0).
            logging.warning(f"No contour found for defect component {i} with area {area_px}px. Skipping.")
            continue # Skip.
        
        defect_contour = contours[0] # Assume the largest/only contour for the component.

        # --- Precise Dimension Calculation using minAreaRect ---
        # cv2.minAreaRect returns ((center_x, center_y), (width, height), angle_of_rotation)
        # The width and height from minAreaRect are more accurate for oriented defects.
        rotated_rect = cv2.minAreaRect(defect_contour) # Fit minimum area rotated rectangle.
        rect_center_px = rotated_rect[0] # Center of the rotated rectangle.
        rect_dims_px = tuple(sorted(rotated_rect[1])) # (minor_axis_px, major_axis_px) or (width_px, length_px).
        rect_angle_deg = rotated_rect[2] # Angle of rotation.

        width_oriented_px = rect_dims_px[0] # Shorter side of the rotated rectangle.
        length_oriented_px = rect_dims_px[1] # Longer side of the rotated rectangle.

        # --- Classification (Scratch vs. Pit/Dig) ---
        # Based on aspect ratio of the oriented rectangle.
        aspect_ratio_oriented = length_oriented_px / (width_oriented_px + 1e-6) # Add epsilon to avoid division by zero.
        
        classification: str # Initialize classification string.
        if aspect_ratio_oriented > scratch_aspect_ratio_threshold: # If aspect ratio indicates a scratch.
            classification = "Scratch"
        else: # Otherwise, classify as Pit/Dig.
            classification = "Pit/Dig"
        logging.debug(f"Defect {defect_id_str}: Area={area_px}px, OrientedDims=({width_oriented_px:.1f},{length_oriented_px:.1f})px, AR={aspect_ratio_oriented:.2f} -> {classification}")

        # --- Micron Conversion (if scale is available) ---
        area_um2: Optional[float] = None # Initialize area in microns.
        length_um: Optional[float] = None # Initialize length in microns.
        width_um: Optional[float] = None # Initialize width in microns.
        effective_diameter_um: Optional[float] = None # Initialize effective diameter in microns.

        if um_per_px is not None and um_per_px > 0: # If micron scale is available.
            area_um2 = area_px * (um_per_px ** 2) # Convert area to um^2.
            length_um = length_oriented_px * um_per_px # Convert length to um.
            width_um = width_oriented_px * um_per_px # Convert width to um.
            if classification == "Pit/Dig": # For Pit/Dig, calculate effective diameter.
                # Effective diameter from area is a common metric.
                effective_diameter_um = np.sqrt(4 * area_um2 / np.pi) if area_um2 > 0 else 0.0
        
        # --- Zone Assignment ---
        # Determine which zone the defect's centroid falls into.
        zone_name = "Unknown" # Default zone name.
        for z_name, z_mask in zone_masks.items(): # Iterate through zone masks.
            # Ensure centroid coordinates are within image bounds.
            c_x_int, c_y_int = int(centroid_x_px), int(centroid_y_px) # Convert centroid to int.
            if 0 <= c_y_int < z_mask.shape[0] and 0 <= c_x_int < z_mask.shape[1]: # Check bounds.
                if z_mask[c_y_int, c_x_int] > 0: # If centroid is within the zone mask.
                    zone_name = z_name # Assign zone name.
                    break # Stop checking once zone is found.
        logging.debug(f"Defect {defect_id_str} centroid ({centroid_x_px:.0f},{centroid_y_px:.0f}) assigned to zone: {zone_name}")

        # --- Store Defect Information ---
        # The confidence score would ideally come from the fusion map value at the defect's location,
        # or be an aggregation of contributing algorithm confidences.
        if confidence_map is not None:
            # Sample confidence values at defect location
            defect_mask_single = (labels_img == i).astype(np.uint8)
            confidence_values = confidence_map[defect_mask_single > 0]
            if len(confidence_values) > 0:
                # Use mean confidence value for the defect
                confidence_score = float(np.mean(confidence_values))
            else:
                confidence_score = 0.5  # Default if no confidence data
        else:
            confidence_score = 0.5  # Default if no confidence map provided

        defect_data = { # Create dictionary for defect data.
            "defect_id": defect_id_str,
            "zone": zone_name,
            "classification": classification,
            "confidence_score": confidence_score_placeholder, # Placeholder.
            "centroid_x_px": round(centroid_x_px, 2),
            "centroid_y_px": round(centroid_y_px, 2),
            "area_px": round(area_px, 2),
            "length_px": round(length_oriented_px, 2), # Oriented length.
            "width_px": round(width_oriented_px, 2),   # Oriented width.
            "aspect_ratio_oriented": round(aspect_ratio_oriented, 2),
            "bbox_x_px": x_bbox, # Bounding box (axis-aligned).
            "bbox_y_px": y_bbox,
            "bbox_w_px": w_bbox,
            "bbox_h_px": h_bbox,
            "rotated_rect_center_px": (round(rect_center_px[0],2), round(rect_center_px[1],2)),
            "rotated_rect_angle_deg": round(rect_angle_deg,2),
            "contour_points_px": defect_contour.squeeze().tolist() # Store contour points.
        }
        if um_per_px is not None: # If micron scale is available.
            defect_data["area_um2"] = round(area_um2, 2) if area_um2 is not None else None
            defect_data["length_um"] = round(length_um, 2) if length_um is not None else None
            defect_data["width_um"] = round(width_um, 2) if width_um is not None else None
            if classification == "Pit/Dig" and effective_diameter_um is not None: # If Pit/Dig and diameter calculated.
                defect_data["effective_diameter_um"] = round(effective_diameter_um, 2)

        characterized_defects.append(defect_data) # Add defect data to list.

    logging.info(f"Characterized and classified {len(characterized_defects)} defects.")
    return characterized_defects # Return list of characterized defects.

def calculate_defect_density(defects: List[Dict[str, Any]], zone_area_px: float) -> float:
    """
    Calculates defect density (defects per unit area).
    """
    total_defect_area = sum(d.get('area_px', 0) for d in defects)
    return total_defect_area / zone_area_px if zone_area_px > 0 else 0

# --- Pass/Fail Evaluation ---
def apply_pass_fail_rules(
    characterized_defects: List[Dict[str, Any]],
    fiber_type_key: str # e.g., "single_mode_pc", to fetch correct rules from config.
) -> Tuple[str, List[str]]:
    """
    Applies pass/fail criteria based on IEC 61300-3-35 rules loaded from config.

    Args:
        characterized_defects: List of defect dictionaries from characterization.
        fiber_type_key: The key for the fiber type to retrieve specific zone rules.

    Returns:
        A tuple: (overall_status: str, failure_reasons: List[str])
                 Overall status is "PASS" or "FAIL".
    """
    overall_status = "PASS" # Initialize overall status to PASS.
    failure_reasons: List[str] = [] # Initialize list for failure reasons.

    try:
        # Get zone definitions which include pass/fail rules.
        zone_rule_definitions = get_zone_definitions(fiber_type_key) # Fetch zone definitions from config.
    except ValueError as e: # Handle if fiber type not found in config.
        logging.error(f"Cannot apply pass/fail rules: {e}")
        return "ERROR_CONFIG", [f"Configuration error for fiber type '{fiber_type_key}': {e}"] # Return error status.

    # Group defects by zone for easier rule application.
    defects_by_zone: Dict[str, List[Dict[str, Any]]] = { # Initialize dictionary to group defects by zone.
        zone_def["name"]: [] for zone_def in zone_rule_definitions
    }
    for defect in characterized_defects: # Iterate through characterized defects.
        if defect["zone"] in defects_by_zone: # If defect's zone is in defined zones.
            defects_by_zone[defect["zone"]].append(defect) # Add defect to corresponding zone group.
        elif defect["zone"] != "Unknown": # If defect zone is not "Unknown" but not in definitions.
            logging.warning(f"Defect {defect['defect_id']} in zone '{defect['zone']}' which has no defined rules for fiber type '{fiber_type_key}'.")

    # Apply rules for each zone.
    for zone_def_rules in zone_rule_definitions: # Iterate through zone rule definitions.
        zone_name = zone_def_rules["name"] # Get zone name.
        rules = zone_def_rules.get("pass_fail_rules", {}) # Get pass/fail rules for the zone.
        current_zone_defects = defects_by_zone.get(zone_name, []) # Get defects in current zone.

        if not current_zone_defects: continue # Skip if no defects in this zone.

        # --- Rule: Max defect count (overall for zone) ---
        # This is a simplified example; IEC rules are more granular (per defect type).
        # The config example has max_scratches, max_defects, max_defect_size_um
        
        # Separate defects in zone by classification for rule checking
        scratches_in_zone = [d for d in current_zone_defects if d["classification"] == "Scratch"] # Get scratches.
        pits_digs_in_zone = [d for d in current_zone_defects if d["classification"] == "Pit/Dig"] # Get pits/digs.

        # Check scratch count
        max_scratches_allowed = rules.get("max_scratches") # Get max allowed scratches.
        if isinstance(max_scratches_allowed, int) and len(scratches_in_zone) > max_scratches_allowed: # Check count.
            overall_status = "FAIL" # Set status to FAIL.
            failure_reasons.append(f"Zone '{zone_name}': Too many scratches ({len(scratches_in_zone)} > {max_scratches_allowed}).")
            logging.warning(f"FAIL Rule (Scratch Count): Zone '{zone_name}', Count={len(scratches_in_zone)}, Allowed={max_scratches_allowed}")


        # Check "Pit/Dig" count (config key "max_defects" often refers to these)
        max_pits_digs_allowed = rules.get("max_defects") # Get max allowed pits/digs.
        if isinstance(max_pits_digs_allowed, int) and len(pits_digs_in_zone) > max_pits_digs_allowed: # Check count.
            overall_status = "FAIL" # Set status to FAIL.
            failure_reasons.append(f"Zone '{zone_name}': Too many Pits/Digs ({len(pits_digs_in_zone)} > {max_pits_digs_allowed}).")
            logging.warning(f"FAIL Rule (Pit/Dig Count): Zone '{zone_name}', Count={len(pits_digs_in_zone)}, Allowed={max_pits_digs_allowed}")

        # Check defect sizes
        max_defect_size_um_allowed = rules.get("max_defect_size_um") # Max overall defect size.
        max_scratch_length_um_allowed = rules.get("max_scratch_length_um") # Max scratch length (if specified separately).

        for defect in current_zone_defects: # Iterate through defects in zone.
            primary_dimension_um: Optional[float] = None # Initialize primary dimension.
            defect_type_for_rule = defect["classification"] # Get defect classification.
            
            if defect_type_for_rule == "Scratch": # If defect is a scratch.
                primary_dimension_um = defect.get("length_um") # Use length.
                current_max_size_rule = max_scratch_length_um_allowed if max_scratch_length_um_allowed is not None else max_defect_size_um_allowed # Use specific scratch rule or general.
            else: # If defect is Pit/Dig.
                primary_dimension_um = defect.get("effective_diameter_um", defect.get("length_um")) # Use diameter, fallback to length (max of oriented box).
                current_max_size_rule = max_defect_size_um_allowed # Use general defect size rule.

            if primary_dimension_um is not None and isinstance(current_max_size_rule, (int, float)): # If dimension and rule exist.
                if primary_dimension_um > current_max_size_rule: # Check size against rule.
                    overall_status = "FAIL" # Set status to FAIL.
                    reason = f"Zone '{zone_name}': {defect_type_for_rule} '{defect['defect_id']}' size ({primary_dimension_um:.2f}µm) exceeds limit ({current_max_size_rule}µm)."
                    failure_reasons.append(reason)
                    logging.warning(f"FAIL Rule (Defect Size): {reason}")
            
            # Example for "max_scratches_gt_5um" from config example
            if defect_type_for_rule == "Scratch" and rules.get("max_scratches_gt_5um") == 0:
                limit_size_for_zero_count = 5.0 # Example, should be from config if more generic needed
                if primary_dimension_um is not None and primary_dimension_um > limit_size_for_zero_count:
                    overall_status = "FAIL"
                    reason = f"Zone '{zone_name}': Scratch '{defect['defect_id']}' size ({primary_dimension_um:.2f}µm) found, but no scratches > {limit_size_for_zero_count}µm allowed."
                    failure_reasons.append(reason)
                    logging.warning(f"FAIL Rule (Specific Scratch Size Count): {reason}")


    if not failure_reasons and overall_status == "PASS": # If no failures and status is PASS.
        logging.info("Pass/Fail Evaluation: Overall PASS.")
    else: # If failures or status is FAIL.
        logging.warning(f"Pass/Fail Evaluation: Overall {overall_status}. Reasons: {'; '.join(failure_reasons)}")
        
    return overall_status, list(set(failure_reasons)) # Return unique failure reasons.

# --- Main function for testing this module (optional) ---
if __name__ == "__main__":
    # This block is for testing the analysis module independently.
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s') # Basic logging.

    # --- Dummy Data for Testing ---
    dummy_image_filename = "test_img.png" # Dummy filename.
    dummy_um_per_px = 0.5  # Example: 0.5 microns per pixel.

    # Create a dummy final_defect_mask (e.g., 100x100 image with two defects)
    dummy_mask = np.zeros((200, 200), dtype=np.uint8) # Initialize dummy mask.
    # Defect 1 (Scratch-like)
    cv2.rectangle(dummy_mask, (20, 30), (25, 100), 255, -1) # Draw rectangle for defect 1.
    # Defect 2 (Pit/Dig-like)
    cv2.circle(dummy_mask, (100, 100), 10, 255, -1) # Draw circle for defect 2.
    # Defect 3 (Small, should be filtered by area)
    cv2.rectangle(dummy_mask, (150,150), (152,152), 255, -1) # Draw small defect.


    # Dummy zone masks (ensure they cover the defect areas for testing)
    dummy_zone_masks = { # Initialize dummy zone masks.
        "Core": np.zeros((200, 200), dtype=np.uint8),
        "Cladding": np.zeros((200, 200), dtype=np.uint8)
    }
    cv2.circle(dummy_zone_masks["Core"], (100, 100), 50, 255, -1)  # Core covers defect 2.
    cv2.circle(dummy_zone_masks["Cladding"], (100, 100), 100, 255, -1) # Draw cladding circle.
    cv2.circle(dummy_zone_masks["Cladding"], (100, 100), 50, 0, -1) # Create annulus for cladding (defect 1 in cladding).
    # Ensure defect 1 (20,30)-(25,100) falls into Cladding or Core for testing.
    # Adjusting Core mask to not overlap with defect 1 for clearer zone assignment test
    cv2.circle(dummy_zone_masks["Core"], (100, 100), 40, 255, -1) # Make core smaller
    # Defect 1 (22, 65) should be in Cladding. Defect 2 (100,100) should be in Core.

    dummy_profile_cfg = get_config()["processing_profiles"]["deep_inspection"] # Get dummy profile config.


    # --- Test Case 1: Characterize and Classify Defects ---
    logging.info("\n--- Test Case 1: Characterize and Classify Defects ---")
    characterized = characterize_and_classify_defects( # Characterize defects.
        dummy_mask, dummy_zone_masks, dummy_profile_cfg, dummy_um_per_px, dummy_image_filename
    )
    if characterized: # If defects characterized.
        logging.info(f"Characterized {len(characterized)} defects:")
        for defect_item in characterized: # Iterate through characterized defects.
            logging.info(f"  {defect_item}")
    else: # If no defects characterized.
        logging.info("No defects characterized.")

    # --- Test Case 2: Apply Pass/Fail Rules ---
    logging.info("\n--- Test Case 2: Apply Pass/Fail Rules ---")
    if characterized: # If defects characterized.
        # Assume "single_mode_pc" for fetching rules from dummy_config.
        status, reasons = apply_pass_fail_rules(characterized, "single_mode_pc") # Apply pass/fail rules.
        logging.info(f"Pass/Fail Status: {status}")
        if reasons: # If failure reasons exist.
            logging.info("Failure Reasons:")
            for reason_item in reasons: # Iterate through reasons.
                logging.info(f"  - {reason_item}")
    else: # If no defects characterized.
        logging.warning("Skipping pass/fail test as no defects were characterized.")

    # Example: A defect that should fail Core rules
    failing_core_defect = [{ # Define a failing core defect.
        "defect_id": "test_img_D3", "zone": "Core", "classification": "Pit/Dig",
        "confidence_score": 1.0, "centroid_x_px": 100, "centroid_y_px": 100,
        "area_px": 314, "length_px": 20, "width_px": 20,
        "aspect_ratio_oriented": 1.0, "area_um2": 78.5, "length_um": 10, "width_um": 10,
        "effective_diameter_um": 10.0 # This should fail (max_defect_size_um: 3 for Core)
    }]
    status_fail, reasons_fail = apply_pass_fail_rules(failing_core_defect, "single_mode_pc") # Apply rules.
    logging.info(f"\nTest Failing Core Defect -> Status: {status_fail}, Reasons: {reasons_fail}")
    assert status_fail == "FAIL" # Assert that status is FAIL.

    # Example: A defect that should pass Cladding rules but many of them
    passing_cladding_defects = [] # Initialize list for passing cladding defects.
    for k in range(6): # Create 6 defects.
        passing_cladding_defects.append({
            "defect_id": f"test_img_D{k+4}", "zone": "Cladding", "classification": "Scratch",
            "confidence_score": 1.0, "centroid_x_px": 25, "centroid_y_px": 50+k*5,
            "area_px": 20, "length_px": 20, "width_px": 1, "aspect_ratio_oriented": 20.0,
            "area_um2": 5, "length_um": 10, "width_um": 0.5 # length 10um should be fine if general max_defect_size_um=10
        })
    # The single_mode_pc rule for Cladding has "max_scratches": 5
    status_pass_count, reasons_pass_count = apply_pass_fail_rules(passing_cladding_defects, "single_mode_pc") # Apply rules.
    logging.info(f"\nTest Cladding Scratch Count -> Status: {status_pass_count}, Reasons: {reasons_pass_count}")
    assert status_pass_count == "FAIL" # Assert that status is FAIL due to count.
