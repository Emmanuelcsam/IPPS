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
    confidence_map: Optional[np.ndarray] = None
) -> Tuple[List[Dict[str, Any]], str, int]:
    """
    Returns:
        characterized_defects, overall_status, total_defect_count
    """
    characterized_defects: List[Dict[str, Any]] = [] # Initialize list to store defect details.
    if np.sum(final_defect_mask) == 0: # Check if there are any defect pixels.
        logging.info("No defects found in the final fused mask.")
        return characterized_defects, "PASS", 0 # Return empty list, PASS status, and 0 count.

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
        rotated_rect = cv2.minAreaRect(defect_contour)
        # box_points = cv2.boxPoints(rotated_rect) # Not strictly needed for dimensions but useful for drawing
        # box_points = np.intp(box_points)

        # Compute dimensions in pixels
        width_px = rotated_rect[1][0]
        height_px = rotated_rect[1][1]
        aspect_ratio = max(width_px, height_px) / (min(width_px, height_px) + 1e-6) # Add epsilon to prevent div by zero


        perimeter = cv2.arcLength(defect_contour, True)
        circularity = 4 * np.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0
        
        # Calculate solidity (ratio of contour area to convex hull area)
        hull = cv2.convexHull(defect_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_px / hull_area if hull_area > 0 else 0
        
        # Calculate extent (ratio of contour area to bounding rectangle area)
        rect_area = w_bbox * h_bbox
        extent = area_px / rect_area if rect_area > 0 else 0
        
        # Get oriented bounding box for better dimension calculation
        rotated_rect = cv2.minAreaRect(defect_contour)
        (cx_rr, cy_rr), (width_rr, height_rr), angle = rotated_rect
        
        # Ensure width is the longer dimension
        if height_rr > width_rr:
            width_rr, height_rr = height_rr, width_rr
        
        aspect_ratio = width_rr / (height_rr + 1e-6)
        
        # Enhanced classification criteria based on paper
        # Scratches: high aspect ratio, low circularity, low solidity, low extent
        # Pits: low aspect ratio, high circularity, high solidity, high extent
        if aspect_ratio >= scratch_aspect_ratio_threshold and circularity < 0.4 and solidity < 0.7 and extent < 0.5:
            classification = "Scratch"
        elif aspect_ratio < 2.0 and circularity > 0.6 and solidity > 0.8 and extent > 0.7:
            classification = "Pit/Dig"
        else:
            # Ambiguous cases - use weighted scoring
            scratch_score = (aspect_ratio / 10.0) + (1 - circularity) + (1 - solidity) + (1 - extent)
            pit_score = (1 / (aspect_ratio + 0.1)) + circularity + solidity + extent
            
            if scratch_score > pit_score:
                classification = "Scratch"
            else:
                classification = "Pit/Dig"

        # Compute size in microns if um_per_px provided
        length_um = None
        width_um = None
        if um_per_px:
            length_um = max(width_px, height_px) * um_per_px
            width_um = min(width_px, height_px) * um_per_px

        # Build defect dict
        defect_dict = {
            "defect_id": defect_id_str,
            "contour_points_px": defect_contour.reshape(-1, 2).tolist(),
            "bbox_x_px": x_bbox,
            "bbox_y_px": y_bbox,
            "bbox_w_px": w_bbox,
            "bbox_h_px": h_bbox,
            "centroid_x_px": float(centroid_x_px),
            "centroid_y_px": float(centroid_y_px),
            "area_px": int(area_px),
            "width_px": float(width_px), # From minAreaRect
            "height_px": float(height_px), # From minAreaRect
            "aspect_ratio": float(aspect_ratio),
            "classification": classification,
            "length_um": length_um, # Max dimension in um
            "width_um": width_um,   # Min dimension in um
            "zone": "Unknown",  # Default, to be updated
        }

        # Determine zone based on centroid
        for zone_name, zone_mask in zone_masks.items():
            # Ensure centroid coordinates are within mask bounds
            y_coord = int(centroid_y_px)
            x_coord = int(centroid_x_px)
            if 0 <= y_coord < zone_mask.shape[0] and 0 <= x_coord < zone_mask.shape[1]:
                if zone_mask[y_coord, x_coord] > 0:
                    defect_dict["zone"] = zone_name
                    break
            else:
                logging.warning(
                    f"Defect {defect_id_str} centroid ({x_coord}, {y_coord}) is outside zone mask dimensions {zone_mask.shape}."
                )


        characterized_defects.append(defect_dict)

    # After looping through components:
    total_defect_count = len(characterized_defects)
    overall_status = "PASS" # Default to PASS, apply_pass_fail_rules will give the final verdict.
    # Placeholder default rules: e.g., any defect in core might be a quick preliminary FAIL.
    # The more detailed apply_pass_fail_rules function is the primary method for pass/fail.
    # For this preliminary check:
    for d in characterized_defects:
        # Example: A simple preliminary check, like failing if anything is in the Core.
        # The more robust size checks are handled by apply_pass_fail_rules using zone-specific rules.
        if d["zone"] == "Core":
            overall_status = "FAIL" # Preliminarily FAIL if a defect is in the Core zone.
            # The original placeholder size check against a general profile config key was:
            # if d["zone"] == "Core" or (
            #    d.get("length_um", 0) and um_per_px and 
            #    d["length_um"] > profile_config.get("defect_detection", {}).get("max_defect_size_um", float('inf'))
            # ):
            # Using float('inf') as default if key is missing makes it less likely to fail unexpectedly due to missing config.
            # However, this overall_status is often superseded by apply_pass_fail_rules.
            break # Exit loop after first preliminary failure condition.

    return characterized_defects, overall_status, total_defect_count


def calculate_defect_density(defects: List[Dict[str, Any]], zone_area_px: float) -> float:
    """
    Calculates defect density (defects per unit area).
    """
    total_defect_area = sum(d.get('area_px', 0) for d in defects)
    return total_defect_area / zone_area_px if zone_area_px > 0 else 0

def analyze_defects_by_zone(characterized_defects: List[Dict[str, Any]], 
                           zone_masks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """
    Perform detailed region-specific analysis of defects.
    
    Args:
        characterized_defects: List of characterized defect dictionaries
        zone_masks: Dictionary of zone masks
        
    Returns:
        Dictionary with zone-specific statistics
    """
    zone_stats = {}
    
    for zone_name, zone_mask in zone_masks.items():
        # Use bitwise_and to isolate defects in this zone
        zone_defects = [d for d in characterized_defects if d.get('zone') == zone_name]
        
        # Calculate zone area for density calculations
        zone_area_px = np.sum(zone_mask > 0)
        
        # Separate by type
        scratches = [d for d in zone_defects if d['classification'] == 'Scratch']
        pits_digs = [d for d in zone_defects if d['classification'] == 'Pit/Dig']
        
        # Calculate statistics
        total_defect_area = sum(d.get('area_px', 0) for d in zone_defects)
        defect_density = total_defect_area / zone_area_px if zone_area_px > 0 else 0
        
        # Size statistics
        defect_sizes = [d.get('length_um', d.get('length_px', 0)) for d in zone_defects]
        
        zone_stats[zone_name] = {
            'total_defects': len(zone_defects),
            'scratch_count': len(scratches),
            'pit_dig_count': len(pits_digs),
            'total_area_px': total_defect_area,
            'defect_density': defect_density,
            'zone_area_px': zone_area_px,
            'max_defect_size': max(defect_sizes) if defect_sizes else 0,
            'avg_defect_size': np.mean(defect_sizes) if defect_sizes else 0,
            'defects': zone_defects
        }
        
        logging.info(f"Zone '{zone_name}': {len(zone_defects)} defects "
                     f"({len(scratches)} scratches, {len(pits_digs)} pits/digs), "
                     f"density: {defect_density:.4f}")
    
    return zone_stats

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
        if not zone_rule_definitions:
             logging.warning(f"No zone definitions found for fiber type '{fiber_type_key}'. Cannot apply rules.")
             return "ERROR_CONFIG", [f"No zone definitions for fiber type '{fiber_type_key}'."]

    except ValueError as e: # Handle if fiber type not found in config.
        logging.error(f"Cannot apply pass/fail rules: {e}")
        return "ERROR_CONFIG", [f"Configuration error for fiber type '{fiber_type_key}': {e}"] # Return error status.

    # Group defects by zone for easier rule application.
    defects_by_zone: Dict[str, List[Dict[str, Any]]] = { # Initialize dictionary to group defects by zone.
        zone_def["name"]: [] for zone_def in zone_rule_definitions
    }
    for defect in characterized_defects: # Iterate through characterized defects.
        zone_name = defect.get("zone", "Unknown")
        if zone_name in defects_by_zone: # If defect's zone is in defined zones.
            defects_by_zone[zone_name].append(defect) # Add defect to corresponding zone group.
        elif zone_name != "Unknown": # If defect zone is not "Unknown" but not in definitions.
            # This case might occur if a zone exists in zone_masks but not in zone_rule_definitions
            logging.warning(f"Defect {defect['defect_id']} in zone '{zone_name}' which has no defined rules for fiber type '{fiber_type_key}'. This defect will not be evaluated against specific zone rules.")
            # Optionally, you could have a default rule set for "Unknown" zones or unclassified zones.
        # Defects in "Unknown" zones are implicitly ignored by the rule loop below unless "Unknown" is a defined zone.


    # Apply rules for each zone.
    for zone_def_rules in zone_rule_definitions: # Iterate through zone rule definitions.
        zone_name = zone_def_rules["name"] # Get zone name.
        rules = zone_def_rules.get("pass_fail_rules", {}) # Get pass/fail rules for the zone.
        current_zone_defects = defects_by_zone.get(zone_name, []) # Get defects in current zone.

        # If no defects in this zone, or no rules defined for this zone, skip.
        if not current_zone_defects and not rules:
            logging.debug(f"No defects or rules for zone '{zone_name}'. Skipping.")
            continue
        
        if not rules:
            logging.debug(f"No specific pass/fail rules defined for zone '{zone_name}' in config. Defects in this zone will not cause failure based on these rules.")
            continue


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
        max_defect_size_um_allowed = rules.get("max_defect_size_um") # Max overall defect size for this zone.
        max_scratch_length_um_allowed = rules.get("max_scratch_length_um") # Max scratch length for this zone (if specified separately).

        for defect in current_zone_defects: # Iterate through defects in zone.
            primary_dimension_um: Optional[float] = None # Initialize primary dimension.
            defect_type_for_rule = defect["classification"] # Get defect classification.
            
            if defect_type_for_rule == "Scratch": # If defect is a scratch.
                primary_dimension_um = defect.get("length_um") # Use length_um (max dimension of oriented box).
                # Use specific scratch length rule if available, otherwise general defect size rule for this zone.
                current_max_size_rule = max_scratch_length_um_allowed if max_scratch_length_um_allowed is not None else max_defect_size_um_allowed
            else: # If defect is Pit/Dig.
                 # For Pits/Digs, 'length_um' (max dimension of oriented box) is used here.
                 # If 'effective_diameter_um' were calculated and stored in defect_dict, it could be preferred:
                 # primary_dimension_um = defect.get("effective_diameter_um", defect.get("length_um"))
                primary_dimension_um = defect.get("length_um")
                current_max_size_rule = max_defect_size_um_allowed # Use general defect size rule for this zone.

            if primary_dimension_um is not None and isinstance(current_max_size_rule, (int, float)): # If dimension and rule exist.
                if primary_dimension_um > current_max_size_rule: # Check size against rule.
                    overall_status = "FAIL" # Set status to FAIL.
                    reason = f"Zone '{zone_name}': {defect_type_for_rule} '{defect['defect_id']}' size ({primary_dimension_um:.2f}µm) exceeds limit ({current_max_size_rule}µm)."
                    failure_reasons.append(reason)
                    logging.warning(f"FAIL Rule (Defect Size): {reason}")
            
            # Example for a specific rule like "max_scratches_gt_5um" (if it were in config)
            # This rule requires "max_scratches_gt_5um": 0 in the config to trigger this specific check.
            specific_scratch_size_limit_key = "max_scratches_gt_5um" # Example key
            specific_scratch_size_limit_value = 5.0 # Example size threshold for this rule
            
            if defect_type_for_rule == "Scratch" and rules.get(specific_scratch_size_limit_key) == 0:
                if primary_dimension_um is not None and primary_dimension_um > specific_scratch_size_limit_value:
                    overall_status = "FAIL"
                    reason = (
                        f"Zone '{zone_name}': Scratch '{defect['defect_id']}' size ({primary_dimension_um:.2f}µm) "
                        f"found, but no scratches > {specific_scratch_size_limit_value}µm allowed by rule '{specific_scratch_size_limit_key}'."
                    )
                    failure_reasons.append(reason)
                    logging.warning(f"FAIL Rule (Specific Scratch Size Count): {reason}")


    if not failure_reasons and overall_status == "PASS": # If no failures and status is PASS.
        logging.info(f"Pass/Fail Evaluation for '{fiber_type_key}': Overall PASS.")
    elif overall_status == "FAIL": # If failures or status is FAIL.
        logging.warning(f"Pass/Fail Evaluation for '{fiber_type_key}': Overall FAIL. Reasons: {'; '.join(list(set(failure_reasons)))}")
    # If status is not FAIL but reasons were added (e.g. from an ERROR_CONFIG state), log them.
    elif failure_reasons:
         logging.error(f"Pass/Fail Evaluation for '{fiber_type_key}': Status {overall_status}. Issues: {'; '.join(list(set(failure_reasons)))}")


    return overall_status, list(set(failure_reasons)) # Return unique failure reasons.

# --- Main function for testing this module (optional) ---
if __name__ == "__main__":
    # This block is for testing the analysis module independently.
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s') # Basic logging.

    # --- Dummy Data for Testing ---
    dummy_image_filename = "test_img.png" # Dummy filename.
    dummy_um_per_px = 0.5  # Example: 0.5 microns per pixel.

    # Create a dummy final_defect_mask (e.g., 100x100 image with two defects)
    dummy_mask = np.zeros((200, 200), dtype=np.uint8) # Initialize dummy mask.
    # Defect 1 (Scratch-like) - CV: (20,30) to (25,100), centroid approx (22.5, 65)
    # cv2.rectangle(dummy_mask, (20, 30), (25, 100), (255), -1) # Draw rectangle for defect 1. Area = 5 * 70 = 350
    cv2.rectangle(dummy_mask, (20, 30), (25, 100), (255,), -1) # Draw rectangle for defect 1. Area = 5 * 70 = 350
    # Defect 2 (Pit/Dig-like) - CV: center (100,100), radius 10. Area approx pi*10^2 = 314
    # cv2.circle(dummy_mask, (100, 100), 10, (255), -1) # Draw circle for defect 2.
    cv2.circle(dummy_mask, (100, 100), 10, (255,), -1) # Draw circle for defect 2.
    # Defect 3 (Small, should be filtered by area if min_defect_area_px is e.g. 5) Area = 2*2=4
    # cv2.rectangle(dummy_mask, (150,150), (152,152), (255), -1) # Draw small defect.
    cv2.rectangle(dummy_mask, (150,150), (152,152), (255,), -1) # Draw small defect.


    # Dummy zone masks (ensure they cover the defect areas for testing)
    # Dummy zone masks (ensure they cover the defect areas for testing)
    dummy_zone_masks = { # Initialize dummy zone masks.
        "Core": np.zeros((200, 200), dtype=np.uint8),
        "Cladding": np.zeros((200, 200), dtype=np.uint8),
        "Adhesive": np.zeros((200,200), dtype=np.uint8) # Adding for completeness
    }
    
    
    # Core centered at (100,100) with radius 40. Defect 2 (100,100) should be in Core.
    # cv2.circle(dummy_zone_masks["Core"], (100, 100), 40, (255), -1)
    cv2.circle(dummy_zone_masks["Core"], (100, 100), 40, (255,), -1)
    # Cladding as an annulus around Core. Radius 80 for outer, 40 for inner.
    # Defect 1 centroid (22.5, 65) should fall in this Cladding.
    # cv2.circle(dummy_zone_masks["Cladding"], (100, 100), 80, (255), -1) # Outer cladding circle
    cv2.circle(dummy_zone_masks["Cladding"], (100, 100), 80, (255,), -1) # Outer cladding circle
    # cv2.circle(dummy_zone_masks["Cladding"], (100, 100), 40, (0), -1)   # "Subtract" core area to make it an annulus
    cv2.circle(dummy_zone_masks["Cladding"], (100, 100), 40, (0,), -1)   # "Subtract" core area to make it an annulus
    # Make Adhesive zone cover a different area
    # cv2.rectangle(dummy_zone_masks["Adhesive"], (0,0), (200,20), (255), -1)
    cv2.rectangle(dummy_zone_masks["Adhesive"], (0,0), (200,20), (255,), -1)


    dummy_profile_cfg = get_config()["processing_profiles"]["deep_inspection"] # Get dummy profile config.
    # Set min_defect_area_px for this test to ensure defect 3 is filtered
    dummy_profile_cfg["defect_detection"]["min_defect_area_px"] = 5


    # --- Test Case 1: Characterize and Classify Defects ---
    logging.info("\n--- Test Case 1: Characterize and Classify Defects ---")
    # Unpack all return values from the function
    characterized_defects_list, initial_overall_status, total_defect_count_val = characterize_and_classify_defects(
        dummy_mask, dummy_zone_masks, dummy_profile_cfg, dummy_um_per_px, dummy_image_filename
    )
    logging.info(f"Initial characterization status: {initial_overall_status}, Total defects found (after filtering): {total_defect_count_val}")

    if characterized_defects_list: # Check if the list of defects is not empty.
        logging.info(f"Characterized {len(characterized_defects_list)} defects:")
        for defect_item in characterized_defects_list: # Iterate through the list of defects.
            logging.info(f"  ID: {defect_item['defect_id']}, Class: {defect_item['classification']}, Zone: {defect_item['zone']}, AreaPx: {defect_item['area_px']}, LengthUm: {defect_item.get('length_um', 'N/A')}")
    else: # If no defects characterized.
        logging.info("No defects characterized by the function (either none found or all filtered).")

    # --- Test Case 2: Apply Pass/Fail Rules ---
    logging.info("\n--- Test Case 2: Apply Pass/Fail Rules (using defects from Test Case 1) ---")
    # Use the characterized_defects_list for applying pass/fail rules.
    status_tc2, reasons_tc2 = apply_pass_fail_rules(characterized_defects_list, "single_mode_pc")
    logging.info(f"Pass/Fail Status (from apply_pass_fail_rules TC2): {status_tc2}")
    if reasons_tc2: # If failure reasons exist.
        logging.info("Failure Reasons (TC2):")
        for reason_item in reasons_tc2: # Iterate through reasons.
            logging.info(f"  - {reason_item}")
    elif status_tc2 == "PASS":
        logging.info("No failure reasons (TC2): Overall PASS.")

    # Expected: Defect 1 (Scratch, in Cladding, length 70px*0.5=35um) -> Cladding max_defect_size_um: 10. FAIL.
    # Expected: Defect 2 (Pit/Dig, in Core, approx diameter 20px*0.5=10um) -> Core max_defect_size_um: 3. FAIL.
    # Defect 3 is filtered out (area 4px < 5px min_defect_area_px).
    # So, TC2 should result in FAIL.

    # --- Test Case 3: Failing Core Defect (manual example) ---
    logging.info("\n--- Test Case 3: Failing Core Defect (manual) ---")
    failing_core_defect = [{ # Define a failing core defect.
        "defect_id": "test_img_D-CoreFail", "zone": "Core", "classification": "Pit/Dig",
        "confidence_score": 1.0, "centroid_x_px": 100, "centroid_y_px": 100,
        "area_px": 314, "length_px": 20, "width_px": 20, "aspect_ratio": 1.0,
        "area_um2": 78.5, "length_um": 10.0, "width_um": 10.0 # length_um 10.0 > 3um (Core rule)
        # "effective_diameter_um": 10.0 # If this key was used and populated
    }]
    status_fail_core, reasons_fail_core = apply_pass_fail_rules(failing_core_defect, "single_mode_pc") # Apply rules.
    logging.info(f"Test Failing Core Defect -> Status: {status_fail_core}, Reasons: {reasons_fail_core}")
    assert status_fail_core == "FAIL", f"Expected FAIL for TC3, got {status_fail_core}"

    # --- Test Case 4: Cladding Scratch Count (manual example) ---
    logging.info("\n--- Test Case 4: Cladding Scratch Count (manual) ---")
    passing_cladding_defects_list = [] # Initialize list for passing cladding defects.
    # Cladding rule: "max_scratches": 5
    for k in range(6): # Create 6 scratches in Cladding.
        passing_cladding_defects_list.append({
            "defect_id": f"test_img_D-CladScratch{k+1}", "zone": "Cladding", "classification": "Scratch",
            "confidence_score": 1.0, "centroid_x_px": 25, "centroid_y_px": 50+k*5,
            "area_px": 20, "length_px": 20, "width_px": 1, "aspect_ratio": 20.0,
            "area_um2": 5, "length_um": 10.0, "width_um": 0.5 # length 10um, Cladding max_defect_size_um is 10. This size is OK.
        })
    status_clad_count, reasons_clad_count = apply_pass_fail_rules(passing_cladding_defects_list, "single_mode_pc") # Apply rules.
    logging.info(f"Test Cladding Scratch Count (6 scratches) -> Status: {status_clad_count}, Reasons: {reasons_clad_count}")
    assert status_clad_count == "FAIL", f"Expected FAIL for TC4 due to scratch count, got {status_clad_count}"
    assert any("Too many scratches" in reason for reason in reasons_clad_count), "TC4 failed but not for scratch count."

    # --- Test Case 5: Unlimited Defects in Adhesive Zone ---
    logging.info("\n--- Test Case 5: Unlimited Defects in Adhesive Zone (manual) ---")
    adhesive_defects_list = []
    # Adhesive rules: {"max_defects": "unlimited", "max_defect_size_um": 50}
    for k in range(10): # 10 Pit/Digs
        adhesive_defects_list.append({
            "defect_id": f"test_img_D-Adh{k+1}", "zone": "Adhesive", "classification": "Pit/Dig",
            "length_um": 40.0 # Size is within 50um limit
        })
    status_adh, reasons_adh = apply_pass_fail_rules(adhesive_defects_list, "single_mode_pc")
    logging.info(f"Test Adhesive Unlimited Defects (10 Pits/Digs < 50um) -> Status: {status_adh}, Reasons: {reasons_adh}")
    assert status_adh == "PASS", f"Expected PASS for TC5, got {status_adh}"

    # --- Test Case 6: Defect in Adhesive Zone Exceeding Size ---
    logging.info("\n--- Test Case 6: Defect in Adhesive Zone Exceeding Size (manual) ---")
    adhesive_large_defect = [{
        "defect_id": "test_img_D-AdhLarge", "zone": "Adhesive", "classification": "Pit/Dig",
        "length_um": 60.0 # Size 60um > 50um limit
    }]
    status_adh_large, reasons_adh_large = apply_pass_fail_rules(adhesive_large_defect, "single_mode_pc")
    logging.info(f"Test Adhesive Large Defect (>50um) -> Status: {status_adh_large}, Reasons: {reasons_adh_large}")
    assert status_adh_large == "FAIL", f"Expected FAIL for TC6, got {status_adh_large}"
    assert any("exceeds limit" in reason for reason in reasons_adh_large), "TC6 failed but not for size."

    logging.info("\n--- All Tests in __main__ completed ---")