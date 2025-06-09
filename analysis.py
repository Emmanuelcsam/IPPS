
#!/usr/bin/env python3
# analysis.py

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from pathlib import Path

# --- C++ Accelerator Integration ---
try:
    import accelerator
    CPP_ACCELERATOR_AVAILABLE = True
    logging.info("Successfully imported 'accelerator' C++ module. Analysis will be accelerated.")
except ImportError:
    CPP_ACCELERATOR_AVAILABLE = False
    logging.warning("C++ accelerator module ('accelerator') not found. "
                    "Falling back to pure Python analysis implementations.")
    
try:
    from ml_classifier import DefectClassifier
    ML_CLASSIFIER_AVAILABLE = True
except ImportError:
    ML_CLASSIFIER_AVAILABLE = False
    logging.warning("ML classifier not available, using rule-based classification")
    

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
            characterized_defects = accelerator.characterize_and_classify_defects(
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

        # Use ML classifier if available
        if ML_CLASSIFIER_AVAILABLE:
            ml_classifier = profile_config.get("ml_classifier_instance")
            if ml_classifier and ml_classifier.fitted:
                # Need access to original image for intensity features
                original_image = profile_config.get("_temp_original_image")
                if original_image is not None:
                    classification, confidence = ml_classifier.predict(
                        defect_dict, original_image
                    )
                    defect_dict["ml_confidence"] = confidence
                else:
                    # Fallback to rule-based
                    classification = "Scratch" if aspect_ratio >= scratch_aspect_ratio_threshold else "Pit/Dig"
            else:
                # Fallback to rule-based
                classification = "Scratch" if aspect_ratio >= scratch_aspect_ratio_threshold else "Pit/Dig"
        else:
            # Rule-based classification
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