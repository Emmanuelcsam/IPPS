#!/usr/bin/env python3
# analysis.py

"""
Defect Analysis and Rule Application Module
==========================================================
This module takes the confirmed defect masks from image_processing.py,
characterizes each defect (size, shape, location), classifies them,
and applies pass/fail criteria based on loaded IEC 61300-3-35 rules
from the configuration.
"""

import cv2 # Imports the OpenCV library, essential for all image processing tasks, especially contour analysis and rotated rectangle calculations.
import numpy as np # Imports the NumPy library, used for efficient numerical operations and array manipulations, which is fundamental for image data.
from typing import Dict, Any, Optional, List, Tuple, Union # Imports type hinting classes from the standard library to improve code readability and maintainability.
import logging # Imports the standard library's logging module to record events, warnings, and errors during execution.
from pathlib import Path # Imports the Path object from the standard library for modern, object-oriented manipulation of filesystem paths.


try:
    # Assumes config_loader.py is in the same directory or within the Python path for proper module resolution.
    from config_loader import get_config, get_zone_definitions # Imports functions to access the global configuration and specific zone rule definitions.
# This except block handles the case where the import fails, which is common when running the script as a standalone file for testing.
except ImportError:
    # Logs a warning that the primary config loader could not be found, indicating that the script is likely running in a test mode.
    logging.warning("Could not import from config_loader. Using dummy functions/data for standalone testing.")
    # Defines a dummy (placeholder) function for get_config to provide necessary data structures for testing.
    def get_config() -> Dict[str, Any]: # The function signature with type hints, indicating it returns a dictionary.
        """Returns a dummy configuration for standalone testing."""
        # Returns a simplified dictionary mimicking the structure of the actual configuration file.
        return { 
            # Defines a dictionary for different processing profiles.
            "processing_profiles": {
                # Defines a specific profile named "deep_inspection".
                "deep_inspection": { 
                    # Contains parameters related to defect detection.
                    "defect_detection": {
                        # Defines the aspect ratio threshold to classify a defect as a scratch.
                        "scratch_aspect_ratio_threshold": 3.0,
                        # Defines the minimum area in pixels for a component to be considered a defect, filtering out noise.
                        "min_defect_area_px": 5
                    }
                }
            },
            # Defines a dictionary for zone definitions based on the IEC 61300-3-35 standard.
            "zone_definitions_iec61300_3_35": { 
                # Defines rules for a specific fiber type: single-mode physical contact.
                "single_mode_pc": [
                    # A dictionary defining the "Core" zone and its associated pass/fail rules.
                    {"name": "Core", "pass_fail_rules": {"max_scratches": 0, "max_defects": 0, "max_defect_size_um": 3}},
                    # A dictionary defining the "Cladding" zone and its rules.
                    {"name": "Cladding", "pass_fail_rules": {"max_scratches": 5, "max_defect_size_um": 10}},
                    # A dictionary defining the "Adhesive" zone and its rules, allowing unlimited defects below a certain size.
                    {"name": "Adhesive", "pass_fail_rules": {"max_defects": "unlimited", "max_defect_size_um": 50}},
                    # A dictionary defining the "Contact" zone and its rules.
                    {"name": "Contact", "pass_fail_rules": {"max_defects": "unlimited", "max_defect_size_um": 100}}
                ]
            }
        }
    # Defines a dummy (placeholder) function for get_zone_definitions for testing purposes.
    def get_zone_definitions(fiber_type_key: str = "single_mode_pc") -> List[Dict[str, Any]]: # The function takes a fiber type key and returns a list of dictionaries.
        """Returns dummy zone definitions."""
        # Retrieves and returns the list of zone definitions from the dummy configuration created above.
        return get_config()["zone_definitions_iec61300_3_35"].get(fiber_type_key, [])


# --- Defect Characterization and Classification ---
# This function analyzes the defect mask to find, measure, and classify each individual defect.
def characterize_and_classify_defects(
    final_defect_mask: np.ndarray, # The input binary image where white pixels represent defects.
    zone_masks: Dict[str, np.ndarray], # A dictionary of masks, one for each inspection zone (e.g., "Core", "Cladding").
    profile_config: Dict[str, Any], # The configuration settings for the current processing profile.
    um_per_px: Optional[float], # The conversion factor from pixels to micrometers; can be None.
    image_filename: str, # The filename of the image being analyzed, used for creating unique defect IDs.
    confidence_map: Optional[np.ndarray] = None # An optional map indicating the confidence of each defect pixel (not used in this version but good for future extension).
) -> Tuple[List[Dict[str, Any]], str, int]: # The function returns a tuple containing the list of characterized defects, an overall status string, and the total defect count.
    """
    Returns:
        characterized_defects, overall_status, total_defect_count
    """
    characterized_defects: List[Dict[str, Any]] = [] # Initializes an empty list to store dictionaries, where each dictionary will hold the properties of a single defect.
    if np.sum(final_defect_mask) == 0: # Checks if there are any white pixels (defects) in the mask by summing all pixel values.
        logging.info("No defects found in the final fused mask.") # If the sum is zero, it logs that no defects were detected.
        return characterized_defects, "PASS", 0 # Returns the empty list, a "PASS" status, and a defect count of 0.

    # Finds all separate, connected groups of white pixels (defects) in the binary mask.
    # stats: Provides bounding box and area information for each defect.
    # centroids: Provides the center point (x, y) for each defect.
    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(
        final_defect_mask, connectivity=8, ltype=cv2.CV_32S
    ) # The function returns the number of labels, an image where each defect has a unique integer label, stats, and centroids.

    # Logs the number of potential defects found. It's num_labels - 1 because label 0 is always the background.
    logging.info(f"Found {num_labels - 1} potential defect components from fused mask.")

    # Retrieves the defect detection parameters dictionary from the main profile configuration.
    defect_params = profile_config.get("defect_detection", {}) # Uses .get() to avoid errors if the key doesn't exist.
    # Gets the minimum defect area from the parameters; defaults to 5 pixels if not specified. This filters out noise.
    min_defect_area_px = defect_params.get("min_defect_area_px", 5) 
    # Gets the aspect ratio threshold for classifying scratches; defaults to 3.0.
    scratch_aspect_ratio_threshold = defect_params.get("scratch_aspect_ratio_threshold", 3.0) 

    defect_id_counter = 0 # Initializes a counter to assign a unique ID to each valid defect.

    # Iterates through each component found by connectedComponentsWithStats. The loop starts at 1 to skip the background (label 0).
    for i in range(1, num_labels): 
        area_px = stats[i, cv2.CC_STAT_AREA] # Extracts the area (in pixels) of the current component from the 'stats' array.

        if area_px < min_defect_area_px: # Checks if the component's area is smaller than the configured minimum threshold.
            logging.debug(f"Skipping defect component {i} due to small area: {area_px}px < {min_defect_area_px}px.") # Logs that the component is being skipped as it's likely noise.
            continue # Skips the rest of the loop for this component and moves to the next one.

        defect_id_counter += 1 # Increments the counter for each valid defect.
        defect_id_str = f"{Path(image_filename).stem}_D{defect_id_counter}" # Creates a unique ID string for the defect (e.g., "test_image_D1").

        # Extracts the bounding box properties for the current defect from the 'stats' array.
        x_bbox = stats[i, cv2.CC_STAT_LEFT] # The leftmost x-coordinate of the bounding box.
        y_bbox = stats[i, cv2.CC_STAT_TOP] # The topmost y-coordinate of the bounding box.
        w_bbox = stats[i, cv2.CC_STAT_WIDTH] # The width of the bounding box.
        h_bbox = stats[i, cv2.CC_STAT_HEIGHT] # The height of the bounding box.
        centroid_x_px, centroid_y_px = centroids[i] # Extracts the (x, y) coordinates of the defect's centroid.

        # Creates a binary mask for only the current defect component to isolate it for further analysis.
        component_mask = (labels_img == i).astype(np.uint8) * 255 # Sets pixels of the current defect to 255 (white) and all others to 0 (black).
        # Finds the external contour (outline) of the isolated defect.
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        
        if not contours: # A safety check; this should rarely happen if the area is greater than 0.
            logging.warning(f"No contour found for defect component {i} with area {area_px}px. Skipping.") # Logs a warning if no contour was found.
            continue # Skips to the next component.
        
        defect_contour = contours[0] # Assumes the first and largest contour represents the defect.

        # --- Precise Dimension Calculation using minAreaRect ---
        # This function calculates the tightest-fitting rotated rectangle around the contour, which gives more accurate dimensions for non-axis-aligned defects.
        # It returns ((center_x, center_y), (width, height), angle_of_rotation).
        rotated_rect = cv2.minAreaRect(defect_contour)
        # These lines, if uncommented, would calculate the four corner points of the rotated rectangle, useful for drawing it.
        # box_points = cv2.boxPoints(rotated_rect) 
        # box_points = np.intp(box_points)

        # Extracts the width and height from the rotated rectangle object.
        width_px = rotated_rect[1][0]
        height_px = rotated_rect[1][1]
        # Calculates the aspect ratio. An epsilon (1e-6) is added to prevent division by zero for very thin defects.
        aspect_ratio = max(width_px, height_px) / (min(width_px, height_px) + 1e-6) 


        perimeter = cv2.arcLength(defect_contour, True) # Calculates the perimeter of the defect's contour.
        # Calculates circularity, a measure of how close the shape is to a perfect circle (1.0).
        circularity = 4 * np.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0
        
        # Calculates the solidity of the defect, which is the ratio of its actual area to the area of its convex hull (its "wrapped" shape).
        hull = cv2.convexHull(defect_contour) # Computes the convex hull of the contour.
        hull_area = cv2.contourArea(hull) # Calculates the area of the convex hull.
        solidity = area_px / hull_area if hull_area > 0 else 0 # Computes solidity; low values indicate irregular or concave shapes.
        
        # Calculates the extent, which is the ratio of the defect's area to the area of its axis-aligned bounding box.
        rect_area = w_bbox * h_bbox # Area of the simple, non-rotated bounding box.
        extent = area_px / rect_area if rect_area > 0 else 0 # Computes extent; low values suggest an irregular shape that doesn't fill its bounding box.
        
        # This re-calculates the rotated rectangle, which is redundant from the earlier calculation but shown here for clarity in this block.
        rotated_rect = cv2.minAreaRect(defect_contour)
        (cx_rr, cy_rr), (width_rr, height_rr), angle = rotated_rect # Unpacks the properties of the rotated rectangle.
        
        # This block ensures that 'width_rr' always refers to the longer dimension of the rotated rectangle, simplifying aspect ratio calculation.
        if height_rr > width_rr:
            width_rr, height_rr = height_rr, width_rr # Swaps the values if height is greater than width.
        
        # Re-calculates aspect ratio using the consistently ordered dimensions.
        aspect_ratio = width_rr / (height_rr + 1e-6)
        
        # This block uses a set of heuristic rules based on shape metrics to classify the defect.
        # Scratches are expected to be long and thin (high aspect ratio) and not very circular or solid.
        if aspect_ratio >= scratch_aspect_ratio_threshold and circularity < 0.4 and solidity < 0.7 and extent < 0.5:
            classification = "Scratch" # Classifies as a scratch if it meets these criteria.
        # Pits/Digs are expected to be roughly circular and solid.
        elif aspect_ratio < 2.0 and circularity > 0.6 and solidity > 0.8 and extent > 0.7:
            classification = "Pit/Dig" # Classifies as a pit/dig if it meets these criteria.
        # This 'else' block handles ambiguous cases that don't clearly fit either category.
        else:
            # It calculates a weighted score for both "Scratch" and "Pit/Dig" based on the shape metrics.
            scratch_score = (aspect_ratio / 10.0) + (1 - circularity) + (1 - solidity) + (1 - extent)
            pit_score = (1 / (aspect_ratio + 0.1)) + circularity + solidity + extent
            
            # The defect is classified based on which score is higher.
            if scratch_score > pit_score:
                classification = "Scratch"
            else:
                classification = "Pit/Dig"

        # Initializes size variables in microns to None.
        length_um = None
        width_um = None
        # If the pixel-to-micron conversion factor is available, it calculates the defect's dimensions in microns.
        if um_per_px:
            # The length in microns is the longer dimension of the rotated rectangle multiplied by the conversion factor.
            length_um = max(width_px, height_px) * um_per_px
            # The width in microns is the shorter dimension.
            width_um = min(width_px, height_px) * um_per_px

        # Compiles all the gathered information about the defect into a single dictionary.
        defect_dict = {
            "defect_id": defect_id_str, # The unique ID for this defect.
            "contour_points_px": defect_contour.reshape(-1, 2).tolist(), # The list of (x,y) points defining the defect's outline.
            "bbox_x_px": x_bbox, # The x-coordinate of the axis-aligned bounding box.
            "bbox_y_px": y_bbox, # The y-coordinate of the axis-aligned bounding box.
            "bbox_w_px": w_bbox, # The width of the axis-aligned bounding box.
            "bbox_h_px": h_bbox, # The height of the axis-aligned bounding box.
            "centroid_x_px": float(centroid_x_px), # The x-coordinate of the defect's center.
            "centroid_y_px": float(centroid_y_px), # The y-coordinate of the defect's center.
            "area_px": int(area_px), # The area of the defect in square pixels.
            "width_px": float(width_px), # The more accurate width from the rotated rectangle.
            "height_px": float(height_px), # The more accurate height from the rotated rectangle.
            "aspect_ratio": float(aspect_ratio), # The calculated aspect ratio.
            "classification": classification, # The final classification ("Scratch" or "Pit/Dig").
            "length_um": length_um, # The maximum dimension in microns (if available).
            "width_um": width_um,   # The minimum dimension in microns (if available).
            "zone": "Unknown",  # Initializes the zone to "Unknown"; this will be updated next.
        }

        # This loop determines which inspection zone the defect belongs to.
        for zone_name, zone_mask in zone_masks.items(): # Iterates through the provided zone masks (e.g., "Core", "Cladding").
            # Converts the floating-point centroid coordinates to integers to use as array indices.
            y_coord = int(centroid_y_px)
            x_coord = int(centroid_x_px)
            # A safety check to ensure the centroid coordinates are within the boundaries of the mask image.
            if 0 <= y_coord < zone_mask.shape[0] and 0 <= x_coord < zone_mask.shape[1]:
                # Checks the pixel value of the zone mask at the defect's centroid location.
                if zone_mask[y_coord, x_coord] > 0: # If the pixel is non-zero (white), the defect is in this zone.
                    defect_dict["zone"] = zone_name # Updates the defect's dictionary with the correct zone name.
                    break # Exits the loop since the zone has been found.
            # This 'else' block handles the rare case where a defect's centroid is outside the image dimensions.
            else:
                logging.warning(
                    f"Defect {defect_id_str} centroid ({x_coord}, {y_coord}) is outside zone mask dimensions {zone_mask.shape}."
                )

        # Adds the fully characterized defect dictionary to the list of all defects.
        characterized_defects.append(defect_dict)

    # After iterating through all components, this is the final count of valid defects.
    total_defect_count = len(characterized_defects)
    # Initializes the overall status to "PASS". This can be changed by a preliminary check or, more definitively, by the apply_pass_fail_rules function.
    overall_status = "PASS" 
    # This section contains a placeholder for a simple, preliminary pass/fail check before the main rule application.
    # The primary function for this is apply_pass_fail_rules, which uses much more detailed logic.
    # This loop provides a quick check, for example, immediately failing if any defect is found in the most critical zone.
    for d in characterized_defects:
        # Example rule: a simple preliminary check that fails the inspection if any defect is located in the "Core" zone.
        # The more robust size-based checks are handled later by apply_pass_fail_rules using zone-specific rules from the config.
        if d["zone"] == "Core":
            overall_status = "FAIL" # Sets the preliminary status to "FAIL".
            # The commented-out code below shows a more complex placeholder that was replaced by the simpler "any defect in core" rule.
            # if d["zone"] == "Core" or (
            #    d.get("length_um", 0) and um_per_px and 
            #    d["length_um"] > profile_config.get("defect_detection", {}).get("max_defect_size_um", float('inf'))
            # ):
            # Using float('inf') as a default ensures that a missing config key doesn't cause an unexpected failure.
            # However, this overall_status is typically overridden by the more comprehensive apply_pass_fail_rules function.
            break # Exits the loop as soon as the first preliminary failure condition is met.

    # Returns the final list of defects, the preliminary overall status, and the total defect count.
    return characterized_defects, overall_status, total_defect_count


# This function calculates the density of defects within a specific zone.
def calculate_defect_density(defects: List[Dict[str, Any]], zone_area_px: float) -> float:
    """
    Calculates defect density (defects per unit area).
    """
    # Sums the 'area_px' of all defects in the provided list.
    total_defect_area = sum(d.get('area_px', 0) for d in defects)
    # Returns the total defect area divided by the zone area, avoiding division by zero.
    return total_defect_area / zone_area_px if zone_area_px > 0 else 0

# This function analyzes and summarizes defect statistics for each individual zone.
def analyze_defects_by_zone(characterized_defects: List[Dict[str, Any]], 
                           zone_masks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """
    Perform detailed region-specific analysis of defects.
    
    Args:
        characterized_defects: List of characterized defect dictionaries.
        zone_masks: Dictionary of zone masks.
        
    Returns:
        Dictionary with zone-specific statistics.
    """
    zone_stats = {} # Initializes an empty dictionary to hold the statistics for each zone.
    
    # Iterates through each zone name and its corresponding mask.
    for zone_name, zone_mask in zone_masks.items():
        # Creates a list of defects that are located within the current zone.
        zone_defects = [d for d in characterized_defects if d.get('zone') == zone_name]
        
        # Calculates the total area of the current zone in square pixels for density calculations.
        zone_area_px = np.sum(zone_mask > 0)
        
        # Separates the defects within the zone by their classification type.
        scratches = [d for d in zone_defects if d['classification'] == 'Scratch']
        pits_digs = [d for d in zone_defects if d['classification'] == 'Pit/Dig']
        
        # Calculates various statistics for the current zone.
        total_defect_area = sum(d.get('area_px', 0) for d in zone_defects) # Sums the area of all defects in the zone.
        # Calculates the defect density for the zone.
        defect_density = total_defect_area / zone_area_px if zone_area_px > 0 else 0
        
        # Gathers a list of defect sizes (preferring microns, but falling back to pixels).
        defect_sizes = [d.get('length_um', d.get('length_px', 0)) for d in zone_defects]
        
        # Compiles all calculated statistics for the zone into a dictionary.
        zone_stats[zone_name] = {
            'total_defects': len(zone_defects), # The total number of defects in the zone.
            'scratch_count': len(scratches), # The number of scratches in the zone.
            'pit_dig_count': len(pits_digs), # The number of pits/digs in the zone.
            'total_area_px': total_defect_area, # The total area covered by defects in pixels.
            'defect_density': defect_density, # The calculated defect density.
            'zone_area_px': zone_area_px, # The total area of the zone in pixels.
            'max_defect_size': max(defect_sizes) if defect_sizes else 0, # The size of the largest defect.
            'avg_defect_size': np.mean(defect_sizes) if defect_sizes else 0, # The average size of defects.
            'defects': zone_defects # The list of raw defect dictionaries for this zone.
        }
        
        # Logs a summary of the findings for the current zone.
        logging.info(f"Zone '{zone_name}': {len(zone_defects)} defects "
                     f"({len(scratches)} scratches, {len(pits_digs)} pits/digs), "
                     f"density: {defect_density:.4f}")
    
    # Returns the dictionary containing the comprehensive statistics for all zones.
    return zone_stats

# --- Pass/Fail Evaluation ---
# This function applies the official pass/fail rules (e.g., from IEC 61300-3-35) to the list of characterized defects.
def apply_pass_fail_rules(
    characterized_defects: List[Dict[str, Any]], # The list of all characterized defects from the image.
    fiber_type_key: str # A key (e.g., "single_mode_pc") to fetch the correct set of rules from the configuration.
) -> Tuple[str, List[str]]: # Returns a tuple containing the final overall status ("PASS" or "FAIL") and a list of reasons for failure.
    """
    Applies pass/fail criteria based on IEC 61300-3-35 rules loaded from config.

    Args:
        characterized_defects: List of defect dictionaries from characterization.
        fiber_type_key: The key for the fiber type to retrieve specific zone rules.

    Returns:
        A tuple: (overall_status: str, failure_reasons: List[str])
                 Overall status is "PASS" or "FAIL".
    """
    overall_status = "PASS" # Initializes the overall status to "PASS"; it will be changed to "FAIL" if any rule is violated.
    failure_reasons: List[str] = [] # Initializes an empty list to collect descriptions of each rule violation.

    # This 'try' block safely retrieves the configuration, handling cases where it might be missing.
    try:
        # Fetches the specific list of zone definitions and rules for the given fiber type from the configuration.
        zone_rule_definitions = get_zone_definitions(fiber_type_key) 
        if not zone_rule_definitions: # Checks if the returned list is empty.
             logging.warning(f"No zone definitions found for fiber type '{fiber_type_key}'. Cannot apply rules.") # Logs a warning if no rules are found.
             return "ERROR_CONFIG", [f"No zone definitions for fiber type '{fiber_type_key}'."] # Returns an error status indicating a configuration problem.

    except ValueError as e: # Catches potential errors during configuration loading.
        logging.error(f"Cannot apply pass/fail rules: {e}") # Logs the specific error.
        return "ERROR_CONFIG", [f"Configuration error for fiber type '{fiber_type_key}': {e}"] # Returns an error status.

    # This dictionary comprehension creates a structure to group defects by their zone for easier rule checking.
    defects_by_zone: Dict[str, List[Dict[str, Any]]] = { # Example: {"Core": [], "Cladding": [], ...}
        zone_def["name"]: [] for zone_def in zone_rule_definitions
    }
    for defect in characterized_defects: # Iterates through each characterized defect.
        zone_name = defect.get("zone", "Unknown") # Gets the zone name for the defect.
        if zone_name in defects_by_zone: # Checks if the defect's zone is one of the zones with defined rules.
            defects_by_zone[zone_name].append(defect) # Adds the defect to the appropriate list in the dictionary.
        elif zone_name != "Unknown": # Handles cases where a defect is in a zone that doesn't have rules defined for this fiber type.
            # This can happen if zone_masks are generated for more areas than are specified in the pass/fail rules.
            logging.warning(f"Defect {defect['defect_id']} in zone '{zone_name}' which has no defined rules for fiber type '{fiber_type_key}'. This defect will not be evaluated against specific zone rules.")
            # Defects in "Unknown" zones are implicitly ignored by the rule application loop below.


    # This is the main loop for applying rules to each zone.
    for zone_def_rules in zone_rule_definitions: # Iterates through the definitions for each zone (e.g., Core, Cladding).
        zone_name = zone_def_rules["name"] # Gets the name of the current zone.
        rules = zone_def_rules.get("pass_fail_rules", {}) # Gets the dictionary of pass/fail rules for this zone.
        current_zone_defects = defects_by_zone.get(zone_name, []) # Gets the list of defects found in this zone.

        # Skips the zone if there are no defects in it, which is an efficient optimization.
        if not current_zone_defects:
            logging.debug(f"No defects found in zone '{zone_name}'. Skipping rule checks for this zone.")
            continue
        
        # Skips if no rules are defined for the zone, preventing errors and unnecessary processing.
        if not rules:
            logging.debug(f"No specific pass/fail rules defined for zone '{zone_name}' in config. Defects in this zone will not cause failure.")
            continue


        # Separates defects within the current zone by their classification to check against type-specific rules.
        scratches_in_zone = [d for d in current_zone_defects if d["classification"] == "Scratch"] # Creates a list of all scratches.
        pits_digs_in_zone = [d for d in current_zone_defects if d["classification"] == "Pit/Dig"] # Creates a list of all pits/digs.

        # --- Rule Check 1: Scratch Count ---
        max_scratches_allowed = rules.get("max_scratches") # Gets the maximum number of allowed scratches from the rules.
        # Checks if the rule is defined as an integer and if the actual count of scratches exceeds the allowed number.
        if isinstance(max_scratches_allowed, int) and len(scratches_in_zone) > max_scratches_allowed: 
            overall_status = "FAIL" # Sets the final status to FAIL.
            failure_reasons.append(f"Zone '{zone_name}': Too many scratches ({len(scratches_in_zone)} > {max_scratches_allowed}).") # Adds a descriptive reason for the failure.
            logging.warning(f"FAIL Rule (Scratch Count): Zone '{zone_name}', Count={len(scratches_in_zone)}, Allowed={max_scratches_allowed}")


        # --- Rule Check 2: Pit/Dig Count ---
        # The config key "max_defects" is often used to refer to non-scratch defects like pits and digs.
        max_pits_digs_allowed = rules.get("max_defects") # Gets the maximum number of allowed pits/digs from the rules.
        # Checks if the rule is an integer and if the actual count exceeds the limit.
        if isinstance(max_pits_digs_allowed, int) and len(pits_digs_in_zone) > max_pits_digs_allowed: 
            overall_status = "FAIL" # Sets the final status to FAIL.
            failure_reasons.append(f"Zone '{zone_name}': Too many Pits/Digs ({len(pits_digs_in_zone)} > {max_pits_digs_allowed}).") # Adds a descriptive failure reason.
            logging.warning(f"FAIL Rule (Pit/Dig Count): Zone '{zone_name}', Count={len(pits_digs_in_zone)}, Allowed={max_pits_digs_allowed}")

        # --- Rule Check 3: Defect Size ---
        max_defect_size_um_allowed = rules.get("max_defect_size_um") # Gets the general maximum allowed defect size for this zone.
        max_scratch_length_um_allowed = rules.get("max_scratch_length_um") # Gets a potentially more specific rule for maximum scratch length.

        for defect in current_zone_defects: # Iterates through each individual defect within the current zone.
            primary_dimension_um: Optional[float] = None # Initializes the variable that will hold the key dimension for size checking.
            defect_type_for_rule = defect["classification"] # Gets the defect's classification.
            
            if defect_type_for_rule == "Scratch": # Checks if the defect is a scratch.
                primary_dimension_um = defect.get("length_um") # For scratches, the primary dimension is its length.
                # This line intelligently selects the rule to use: the specific scratch rule if it exists, otherwise the general defect size rule.
                current_max_size_rule = max_scratch_length_um_allowed if max_scratch_length_um_allowed is not None else max_defect_size_um_allowed
            else: # This handles defects classified as "Pit/Dig".
                 # For pits/digs, the 'length_um' (the longest dimension of the rotated box) is typically used as the primary size metric.
                 # An alternative could be an 'effective_diameter_um' if that were calculated during characterization.
                primary_dimension_um = defect.get("length_um")
                current_max_size_rule = max_defect_size_um_allowed # For pits/digs, the general defect size rule is used.

            # Checks if the defect has a valid size in microns and if there is a numerical size rule to check against.
            if primary_dimension_um is not None and isinstance(current_max_size_rule, (int, float)): 
                if primary_dimension_um > current_max_size_rule: # Compares the defect's size to the allowed maximum.
                    overall_status = "FAIL" # Sets the final status to FAIL if the defect is too large.
                    reason = f"Zone '{zone_name}': {defect_type_for_rule} '{defect['defect_id']}' size ({primary_dimension_um:.2f}µm) exceeds limit ({current_max_size_rule}µm)." # Creates a detailed failure reason string.
                    failure_reasons.append(reason) # Adds the reason to the list.
                    logging.warning(f"FAIL Rule (Defect Size): {reason}") # Logs the specific failure.
            
            # This section provides an example for a more complex, specific rule that might exist in a configuration.
            # For example, a rule that states "zero scratches are allowed that are greater than 5µm".
            specific_scratch_size_limit_key = "max_scratches_gt_5um" # Defines the key for this example rule.
            specific_scratch_size_limit_value = 5.0 # Defines the size threshold for this example rule.
            
            # This checks if the defect is a scratch AND if the specific rule (e.g., "max_scratches_gt_5um": 0) exists in the config.
            if defect_type_for_rule == "Scratch" and rules.get(specific_scratch_size_limit_key) == 0:
                # If a scratch is found that is larger than the size specified in this special rule, it's a failure.
                if primary_dimension_um is not None and primary_dimension_um > specific_scratch_size_limit_value:
                    overall_status = "FAIL" # Sets the status to FAIL.
                    # Creates a highly specific failure reason message.
                    reason = (
                        f"Zone '{zone_name}': Scratch '{defect['defect_id']}' size ({primary_dimension_um:.2f}µm) "
                        f"found, but no scratches > {specific_scratch_size_limit_value}µm allowed by rule '{specific_scratch_size_limit_key}'."
                    )
                    failure_reasons.append(reason) # Adds the reason to the list.
                    logging.warning(f"FAIL Rule (Specific Scratch Size Count): {reason}") # Logs the specific failure.


    if not failure_reasons and overall_status == "PASS": # After checking all zones and rules, if no failures were recorded, the inspection passed.
        logging.info(f"Pass/Fail Evaluation for '{fiber_type_key}': Overall PASS.")
    elif overall_status == "FAIL": # If the status was set to FAIL at any point.
        # Logs a summary of the failure, using list(set(...)) to ensure each unique reason is only listed once.
        logging.warning(f"Pass/Fail Evaluation for '{fiber_type_key}': Overall FAIL. Reasons: {'; '.join(list(set(failure_reasons)))}")
    # This handles other states, such as "ERROR_CONFIG", where failure reasons might have been generated without the status being "FAIL".
    elif failure_reasons:
         logging.error(f"Pass/Fail Evaluation for '{fiber_type_key}': Status {overall_status}. Issues: {'; '.join(list(set(failure_reasons)))}")


    return overall_status, list(set(failure_reasons)) # Returns the final status and the list of unique failure reasons.

# This special block of code only runs when the script is executed directly from the command line (e.g., "python analysis.py").
# It does not run when the script is imported as a module into another file.
if __name__ == "__main__":
    # This block is designed for testing the functions within this module in isolation.
    # Configures basic logging to show debug-level messages and detailed information about where the log message originated.
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s') 

    # --- Dummy Data for Testing ---
    dummy_image_filename = "test_img.png" # Creates a fake filename for testing purposes.
    dummy_um_per_px = 0.5  # Defines an example conversion factor: 0.5 microns per pixel.

    # Creates a blank (all black) NumPy array to serve as a dummy image mask for defects.
    dummy_mask = np.zeros((200, 200), dtype=np.uint8) 
    # Draws a long, thin rectangle on the mask to simulate a scratch-like defect.
    # The coordinates define a rectangle from (20,30) to (25,100).
    cv2.rectangle(dummy_mask, (20, 30), (25, 100), (255,), -1) # The (255,) makes it white, and -1 fills it.
    # Draws a circle on the mask to simulate a pit or dig-like defect.
    # The circle is centered at (100,100) with a radius of 10 pixels.
    cv2.circle(dummy_mask, (100, 100), 10, (255,), -1) # The (255,) makes it white, and -1 fills it.
    # Draws a very small 2x2 pixel rectangle. This defect should be filtered out by the 'min_defect_area_px' setting.
    # The coordinates define a rectangle from (150,150) to (152,152).
    cv2.rectangle(dummy_mask, (150,150), (152,152), (255,), -1) # The (255,) makes it white, and -1 fills it.


    # Defines a dictionary of dummy zone masks, which are blank images initially.
    dummy_zone_masks = { 
        "Core": np.zeros((200, 200), dtype=np.uint8), # Creates a mask for the "Core" zone.
        "Cladding": np.zeros((200, 200), dtype=np.uint8), # Creates a mask for the "Cladding" zone.
        "Adhesive": np.zeros((200,200), dtype=np.uint8) # Creates a mask for the "Adhesive" zone for comprehensive testing.
    }
    
    
    # Draws a filled circle on the "Core" mask to define the core area. Defect 2 (at 100,100) should fall inside this.
    cv2.circle(dummy_zone_masks["Core"], (100, 100), 40, (255,), -1)
    # Creates the "Cladding" zone as an annulus (a ring).
    # First, a large circle is drawn for the outer boundary. Defect 1's centroid (approx 22.5, 65) should be in this area.
    cv2.circle(dummy_zone_masks["Cladding"], (100, 100), 80, (255,), -1) 
    # Then, a smaller black circle is drawn in the middle to "cut out" the core area, leaving only the ring.
    cv2.circle(dummy_zone_masks["Cladding"], (100, 100), 40, (0,), -1)   
    # Draws a rectangle at the top of the image to define the "Adhesive" zone area.
    cv2.rectangle(dummy_zone_masks["Adhesive"], (0,0), (200,20), (255,), -1)


    # Retrieves the dummy configuration for the "deep_inspection" profile to be used in the test.
    dummy_profile_cfg = get_config()["processing_profiles"]["deep_inspection"] 
    # Overwrites the min_defect_area_px in the dummy config to ensure the small 2x2 defect is filtered out in the test.
    dummy_profile_cfg["defect_detection"]["min_defect_area_px"] = 5


    # --- Test Case 1: Characterize and Classify Defects ---
    logging.info("\n--- Test Case 1: Characterize and Classify Defects ---")
    # Calls the function under test with all the dummy data created above.
    # It unpacks all three return values into separate variables.
    characterized_defects_list, initial_overall_status, total_defect_count_val = characterize_and_classify_defects(
        dummy_mask, dummy_zone_masks, dummy_profile_cfg, dummy_um_per_px, dummy_image_filename
    )
    # Logs the initial status and the final count of defects after filtering.
    logging.info(f"Initial characterization status: {initial_overall_status}, Total defects found (after filtering): {total_defect_count_val}")

    if characterized_defects_list: # Checks if the function returned any characterized defects.
        logging.info(f"Characterized {len(characterized_defects_list)} defects:") # Logs the number of defects found.
        for defect_item in characterized_defects_list: # Loops through each defect dictionary in the returned list.
            # Logs a summary of each defect's properties.
            logging.info(f"  ID: {defect_item['defect_id']}, Class: {defect_item['classification']}, Zone: {defect_item['zone']}, AreaPx: {defect_item['area_px']}, LengthUm: {defect_item.get('length_um', 'N/A')}")
    else: # This block runs if the list of defects is empty.
        logging.info("No defects characterized by the function (either none found or all filtered).")

    # --- Test Case 2: Apply Pass/Fail Rules ---
    logging.info("\n--- Test Case 2: Apply Pass/Fail Rules (using defects from Test Case 1) ---")
    # Uses the list of defects generated in Test Case 1 to test the rule application function.
    status_tc2, reasons_tc2 = apply_pass_fail_rules(characterized_defects_list, "single_mode_pc")
    # Logs the final pass/fail status returned by the function.
    logging.info(f"Pass/Fail Status (from apply_pass_fail_rules TC2): {status_tc2}")
    if reasons_tc2: # Checks if the function returned any failure reasons.
        logging.info("Failure Reasons (TC2):") # Logs a header for the reasons.
        for reason_item in reasons_tc2: # Loops through and logs each failure reason.
            logging.info(f"  - {reason_item}")
    elif status_tc2 == "PASS": # If the status is PASS, it logs a success message.
        logging.info("No failure reasons (TC2): Overall PASS.")

    # These comments explain the expected outcome of Test Case 2 based on the dummy data and rules.
    # Expected: Defect 1 (Scratch in Cladding, length 70px * 0.5um/px = 35um) -> Cladding rule allows max_defect_size_um of 10. This should FAIL.
    # Expected: Defect 2 (Pit/Dig in Core, diameter approx 20px * 0.5um/px = 10um) -> Core rule allows max_defect_size_um of 3. This should FAIL.
    # Defect 3 (area 4px) is smaller than the min_defect_area_px (5px), so it should be filtered out and not evaluated.
    # Therefore, the final status for Test Case 2 is expected to be "FAIL".

    # --- Test Case 3: Failing Core Defect (manual example) ---
    logging.info("\n--- Test Case 3: Failing Core Defect (manual) ---")
    # Manually creates a defect dictionary representing a defect in the Core zone that is too large.
    failing_core_defect = [{ 
        "defect_id": "test_img_D-CoreFail", "zone": "Core", "classification": "Pit/Dig",
        "confidence_score": 1.0, "centroid_x_px": 100, "centroid_y_px": 100,
        "area_px": 314, "length_px": 20, "width_px": 20, "aspect_ratio": 1.0,
        "area_um2": 78.5, "length_um": 10.0, "width_um": 10.0 # Its length (10.0µm) exceeds the Core's rule limit (3µm).
        # This comment shows how an alternative key might have been used.
        # "effective_diameter_um": 10.0 
    }]
    # Applies the pass/fail rules to this single-defect list.
    status_fail_core, reasons_fail_core = apply_pass_fail_rules(failing_core_defect, "single_mode_pc") 
    # Logs the result of this specific test.
    logging.info(f"Test Failing Core Defect -> Status: {status_fail_core}, Reasons: {reasons_fail_core}")
    # Asserts that the status is "FAIL", which will cause the script to error out if the test doesn't produce the expected result.
    assert status_fail_core == "FAIL", f"Expected FAIL for TC3, got {status_fail_core}"

    # --- Test Case 4: Cladding Scratch Count (manual example) ---
    logging.info("\n--- Test Case 4: Cladding Scratch Count (manual) ---")
    passing_cladding_defects_list = [] # Initializes an empty list to hold the defects for this test.
    # The rule for the Cladding zone allows a maximum of 5 scratches.
    for k in range(6): # This loop creates 6 scratches, which is one more than the allowed limit.
        passing_cladding_defects_list.append({
            "defect_id": f"test_img_D-CladScratch{k+1}", "zone": "Cladding", "classification": "Scratch",
            "confidence_score": 1.0, "centroid_x_px": 25, "centroid_y_px": 50+k*5,
            "area_px": 20, "length_px": 20, "width_px": 1, "aspect_ratio": 20.0,
            "area_um2": 5, "length_um": 10.0, "width_um": 0.5 # The size of each scratch (10.0µm) is within the Cladding's size limit (10µm), so only the count should cause failure.
        })
    # Applies the rules to the list of 6 scratches.
    status_clad_count, reasons_clad_count = apply_pass_fail_rules(passing_cladding_defects_list, "single_mode_pc") 
    # Logs the result of the test.
    logging.info(f"Test Cladding Scratch Count (6 scratches) -> Status: {status_clad_count}, Reasons: {reasons_clad_count}")
    # Asserts that the test result is "FAIL".
    assert status_clad_count == "FAIL", f"Expected FAIL for TC4 due to scratch count, got {status_clad_count}"
    # Asserts that the failure reason contains the expected text, confirming it failed for the correct reason.
    assert any("Too many scratches" in reason for reason in reasons_clad_count), "TC4 failed but not for scratch count."

    # --- Test Case 5: Unlimited Defects in Adhesive Zone ---
    logging.info("\n--- Test Case 5: Unlimited Defects in Adhesive Zone (manual) ---")
    adhesive_defects_list = [] # Initializes an empty list.
    # The rules for the Adhesive zone are {"max_defects": "unlimited", "max_defect_size_um": 50}.
    for k in range(10): # This loop creates 10 Pit/Dig defects. The count doesn't matter because it's "unlimited".
        adhesive_defects_list.append({
            "defect_id": f"test_img_D-Adh{k+1}", "zone": "Adhesive", "classification": "Pit/Dig",
            "length_um": 40.0 # The size of each defect (40.0µm) is within the 50µm limit.
        })
    # Applies the rules to this list of defects.
    status_adh, reasons_adh = apply_pass_fail_rules(adhesive_defects_list, "single_mode_pc")
    # Logs the result of the test.
    logging.info(f"Test Adhesive Unlimited Defects (10 Pits/Digs < 50um) -> Status: {status_adh}, Reasons: {reasons_adh}")
    # Asserts that the status is "PASS", as no rules should have been violated.
    assert status_adh == "PASS", f"Expected PASS for TC5, got {status_adh}"

    # --- Test Case 6: Defect in Adhesive Zone Exceeding Size ---
    logging.info("\n--- Test Case 6: Defect in Adhesive Zone Exceeding Size (manual) ---")
    # Creates a single large defect in the Adhesive zone.
    adhesive_large_defect = [{
        "defect_id": "test_img_D-AdhLarge", "zone": "Adhesive", "classification": "Pit/Dig",
        "length_um": 60.0 # Its size (60.0µm) is larger than the 50µm limit for the Adhesive zone.
    }]
    # Applies the rules to this single defect.
    status_adh_large, reasons_adh_large = apply_pass_fail_rules(adhesive_large_defect, "single_mode_pc")
    # Logs the result of the test.
    logging.info(f"Test Adhesive Large Defect (>50um) -> Status: {status_adh_large}, Reasons: {reasons_adh_large}")
    # Asserts that the test result is "FAIL".
    assert status_adh_large == "FAIL", f"Expected FAIL for TC6, got {status_adh_large}"
    # Asserts that the failure reason indicates a size violation, confirming the test failed for the correct reason.
    assert any("exceeds limit" in reason for reason in reasons_adh_large), "TC6 failed but not for size."

    # Logs a final message indicating that all standalone tests in the script have completed successfully.
    logging.info("\n--- All Tests in __main__ completed ---")