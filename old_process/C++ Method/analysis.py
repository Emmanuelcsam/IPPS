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
    # Dummy functions for standalone testing
    logging.warning("Could not import from config_loader. Using dummy functions/data.")
    def get_config() -> Dict[str, Any]:
        return { 
            "processing_profiles": { 
                "deep_inspection": { 
                    "defect_detection": { 
                        "scratch_aspect_ratio_threshold": 3.0, 
                        "min_defect_area_px": 5 
                    } 
                } 
            } 
        }
    def get_zone_definitions(fiber_type_key: str = "single_mode_pc") -> List[Dict[str, Any]]:
        return [
            {
                "name": "Core",
                "type": "core",
                "pass_fail_rules": {
                    "max_scratches": 0,
                    "max_defects": 0,
                    "max_defect_size_um": 3
                }
            },
            {
                "name": "Cladding",
                "type": "cladding",
                "pass_fail_rules": {
                    "max_scratches": 5,
                    "max_scratches_gt_5um": 0,
                    "max_defects": 5,
                    "max_defect_size_um": 10
                }
            }
        ]


# --- Defect Characterization and Classification ---
def characterize_and_classify_defects(
    all_detected_defects: List[Dict[str, Any]],
    processed_image: np.ndarray,
    localization_data: Dict[str, Any],
    um_per_px: Optional[float],
    global_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Characterizes and classifies all detected defects.
    
    Args:
        all_detected_defects: List of detected defect dictionaries from image_processing
        processed_image: The processed grayscale image
        localization_data: Fiber localization data (centers, radii)
        um_per_px: Microns per pixel conversion factor
        global_config: Global configuration dictionary
        
    Returns:
        List of characterized defect dictionaries
    """
    if not all_detected_defects:
        logging.info("No defects to characterize.")
        return []
    
    # Get configuration parameters
    profile_config = global_config.get("processing_profiles", {}).get("deep_inspection", {})
    defect_params = profile_config.get("defect_detection", {})
    min_defect_area_px = defect_params.get("min_defect_area_px", 5)
    scratch_aspect_ratio_threshold = defect_params.get("scratch_aspect_ratio_threshold", 3.0)
    
    characterized_defects = []
    defect_id_counter = 0
    
    # Process each detected defect
    for defect_info in all_detected_defects:
        defect_mask = defect_info.get('defect_mask')
        zone_name = defect_info.get('zone', 'Unknown')
        
        if defect_mask is None:
            continue
            
        # Find contours of the defect
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        # Process each contour as a separate defect
        for contour in contours:
            # Calculate area
            area_px = cv2.contourArea(contour)
            if area_px < min_defect_area_px:
                continue
                
            defect_id_counter += 1
            
            # Calculate moments for centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get rotated rectangle for better measurements
            if len(contour) >= 5:
                rotated_rect = cv2.minAreaRect(contour)
                (center_x, center_y), (width_px, height_px), angle = rotated_rect
                
                # Ensure width is the smaller dimension
                if width_px > height_px:
                    width_px, height_px = height_px, width_px
                    angle = (angle + 90) % 180
            else:
                # Fallback for small contours
                width_px = float(w)
                height_px = float(h)
                angle = 0.0
                center_x, center_y = cx, cy
                
            # Calculate aspect ratio
            aspect_ratio = height_px / (width_px + 1e-6)
            
            # Classify defect based on aspect ratio
            if aspect_ratio >= scratch_aspect_ratio_threshold:
                classification = "Scratch"
            else:
                classification = "Pit/Dig"
            
            # Calculate intensity features for confidence scoring
            defect_pixels = processed_image[defect_mask > 0]
            if len(defect_pixels) > 0:
                mean_intensity = float(np.mean(defect_pixels))
                std_intensity = float(np.std(defect_pixels))
                
                # Simple confidence based on contrast
                # Get surrounding pixels
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                dilated = cv2.dilate(defect_mask, kernel)
                surrounding_mask = dilated - defect_mask
                surrounding_pixels = processed_image[surrounding_mask > 0]
                
                if len(surrounding_pixels) > 0:
                    surrounding_mean = float(np.mean(surrounding_pixels))
                    contrast = abs(mean_intensity - surrounding_mean)
                    # Normalize contrast to 0-1 range
                    confidence_score = min(1.0, contrast / 50.0)
                else:
                    confidence_score = 0.5
            else:
                confidence_score = 0.5
                
            # Build defect dictionary
            defect_dict = {
                "defect_id": f"D{defect_id_counter}",
                "zone": zone_name,
                "classification": classification,
                "confidence_score": float(confidence_score),
                "centroid_x_px": float(cx),
                "centroid_y_px": float(cy),
                "area_px": int(area_px),
                "length_px": float(height_px),  # Length is the larger dimension
                "width_px": float(width_px),    # Width is the smaller dimension
                "aspect_ratio": float(aspect_ratio),
                "bbox_x_px": int(x),
                "bbox_y_px": int(y),
                "bbox_w_px": int(w),
                "bbox_h_px": int(h),
                "rotated_rect_center_px": (float(center_x), float(center_y)),
                "rotated_rect_angle_deg": float(angle),
                "contour_points_px": contour.reshape(-1, 2).tolist()
            }
            
            # Add measurements in microns if scale is available
            if um_per_px and um_per_px > 0:
                defect_dict["length_um"] = float(height_px * um_per_px)
                defect_dict["width_um"] = float(width_px * um_per_px)
                defect_dict["area_um2"] = float(area_px * um_per_px * um_per_px)
                
                # For pits/digs, calculate effective diameter
                if classification == "Pit/Dig":
                    # Effective diameter = diameter of circle with same area
                    effective_diameter_um = 2 * np.sqrt(area_px / np.pi) * um_per_px
                    defect_dict["effective_diameter_um"] = float(effective_diameter_um)
            
            characterized_defects.append(defect_dict)
    
    logging.info(f"Characterized {len(characterized_defects)} defects from {len(all_detected_defects)} detected regions")
    
    # Sort defects by zone priority (Core first) and then by size
    zone_priority = {"Core": 0, "Cladding": 1, "Unknown": 2}
    characterized_defects.sort(
        key=lambda d: (
            zone_priority.get(d["zone"], 2),
            -d.get("length_um", d.get("length_px", 0))
        )
    )
    
    return characterized_defects


def calculate_defect_density(defects: List[Dict[str, Any]], zone_area_px: float) -> float:
    """
    Calculates defect density (defects per unit area).
    
    Args:
        defects: List of defect dictionaries
        zone_area_px: Area of the zone in pixels
        
    Returns:
        Defect density (total defect area / zone area)
    """
    if zone_area_px <= 0:
        return 0.0
        
    total_defect_area = sum(d.get('area_px', 0) for d in defects)
    return total_defect_area / zone_area_px


def analyze_defects_by_zone(characterized_defects: List[Dict[str, Any]], 
                           zone_masks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """
    Perform detailed zone-specific analysis of defects.
    
    Args:
        characterized_defects: List of characterized defect dictionaries
        zone_masks: Dictionary of zone masks
        
    Returns:
        Dictionary with zone-specific statistics
    """
    zone_stats = {}
    
    # Focus only on Core and Cladding
    for zone_name in ["Core", "Cladding"]:
        if zone_name not in zone_masks:
            continue
            
        zone_mask = zone_masks[zone_name]
        
        # Get defects in this zone
        zone_defects = [d for d in characterized_defects if d.get('zone') == zone_name]
        
        # Calculate zone area
        zone_area_px = np.sum(zone_mask > 0)
        
        # Separate by type
        scratches = [d for d in zone_defects if d['classification'] == 'Scratch']
        pits_digs = [d for d in zone_defects if d['classification'] == 'Pit/Dig']
        
        # Calculate statistics
        total_defect_area = sum(d.get('area_px', 0) for d in zone_defects)
        defect_density = calculate_defect_density(zone_defects, zone_area_px)
        
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
    fiber_type_key: str
) -> Tuple[str, List[str]]:
    """
    Applies pass/fail criteria based on IEC 61300-3-35 rules for Core and Cladding only.

    Args:
        characterized_defects: List of defect dictionaries from characterization
        fiber_type_key: The key for the fiber type (e.g., "single_mode_pc")

    Returns:
        A tuple: (overall_status: str, failure_reasons: List[str])
                 Overall status is "PASS" or "FAIL"
    """
    overall_status = "PASS"
    failure_reasons: List[str] = []

    try:
        # Get zone definitions which include pass/fail rules
        zone_rule_definitions = get_zone_definitions(fiber_type_key)
        if not zone_rule_definitions:
            logging.warning(f"No zone definitions found for fiber type '{fiber_type_key}'. Cannot apply rules.")
            return "ERROR_CONFIG", [f"No zone definitions for fiber type '{fiber_type_key}'."]

    except ValueError as e:
        logging.error(f"Cannot apply pass/fail rules: {e}")
        return "ERROR_CONFIG", [f"Configuration error for fiber type '{fiber_type_key}': {e}"]

    # Group defects by zone
    defects_by_zone: Dict[str, List[Dict[str, Any]]] = {"Core": [], "Cladding": []}
    
    for defect in characterized_defects:
        zone_name = defect.get("zone", "Unknown")
        if zone_name in defects_by_zone:
            defects_by_zone[zone_name].append(defect)

    # Apply rules for Core and Cladding only
    for zone_def in zone_rule_definitions:
        zone_name = zone_def.get("name")
        if zone_name not in ["Core", "Cladding"]:
            continue
            
        rules = zone_def.get("pass_fail_rules", {})
        current_zone_defects = defects_by_zone.get(zone_name, [])

        if not rules:
            logging.debug(f"No specific pass/fail rules defined for zone '{zone_name}'")
            continue

        # Separate defects by classification
        scratches_in_zone = [d for d in current_zone_defects if d["classification"] == "Scratch"]
        pits_digs_in_zone = [d for d in current_zone_defects if d["classification"] == "Pit/Dig"]

        # Check scratch count
        max_scratches_allowed = rules.get("max_scratches")
        if isinstance(max_scratches_allowed, int) and len(scratches_in_zone) > max_scratches_allowed:
            overall_status = "FAIL"
            failure_reasons.append(
                f"Zone '{zone_name}': Too many scratches ({len(scratches_in_zone)} > {max_scratches_allowed})"
            )
            logging.warning(f"FAIL: Zone '{zone_name}' scratch count exceeded")

        # Check pit/dig count
        max_pits_digs_allowed = rules.get("max_defects")
        if isinstance(max_pits_digs_allowed, int) and len(pits_digs_in_zone) > max_pits_digs_allowed:
            overall_status = "FAIL"
            failure_reasons.append(
                f"Zone '{zone_name}': Too many Pits/Digs ({len(pits_digs_in_zone)} > {max_pits_digs_allowed})"
            )
            logging.warning(f"FAIL: Zone '{zone_name}' pit/dig count exceeded")

        # Check defect sizes
        max_defect_size_um = rules.get("max_defect_size_um")
        max_scratch_length_um = rules.get("max_scratch_length_um")

        for defect in current_zone_defects:
            defect_type = defect["classification"]
            primary_dimension_um = defect.get("length_um")
            
            if primary_dimension_um is None:
                continue
                
            # Determine which size limit to use
            if defect_type == "Scratch":
                size_limit = max_scratch_length_um if max_scratch_length_um is not None else max_defect_size_um
            else:
                size_limit = max_defect_size_um
                
            if size_limit is not None and primary_dimension_um > size_limit:
                overall_status = "FAIL"
                reason = (
                    f"Zone '{zone_name}': {defect_type} '{defect['defect_id']}' "
                    f"size ({primary_dimension_um:.2f}µm) exceeds limit ({size_limit}µm)"
                )
                failure_reasons.append(reason)
                logging.warning(f"FAIL: {reason}")
                
        # Special rule for Core zone - any defect is a failure
        if zone_name == "Core" and len(current_zone_defects) > 0:
            if overall_status != "FAIL":  # Don't override if already failed
                overall_status = "FAIL"
                failure_reasons.append(f"Zone 'Core': Contains {len(current_zone_defects)} defect(s)")

    if overall_status == "PASS":
        logging.info(f"Pass/Fail Evaluation for '{fiber_type_key}': PASS")
    else:
        logging.warning(f"Pass/Fail Evaluation for '{fiber_type_key}': FAIL - {len(failure_reasons)} reason(s)")

    return overall_status, list(set(failure_reasons))  # Remove duplicate reasons


# --- Main function for testing ---
if __name__ == "__main__":
    # Testing code for standalone execution
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s')
    
    # Create dummy test data
    dummy_defects = [
        {
            'defect_mask': np.ones((10, 10), dtype=np.uint8) * 255,
            'zone': 'Core',
            'area_px': 100,
            'bounding_box': {'x': 10, 'y': 10, 'width': 10, 'height': 10},
            'centroid': (15, 15)
        },
        {
            'defect_mask': np.ones((5, 20), dtype=np.uint8) * 255,
            'zone': 'Cladding',
            'area_px': 100,
            'bounding_box': {'x': 30, 'y': 30, 'width': 20, 'height': 5},
            'centroid': (40, 32)
        }
    ]
    
    dummy_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    dummy_localization = {
        'cladding_center_xy': (50, 50),
        'cladding_radius_px': 40,
        'core_center_xy': (50, 50),
        'core_radius_px': 5
    }
    
    # Test characterization
    logging.info("Testing defect characterization...")
    characterized = characterize_and_classify_defects(
        dummy_defects, dummy_image, dummy_localization, 0.5, get_config()
    )
    
    for defect in characterized:
        logging.info(f"Defect: {defect['defect_id']} - {defect['classification']} in {defect['zone']}")
    
    # Test pass/fail rules
    logging.info("\nTesting pass/fail rules...")
    status, reasons = apply_pass_fail_rules(characterized, "single_mode_pc")
    logging.info(f"Status: {status}")
    if reasons:
        logging.info(f"Reasons: {reasons}")