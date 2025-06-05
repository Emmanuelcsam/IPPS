#!/usr/bin/env python3
# reporting.py

"""
D-Scope Blink: Reporting Module
===============================
This module is responsible for generating all output reports for each processed image,
including annotated images, detailed CSV files for defects, and polar defect
distribution histograms.
"""

import cv2 # OpenCV for drawing on images.
import numpy as np # NumPy for numerical operations, especially for polar histogram.
import matplotlib.pyplot as plt # Matplotlib for generating plots, specifically the polar histogram.
from matplotlib.projections.polar import PolarAxes # <<< ADD THIS LINE
import pandas as pd # Pandas for easy CSV file generation.
from pathlib import Path # Standard library for object-oriented path manipulation.
from typing import Dict, Any, Optional, List, Tuple, cast # Standard library for type hinting.
import logging # Standard library for logging events.
import datetime # Standard library for timestamping or adding dates to reports if needed.


# Attempt to import functions from other D-Scope Blink modules.
try:
    # Assuming config_loader.py is in the same directory or Python path.
    from config_loader import get_config # Function to access the global configuration.
except ImportError:
    # Fallback for standalone testing if config_loader is not directly available.
    logging.warning("Could not import get_config from config_loader. Using dummy config for standalone testing.")
    def get_config() -> Dict[str, Any]: # Define a dummy get_config for standalone testing.
        """Returns a dummy configuration for standalone testing."""
        return { # Simplified dummy config for reporting.
            "reporting": { # Reporting parameters.
                "annotated_image_dpi": 150,
                "defect_label_font_scale": 0.4,
                "defect_label_thickness": 1,
                "pass_fail_stamp_font_scale": 1.5,
                "pass_fail_stamp_thickness": 2,
                "zone_outline_thickness": 1,
                "defect_outline_thickness": 1,
                "display_timestamp_on_image": True, # New: Config to display timestamp on image
                "timestamp_format": "%Y-%m-%d %H:%M:%S" # New: Config for timestamp format on image
            },
            "zone_definitions_iec61300_3_35": { # Zone color definitions.
                "single_mode_pc": [ # Example fiber type.
                    {"name": "Core", "color_bgr": [255,0,0]}, # Blue for Core.
                    {"name": "Cladding", "color_bgr": [0,255,0]}, # Green for Cladding.
                    {"name": "Adhesive", "color_bgr": [0,255,255]}, # Yellow for Adhesive.
                    {"name": "Contact", "color_bgr": [255,0,255]}  # Magenta for Contact.
                ]
            },
            # Add other keys as needed for standalone testing.
        }

# --- Report Generation Functions ---

def generate_annotated_image(
    original_bgr_image: np.ndarray,
    analysis_results: Dict[str, Any],
    localization_data: Dict[str, Any],
    zone_masks: Dict[str, np.ndarray],
    fiber_type_key: str,
    output_path: Path,
    report_timestamp: Optional[datetime.datetime] = None # Optional: For displaying on image
) -> bool:
    """
    Generates and saves an annotated image showing zones, defects, and pass/fail status.

    Args:
        original_bgr_image: The original BGR image.
        analysis_results: Dictionary containing characterized defects and pass/fail status.
        localization_data: Dictionary with fiber localization info (centers, radii/ellipses).
        zone_masks: Dictionary of binary masks for each zone.
        fiber_type_key: Key for the fiber type (e.g., "single_mode_pc") for zone colors.
        output_path: Path object where the annotated image will be saved.
        report_timestamp: Optional datetime object for displaying on the image.

    Returns:
        True if the image was saved successfully, False otherwise.
    """
    annotated_image = original_bgr_image.copy() # Create a copy of the original image to draw on.
    config = get_config() # Get global configuration.
    report_cfg = config.get("reporting", {}) # Get reporting specific configurations.
    zone_defs_all_types = config.get("zone_definitions_iec61300_3_35", {}) # Get all zone definitions.
    current_fiber_zone_defs = zone_defs_all_types.get(fiber_type_key, []) # Get zone definitions for current fiber type.

    # --- Draw Zones ---
    # Zone colors from config.
    zone_color_map = {z["name"]: tuple(z["color_bgr"]) for z in current_fiber_zone_defs if "color_bgr" in z}
    zone_outline_thickness = report_cfg.get("zone_outline_thickness", 1) # Thickness for zone outlines.

    for zone_name, zone_mask_np in zone_masks.items(): # Iterate through zone masks.
        color = zone_color_map.get(zone_name, (128, 128, 128)) # Default to gray if color not defined.
        # Find contours of the zone mask to draw the boundary.
        contours, _ = cv2.findContours(zone_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours.
        cv2.drawContours(annotated_image, contours, -1, color, zone_outline_thickness) # Draw zone contours.
        
        if contours: # If contours exist for labeling.
            largest_contour = max(contours, key=cv2.contourArea) # Get largest contour.
            # Attempt to find a reasonable point for the label (e.g., top-most point)
            # This might need refinement for complex shapes.
            label_pos_candidate = tuple(largest_contour[largest_contour[:,:,1].argmin()][0]) # Top-most point.
            
            # Check if label position is too close to image border, adjust if necessary
            label_x = max(5, label_pos_candidate[0] - 20) # Ensure not too far left
            label_y = max(10, label_pos_candidate[1] - 5) # Ensure not too far up
            if label_y > annotated_image.shape[0] - 10: # If too close to bottom
                 label_y = annotated_image.shape[0] - 10

            cv2.putText(annotated_image, zone_name, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, report_cfg.get("defect_label_font_scale", 0.4) * 0.9,
                        color, report_cfg.get("defect_label_thickness", 1), cv2.LINE_AA) # Add zone label.

    # --- Draw Defects ---
    defects_list = analysis_results.get("characterized_defects", []) # Get list of characterized defects.
    defect_font_scale = report_cfg.get("defect_label_font_scale", 0.4) # Font scale for defect labels.
    defect_line_thickness = report_cfg.get("defect_outline_thickness", 1) # Thickness for defect outlines.

    for defect in defects_list: # Iterate through defects.
        classification = defect.get("classification", "Unknown") # Get defect classification.
        defect_color = (255, 0, 255) if classification == "Scratch" else (0, 165, 255) # Magenta for Scratch, Orange for Pit/Dig.

        contour_pts = defect.get("contour_points_px", None)
        if contour_pts:
            contour_np = np.array(contour_pts, dtype=np.int32).reshape(-1, 1, 2)
            # cv2.minAreaRect requires at least 3 points.
            if contour_np.shape[0] >= 3: # Check if there are enough points for minAreaRect
                rot_rect_params = cv2.minAreaRect(contour_np)
                box_points = cv2.boxPoints(rot_rect_params)
                box_points_int = np.array(box_points, dtype=np.int32) # Use np.int32 for drawContours [cite: 117, 328, 329, 330]
                cv2.drawContours(annotated_image, [box_points_int], 0, defect_color, defect_line_thickness) # Error here
            elif contour_np.size > 0: # Fallback for contours with < 3 points but > 0 points (e.g., a line)
                                      # Or if contour_np is empty after reshape but contour_pts existed.
                x, y, w, h = cv2.boundingRect(contour_np) # Use boundingRect for these cases
                if w > 0 and h > 0:
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), defect_color, defect_line_thickness)
                elif contour_np.shape[0] == 2: # Draw a line if it's two points
                    cv2.line(annotated_image, tuple(contour_np[0,0]), tuple(contour_np[1,0]), defect_color, defect_line_thickness)
                elif contour_np.shape[0] == 1: # Draw a circle if it's one point
                    cv2.circle(annotated_image, tuple(contour_np[0,0]), defect_line_thickness, defect_color, -1) # Small circle
            else: # Fallback to bounding box from defect data if contour_np is effectively empty
                x, y, w, h = (
                    defect.get("bbox_x_px", 0),
                    defect.get("bbox_y_px", 0),
                    defect.get("bbox_w_px", 0),
                    defect.get("bbox_h_px", 0),
                )
                if w > 0 and h > 0: # Only draw if width and height are valid
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), defect_color, defect_line_thickness)
        else: # Fallback to bounding box if contour_points_px is not present
            x, y, w, h = (
                defect.get("bbox_x_px", 0),
                defect.get("bbox_y_px", 0),
                defect.get("bbox_w_px", 0),
                defect.get("bbox_h_px", 0),
            )
            if w > 0 and h > 0: # Only draw if width and height are valid
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), defect_color, defect_line_thickness)

        # Add label (ID, type, primary dimension).
        defect_id = defect.get("defect_id", "N/A") # Get defect ID.
        primary_dim_str = "" # Initialize primary dimension string.
        if defect.get("length_um") is not None: # More robust check
            primary_dim_str = f"{defect['length_um']:.1f}µm"
        elif defect.get("effective_diameter_um") is not None: # More robust check
            primary_dim_str = f"{defect['effective_diameter_um']:.1f}µm"
        elif defect.get("length_px") is not None: # More robust check
            primary_dim_str = f"{defect['length_px']:.0f}px"
        
        label_text = f"{defect_id.split('_')[-1]}:{classification[:3]},{primary_dim_str}" # Create label text.
        label_x = defect.get("bbox_x_px", 0) # Get label x position.
        label_y = defect.get("bbox_y_px", 0) - 5 # Get label y position (slightly above bbox).
        if label_y < 10: label_y = defect.get("bbox_y_px",0) + defect.get("bbox_h_px",10) + 10 # Adjust if too close to top.

        cv2.putText(annotated_image, label_text, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, defect_font_scale, defect_color,
                    report_cfg.get("defect_label_thickness", 1), cv2.LINE_AA) # Add defect label.

    # --- Add PASS/FAIL Stamp ---
    status = analysis_results.get("overall_status", "UNKNOWN") # Get overall status.
    status_color = (0, 255, 0) if status == "PASS" else (0, 0, 255) # Green for PASS, Red for FAIL/other.
    stamp_font_scale = report_cfg.get("pass_fail_stamp_font_scale", 1.5) # Font scale for stamp.
    stamp_thickness = report_cfg.get("pass_fail_stamp_thickness", 2) # Thickness for stamp.
    
    text_size_status, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, stamp_font_scale, stamp_thickness) # Get text size.
    text_x_status = 10 # X position for stamp.
    text_y_status = text_size_status[1] + 10 # Y position for stamp.
    cv2.putText(annotated_image, status, (text_x_status, text_y_status),
                cv2.FONT_HERSHEY_SIMPLEX, stamp_font_scale, status_color, stamp_thickness, cv2.LINE_AA) # Add PASS/FAIL stamp.

    # Add some summary info like filename and total defects.
    img_filename = Path(analysis_results.get("image_filename", "unknown.png")).name # Get image filename.
    total_defects = analysis_results.get("total_defect_count", 0) # Get total defect count.
    
    info_text_y_start = text_y_status + text_size_status[1] + 15 # Starting Y for info text.
    
    cv2.putText(annotated_image, f"File: {img_filename}", (10, info_text_y_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA) # Add filename.
    info_text_y_start += 20
    cv2.putText(annotated_image, f"Defects: {total_defects}", (10, info_text_y_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA) # Add defect count.
    info_text_y_start += 20

    # Add timestamp to image if configured and available
    if report_timestamp and report_cfg.get("display_timestamp_on_image", False):
        ts_format = report_cfg.get("timestamp_format", "%Y-%m-%d %H:%M:%S")
        timestamp_str = report_timestamp.strftime(ts_format)
        cv2.putText(annotated_image, f"Generated: {timestamp_str}", (10, info_text_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)


    # --- Save the Annotated Image ---
    try:
        # DPI is relevant if saving via Matplotlib or other libs that respect DPI.
        # cv2.imwrite does not embed DPI in all formats, but good to have if config changes.
        # dpi_val = report_cfg.get("annotated_image_dpi", 150) # Already have this, but unused by cv2.imwrite directly
        cv2.imwrite(str(output_path), annotated_image) # Save the annotated image.
        logging.info(f"Annotated image saved successfully to '{output_path}'.")
        return True # Return True on success.
    except Exception as e: # Handle errors during saving.
        logging.error(f"Failed to save annotated image to '{output_path}': {e}")
        return False # Return False on failure.

def generate_defect_csv_report(
    analysis_results: Dict[str, Any],
    output_path: Path
) -> bool:
    """
    Generates a CSV file listing all detected defects and their properties.

    Args:
        analysis_results: Dictionary containing characterized defects.
        output_path: Path object where the CSV report will be saved.

    Returns:
        True if the CSV was saved successfully, False otherwise.
    """
    defects_list = analysis_results.get("characterized_defects", []) # Get list of characterized defects.
    if not defects_list: # If no defects.
        logging.info(f"No defects to report for {output_path.name}. CSV not generated.")
        try: 
            cols = ["defect_id", "zone", "classification", "confidence_score",
                    "centroid_x_px", "centroid_y_px", "area_px", "length_px", "width_px",
                    "aspect_ratio_oriented", "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px",
                    "area_um2", "length_um", "width_um", "effective_diameter_um",
                    "rotated_rect_center_px", "rotated_rect_angle_deg"]
            # Create an empty DataFrame with all possible headers to ensure consistency even if empty.
            df = pd.DataFrame([], columns=cols) 
            df.to_csv(output_path, index=False, encoding='utf-8') # Save empty CSV.
            logging.info(f"Empty defect report CSV (with headers) saved to '{output_path}'.")
            return True 
        except Exception as e: 
            logging.error(f"Failed to save empty defect report to '{output_path}': {e}")
            return False 


    try:
        df = pd.DataFrame(defects_list) 
        
        report_columns = [ 
            "defect_id", "zone", "classification", "confidence_score",
            "centroid_x_px", "centroid_y_px",
            "length_um", "width_um", "effective_diameter_um", "area_um2",
            "length_px", "width_px", "area_px",
            "aspect_ratio_oriented",
            "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px",
            "rotated_rect_center_px", "rotated_rect_angle_deg"
            # "contour_points_px" # Usually too verbose for main CSV.
        ]
        final_columns = [col for col in report_columns if col in df.columns] 

        df_report = df[final_columns] 

        df_report.to_csv(output_path, index=False, encoding='utf-8') 
        logging.info(f"Defect CSV report saved successfully to '{output_path}'.")
        return True 
    except Exception as e: 
        logging.error(f"Failed to save defect CSV report to '{output_path}': {e}")
        return False 

def generate_polar_defect_histogram(
    analysis_results: Dict[str, Any],
    localization_data: Dict[str, Any],
    zone_masks: Dict[str, np.ndarray],
    fiber_type_key: str,
    output_path: Path
) -> bool:
    """
    Generates and saves a polar histogram showing defect distribution.

    Args:
        analysis_results: Dictionary containing characterized defects.
        localization_data: Dictionary with fiber localization info (center is crucial).
        zone_masks: Dictionary of zone masks for plotting boundaries.
        fiber_type_key: Key for fiber type to get zone colors.
        output_path: Path object where the histogram PNG will be saved.

    Returns:
        True if the histogram was saved successfully, False otherwise.
    """
    defects_list = analysis_results.get("characterized_defects", [])
    fiber_center_xy = localization_data.get("cladding_center_xy")

    if not defects_list:
        logging.info(f"No defects to plot for polar histogram for {output_path.name}.")
        return True

    if fiber_center_xy is None:
        logging.error("Cannot generate polar histogram: Fiber center not localized.")
        return False

    config = get_config()
    report_cfg = config.get("reporting", {}) # Get reporting config for DPI
    zone_defs_all_types = config.get("zone_definitions_iec61300_3_35", {})
    current_fiber_zone_defs = zone_defs_all_types.get(fiber_type_key, [])
    zone_color_map_bgr = {z["name"]: tuple(z["color_bgr"]) for z in current_fiber_zone_defs if "color_bgr" in z}

    center_x, center_y = fiber_center_xy
    angles_rad: List[float] = []
    radii_px: List[float] = []
    defect_plot_colors_rgb: List[Tuple[float,float,float]] = []

    for defect in defects_list:
        cx_px = defect.get("centroid_x_px")
        cy_px = defect.get("centroid_y_px")

        if cx_px is None or cy_px is None: # Skip if centroid is missing
            logging.warning(f"Defect {defect.get('defect_id')} missing centroid, cannot plot in polar histogram.")
            continue

        dx = cx_px - center_x
        dy = cy_px - center_y

        angle = np.arctan2(dy, dx)
        radius = np.sqrt(dx**2 + dy**2)

        angles_rad.append(angle)
        radii_px.append(radius)

        classification = defect.get("classification", "Unknown")
        bgr_color = (255, 0, 255) if classification == "Scratch" else (0, 165, 255) # Magenta for Scratch, Orange for Pit/Dig
        rgb_color_normalized = (bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0)
        defect_plot_colors_rgb.append(rgb_color_normalized)

    # --- Start of lines to modify/check ---
    fig, ax_untyped = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax: PolarAxes = ax_untyped # Explicitly type hint ax
    # --- End of lines to modify/check (ax definition) ---

    if angles_rad and radii_px:
        ax.scatter(angles_rad, radii_px, c=defect_plot_colors_rgb, s=50, alpha=0.75, edgecolors='k')

    max_display_radius = 0
    plotted_zone_labels = set() # To avoid duplicate labels in legend
    for zone_name, zone_mask_np in sorted(zone_masks.items(), key=lambda item: item[0]): # Sort for consistent legend order
        if np.sum(zone_mask_np) > 0:
            y_coords, x_coords = np.where(zone_mask_np > 0)
            if y_coords.size > 0:
                distances_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                zone_outer_radius_px = np.max(distances_from_center) if distances_from_center.size > 0 else 0
                max_display_radius = max(max_display_radius, zone_outer_radius_px)

                zone_bgr = zone_color_map_bgr.get(zone_name, (128,128,128))
                zone_rgb_normalized = (zone_bgr[2]/255.0, zone_bgr[1]/255.0, zone_bgr[0]/255.0)

                label_to_use = None
                if zone_name not in plotted_zone_labels and zone_outer_radius_px > 0:
                    label_to_use = zone_name
                    plotted_zone_labels.add(zone_name)

                ax.plot(np.linspace(0, 2 * np.pi, 100), [zone_outer_radius_px] * 100,
                        color=zone_rgb_normalized, linestyle='--', label=label_to_use)

# ... [previous code in the function] ...
    current_r_max = 100 # Default r_max if no zones and no defect radii
    if max_display_radius > 0: 
        current_r_max = max_display_radius * 1.1
    elif radii_px: # Check if radii_px is not empty
        current_r_max = max(radii_px) * 1.2 

    ax.set_rlim(0, current_r_max) 

    # Corrected line:
    # ax.set_rticks(np.linspace(0, ax.get_rlim()[1], 5)) # Original problematic line
    ax.set_rticks(np.linspace(0, ax.get_ylim()[1], 5)) # Corrected: Use get_ylim() for PolarAxes

    ax.set_rlabel_position(22.5) 
# ... [rest of the function] ...

    ax.grid(True)
    ax.set_title(f"Defect Distribution: {output_path.stem.replace('_histogram','')}", va='bottom', pad=20)
    if any(label is not None for label in ax.get_legend_handles_labels()[1]):
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0)) # Adjusted legend position slightly


    try:
        plt.tight_layout()
        fig.savefig(output_path, dpi=report_cfg.get("annotated_image_dpi", 150)) # Use DPI from config
        plt.close(fig)
        logging.info(f"Polar defect histogram saved successfully to '{output_path}'.")
        return True
    except Exception as e:
        logging.error(f"Failed to save polar defect histogram to '{output_path}': {e}")
        plt.close(fig)
        return False

# --- Main function for testing this module (optional) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s') 

    # --- Dummy Data for Testing (mimicking outputs from analysis.py and image_processing.py) ---
    dummy_image_base_name = "test_report_image"
    dummy_image_ext = ".png"
    # Create a dummy image if it doesn't exist.
    # For testing, ensure a unique name if run multiple times or handle existing files.
    img = np.full((300, 400, 3), (200, 200, 200), dtype=np.uint8) 
    cv2.circle(img, (200,150), 80, (180,180,180), -1) # "Cladding"
    cv2.circle(img, (200,150), 30, (150,150,150), -1) # "Core"
    
    # Create a temporary dummy image for the test run
    temp_test_image_dir = Path("temp_test_images")
    temp_test_image_dir.mkdir(exist_ok=True)
    dummy_image_path_obj = temp_test_image_dir / f"{dummy_image_base_name}{dummy_image_ext}"
    cv2.imwrite(str(dummy_image_path_obj), img)
    logging.info(f"Created dummy image at {dummy_image_path_obj}")


    dummy_original_bgr = cv2.imread(str(dummy_image_path_obj)) 
    if dummy_original_bgr is None: 
        logging.error(f"Failed to load dummy image for reporting test: {dummy_image_path_obj}")
        exit() 

    current_timestamp = datetime.datetime.now()
    timestamp_str_file = current_timestamp.strftime("%Y%m%d_%H%M%S")


    dummy_analysis_results = { 
        "image_filename": str(dummy_image_path_obj), # Use the actual path of the created dummy
        "overall_status": "FAIL",
        "failure_reasons": ["Zone 'Core': Scratch 'D1_S1' size (11.00µm) exceeds limit (3.0µm)"],
        "total_defect_count": 3, # Added one more defect for testing missing contour
        "characterized_defects": [
            {
                "defect_id": "D1_S1", "zone": "Core", "classification": "Scratch", "confidence_score": 0.95,
                "centroid_x_px": 190, "centroid_y_px": 140, "area_px": 50, "length_px": 22, "width_px": 2.5,
                "aspect_ratio_oriented": 8.8, "bbox_x_px": 180, "bbox_y_px": 130, "bbox_w_px": 15, "bbox_h_px": 25,
                "rotated_rect_center_px": (190.0, 140.0), "rotated_rect_angle_deg": 45.0,
                "contour_points_px": [[180,130],[195,130],[195,155],[180,155]], 
                "area_um2": 12.5, "length_um": 11.0, "width_um": 1.25
            },
            { # Defect with only two contour points to test line drawing fallback
                "defect_id": "D1_L1", "zone": "Cladding", "classification": "Scratch", "confidence_score": 0.88,
                "centroid_x_px": 230, "centroid_y_px": 170, "area_px": 0, "length_px": 10, "width_px": 0, 
                "aspect_ratio_oriented": 0, "bbox_x_px": 225, "bbox_y_px": 165, "bbox_w_px": 10, "bbox_h_px": 10,
                "rotated_rect_center_px": (230.0, 170.0), "rotated_rect_angle_deg": 0.0,
                "contour_points_px": [[225,165],[235,175]], # Only two points - should become a line
                "area_um2": 0, "length_um":5.0, "width_um":0
            },
            { # Defect with no contour points to test fallback to bbox
                "defect_id": "D2_P2", "zone": "Contact", "classification": "Pit/Dig", "confidence_score": 0.70,
                "centroid_x_px": 150, "centroid_y_px": 100, "area_px": 20, "length_px": 5, "width_px": 4,
                "bbox_x_px": 148, "bbox_y_px": 98, "bbox_w_px": 5, "bbox_h_px": 5,
                "area_um2": 5.0, "effective_diameter_um": 2.51
                # No "contour_points_px"
            }
        ]
    }
    dummy_localization_data = { 
        "cladding_center_xy": (200, 150),
        "cladding_radius_px": 80.0,
        "core_center_xy": (200, 150),
        "core_radius_px": 30.0
    }
    dummy_fiber_type = "single_mode_pc" 
    dummy_zone_masks_hist = {} 
    _h, _w = dummy_original_bgr.shape[:2] 
    _Y, _X = np.ogrid[:_h, :_w] 
    _center_x, _center_y = dummy_localization_data["cladding_center_xy"] 
    _dist_sq = (_X - _center_x)**2 + (_Y - _center_y)**2 
    
    _core_r_px = dummy_localization_data["core_radius_px"] 
    _clad_r_px = dummy_localization_data["cladding_radius_px"] 
    dummy_zone_masks_hist["Core"] = (_dist_sq < _core_r_px**2).astype(np.uint8) * 255 
    dummy_zone_masks_hist["Cladding"] = ((_dist_sq >= _core_r_px**2) & (_dist_sq < _clad_r_px**2)).astype(np.uint8) * 255 
    # Simplified Contact zone for testing: everything outside cladding up to a certain radius for visualization
    _contact_outer_r_px = _clad_r_px * 1.5 
    dummy_zone_masks_hist["Contact"] = ((_dist_sq >= _clad_r_px**2) & (_dist_sq < _contact_outer_r_px**2)).astype(np.uint8) * 255


    test_output_dir = Path("test_reporting_output") 
    test_output_dir.mkdir(exist_ok=True) 

    # --- Test Case 1: Generate Annotated Image ---
    logging.info("\n--- Test Case 1: Generate Annotated Image ---")
    annotated_img_path = test_output_dir / f"{dummy_image_base_name}_{timestamp_str_file}_annotated.png"
    success_annotated = generate_annotated_image( 
        dummy_original_bgr, dummy_analysis_results, dummy_localization_data, 
        dummy_zone_masks_hist, dummy_fiber_type, annotated_img_path,
        report_timestamp=current_timestamp # Pass timestamp for display
    )
    logging.info(f"Annotated image generation success: {success_annotated}")

    # --- Test Case 2: Generate CSV Report ---
    logging.info("\n--- Test Case 2: Generate CSV Report ---")
    csv_report_path = test_output_dir / f"{dummy_image_base_name}_{timestamp_str_file}_report.csv" 
    success_csv = generate_defect_csv_report(dummy_analysis_results, csv_report_path) 
    logging.info(f"CSV report generation success: {success_csv}")

    # --- Test Case 3: Generate Polar Defect Histogram ---
    logging.info("\n--- Test Case 3: Generate Polar Defect Histogram ---")
    histogram_path = test_output_dir / f"{dummy_image_base_name}_{timestamp_str_file}_histogram.png" 
    success_hist = generate_polar_defect_histogram( 
        dummy_analysis_results, dummy_localization_data, dummy_zone_masks_hist, dummy_fiber_type, histogram_path
    )
    logging.info(f"Polar histogram generation success: {success_hist}")
    
    # Test with no defects
    logging.info("\n--- Test Case 4: Reporting with NO defects ---")
    dummy_analysis_no_defects = dummy_analysis_results.copy() # Start with a copy
    # Modify the copy for the no-defects scenario
    dummy_analysis_no_defects["characterized_defects"] = []
    dummy_analysis_no_defects["total_defect_count"] = 0
    dummy_analysis_no_defects["overall_status"] = "PASS"
    dummy_analysis_no_defects["failure_reasons"] = []
    
    no_defect_timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    no_defect_csv_path = test_output_dir / f"{dummy_image_base_name}_{no_defect_timestamp_str}_no_defects_report.csv"
    generate_defect_csv_report(dummy_analysis_no_defects, no_defect_csv_path)

    no_defect_hist_path = test_output_dir / f"{dummy_image_base_name}_{no_defect_timestamp_str}_no_defects_histogram.png"
    generate_polar_defect_histogram(dummy_analysis_no_defects, dummy_localization_data, dummy_zone_masks_hist, dummy_fiber_type, no_defect_hist_path)
    
    no_defect_annotated_img_path = test_output_dir / f"{dummy_image_base_name}_{no_defect_timestamp_str}_no_defects_annotated.png"
    generate_annotated_image(
        dummy_original_bgr, dummy_analysis_no_defects, dummy_localization_data,
        dummy_zone_masks_hist, dummy_fiber_type, no_defect_annotated_img_path,
        report_timestamp=datetime.datetime.now()
    )
    
    # Test with defects but no centroids for polar histogram
    logging.info("\n--- Test Case 5: Polar histogram with NO defect centroids ---")
    dummy_analysis_no_centroids = dummy_analysis_results.copy()
    # Make a deep copy of defects to modify them
    defects_no_centroids = [d.copy() for d in dummy_analysis_results["characterized_defects"]]
    for defect in defects_no_centroids:
        if "centroid_x_px" in defect: del defect["centroid_x_px"]
        if "centroid_y_px" in defect: del defect["centroid_y_px"]
    dummy_analysis_no_centroids["characterized_defects"] = defects_no_centroids
    
    no_centroids_timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    no_centroids_hist_path = test_output_dir / f"{dummy_image_base_name}_{no_centroids_timestamp_str}_no_centroids_histogram.png"
    generate_polar_defect_histogram(dummy_analysis_no_centroids, dummy_localization_data, dummy_zone_masks_hist, dummy_fiber_type, no_centroids_hist_path)


    # Clean up dummy image created by this test script
    if dummy_image_path_obj.exists():
        dummy_image_path_obj.unlink()
        logging.info(f"Cleaned up dummy image: {dummy_image_path_obj}")
        try:
            temp_test_image_dir.rmdir() # Remove directory if empty
            logging.info(f"Cleaned up temporary image directory: {temp_test_image_dir}")
        except OSError:
            logging.info(f"Temporary image directory not empty, not removed: {temp_test_image_dir}")