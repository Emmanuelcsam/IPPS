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
import pandas as pd # Pandas for easy CSV file generation.
from pathlib import Path # Standard library for object-oriented path manipulation.
from typing import Dict, Any, Optional, List, Tuple # Standard library for type hinting.
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
                "defect_outline_thickness": 1
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
    output_path: Path
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

    cl_center = localization_data.get("cladding_center_xy") # Get cladding center.
    cl_ellipse_params = localization_data.get("cladding_ellipse_params") # Get cladding ellipse parameters.

    for zone_name, zone_mask_np in zone_masks.items(): # Iterate through zone masks.
        color = zone_color_map.get(zone_name, (128, 128, 128)) # Default to gray if color not defined.
        # Find contours of the zone mask to draw the boundary.
        contours, _ = cv2.findContours(zone_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours.
        cv2.drawContours(annotated_image, contours, -1, color, zone_outline_thickness) # Draw zone contours.
        
        # Add zone name label near the zone boundary
        if contours: # If contours exist for labeling.
            # Find a point on the contour for the label (e.g., top-most point of the largest contour).
            # This can be improved for better label placement.
            largest_contour = max(contours, key=cv2.contourArea) # Get largest contour.
            label_pos_candidate = tuple(largest_contour[largest_contour[:,:,1].argmin()][0]) # Top-most point.
            cv2.putText(annotated_image, zone_name, (label_pos_candidate[0], label_pos_candidate[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, report_cfg.get("defect_label_font_scale", 0.4) * 0.9,
                        color, report_cfg.get("defect_label_thickness", 1), cv2.LINE_AA) # Add zone label.

    # --- Draw Defects ---
    defects_list = analysis_results.get("characterized_defects", []) # Get list of characterized defects.
    defect_font_scale = report_cfg.get("defect_label_font_scale", 0.4) # Font scale for defect labels.
    defect_line_thickness = report_cfg.get("defect_outline_thickness", 1) # Thickness for defect outlines.

    for defect in defects_list: # Iterate through defects.
        classification = defect.get("classification", "Unknown") # Get defect classification.
        defect_color = (255, 0, 255) if classification == "Scratch" else (0, 165, 255) # Magenta for Scratch, Orange for Pit/Dig.

        # Draw rotated rectangle for precise defect outline.
        # minAreaRect points are ((center_x, center_y), (width, height), angle)
        # We need to reconstruct from stored data if not directly available.
        # Assuming 'contour_points_px' is stored.
        contour_np = np.array(defect.get("contour_points_px"), dtype=np.int32).reshape((-1,1,2)) # Reconstruct contour.
        if contour_np.size > 0: # If contour points exist.
            rot_rect_params = cv2.minAreaRect(contour_np) # Recalculate minAreaRect.
            box_points = cv2.boxPoints(rot_rect_params) # Get corner points of the rotated rectangle.
            box_points_int = np.intp(box_points) # Convert points to integer.
            cv2.drawContours(annotated_image, [box_points_int], 0, defect_color, defect_line_thickness) # Draw rotated rectangle.
        else: # Fallback to bounding box if contour points not available.
            x, y, w, h = defect.get("bbox_x_px",0), defect.get("bbox_y_px",0), defect.get("bbox_w_px",0), defect.get("bbox_h_px",0)
            cv2.rectangle(annotated_image, (x,y), (x+w, y+h), defect_color, defect_line_thickness) # Draw bounding box.


        # Add label (ID, type, primary dimension).
        defect_id = defect.get("defect_id", "N/A") # Get defect ID.
        primary_dim_str = "" # Initialize primary dimension string.
        if "length_um" in defect and defect["length_um"] is not None: # If length in um available.
            primary_dim_str = f"{defect['length_um']:.1f}µm"
        elif "effective_diameter_um" in defect and defect["effective_diameter_um"] is not None: # If diameter in um available.
            primary_dim_str = f"{defect['effective_diameter_um']:.1f}µm"
        elif "length_px" in defect: # If length in px available.
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
    
    # Position the stamp (e.g., top-left corner).
    text_size, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, stamp_font_scale, stamp_thickness) # Get text size.
    text_x = 10 # X position for stamp.
    text_y = text_size[1] + 10 # Y position for stamp.
    cv2.putText(annotated_image, status, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, stamp_font_scale, status_color, stamp_thickness, cv2.LINE_AA) # Add PASS/FAIL stamp.

    # Add some summary info like filename and total defects.
    img_filename = Path(analysis_results.get("image_filename", "unknown.png")).name # Get image filename.
    total_defects = analysis_results.get("total_defect_count", 0) # Get total defect count.
    info_text_y_start = text_y + text_size[1] + 15 # Starting Y for info text.
    
    cv2.putText(annotated_image, f"File: {img_filename}", (10, info_text_y_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA) # Add filename.
    cv2.putText(annotated_image, f"Defects: {total_defects}", (10, info_text_y_start + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA) # Add defect count.


    # --- Save the Annotated Image ---
    try:
        # Get DPI from config for saving.
        dpi_val = report_cfg.get("annotated_image_dpi", 150)
        # Note: cv2.imwrite does not directly use DPI. DPI is relevant if saving via Matplotlib
        # or if other image libraries are used that respect DPI metadata.
        # For consistent output, ensure image resolution is sufficient.
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
        # Optionally, create an empty CSV or a CSV with a "No Defects Found" message.
        try: # Attempt to create an empty CSV with headers.
            # Define expected columns based on 'characterize_and_classify_defects' output.
            cols = ["defect_id", "zone", "classification", "confidence_score",
                    "centroid_x_px", "centroid_y_px", "area_px", "length_px", "width_px",
                    "aspect_ratio_oriented", "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px",
                    "area_um2", "length_um", "width_um", "effective_diameter_um"]
            # Filter to only include columns that might actually be present if some are optional
            present_cols = list(set(cols).intersection(defects_list[0].keys())) if defects_list else cols
            df = pd.DataFrame([], columns=present_cols) # Create empty DataFrame with headers.
            df.to_csv(output_path, index=False, encoding='utf-8') # Save empty CSV.
            logging.info(f"Empty defect report CSV saved to '{output_path}'.")
            return True # Return True.
        except Exception as e: # Handle errors.
            logging.error(f"Failed to save empty defect report to '{output_path}': {e}")
            return False # Return False.


    try:
        # Create a Pandas DataFrame from the list of defect dictionaries.
        df = pd.DataFrame(defects_list) # Create DataFrame.
        
        # Select and order columns for the report.
        # Prioritize micron measurements if available, then pixel.
        report_columns = [ # Define desired column order.
            "defect_id", "zone", "classification", "confidence_score",
            "centroid_x_px", "centroid_y_px",
            "length_um", "width_um", "effective_diameter_um", "area_um2",
            "length_px", "width_px", "area_px",
            "aspect_ratio_oriented",
            "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px",
            "rotated_rect_center_px", "rotated_rect_angle_deg"
            # "contour_points_px" # Usually too verbose for main CSV.
        ]
        # Filter to only include columns that actually exist in the DataFrame.
        final_columns = [col for col in report_columns if col in df.columns] # Get existing columns.

        df_report = df[final_columns] # Create DataFrame with selected columns.

        df_report.to_csv(output_path, index=False, encoding='utf-8') # Save DataFrame to CSV.
        logging.info(f"Defect CSV report saved successfully to '{output_path}'.")
        return True # Return True on success.
    except Exception as e: # Handle errors during saving.
        logging.error(f"Failed to save defect CSV report to '{output_path}': {e}")
        return False # Return False on failure.

def generate_polar_defect_histogram(
    analysis_results: Dict[str, Any],
    localization_data: Dict[str, Any],
    zone_masks: Dict[str, np.ndarray], # For drawing zone boundaries on histogram
    fiber_type_key: str, # For getting zone colors
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
    defects_list = analysis_results.get("characterized_defects", []) # Get list of characterized defects.
    fiber_center_xy = localization_data.get("cladding_center_xy") # Get fiber center.

    if not defects_list: # If no defects.
        logging.info(f"No defects to plot for polar histogram for {output_path.name}.")
        # Optionally, create a blank plot or skip.
        return True # Consider it success as there's nothing to plot.
    
    if fiber_center_xy is None: # If fiber center not found.
        logging.error("Cannot generate polar histogram: Fiber center not localized.")
        return False # Return False.

    config = get_config() # Get global config.
    zone_defs_all_types = config.get("zone_definitions_iec61300_3_35", {}) # Get all zone definitions.
    current_fiber_zone_defs = zone_defs_all_types.get(fiber_type_key, []) # Get current fiber type definitions.
    zone_color_map_bgr = {z["name"]: tuple(z["color_bgr"]) for z in current_fiber_zone_defs if "color_bgr" in z} # Zone colors.

    center_x, center_y = fiber_center_xy # Unpack fiber center.
    angles_rad: List[float] = [] # List for defect angles.
    radii_px: List[float] = [] # List for defect radii.
    defect_plot_colors_rgb: List[Tuple[float,float,float]] = [] # List for defect plot colors (RGB for Matplotlib).

    for defect in defects_list: # Iterate through defects.
        cx_px = defect.get("centroid_x_px", center_x) # Get defect centroid X.
        cy_px = defect.get("centroid_y_px", center_y) # Get defect centroid Y.

        # Calculate relative position to fiber center.
        # OpenCV coordinate system: y increases downwards.
        # atan2 takes (y, x).
        dx = cx_px - center_x # Delta X.
        dy = cy_px - center_y # Delta Y.

        angle = np.arctan2(dy, dx) # Calculate angle using arctan2.
        radius = np.sqrt(dx**2 + dy**2) # Calculate radius.

        angles_rad.append(angle) # Add angle to list.
        radii_px.append(radius) # Add radius to list.
        
        # Defect color based on type
        classification = defect.get("classification", "Unknown") # Get classification.
        bgr_color = (0, 165, 255) if classification == "Pit/Dig" else (255, 0, 255) # Orange for Pit/Dig, Magenta for Scratch.
        rgb_color_normalized = (bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0) # Convert BGR to RGB normalized.
        defect_plot_colors_rgb.append(rgb_color_normalized) # Add color to list.


    # --- Create Polar Plot ---
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8)) # Create polar plot.
    
    if angles_rad and radii_px: # If defect data exists.
        ax.scatter(angles_rad, radii_px, c=defect_plot_colors_rgb, s=50, alpha=0.75, edgecolors='k') # Scatter plot of defects.

    # Draw zone boundaries on the polar plot.
    # This requires knowing the pixel radii of the zones.
    # The zone_masks can be used to find the maximum radius for each circular zone.
    max_display_radius = 0 # Initialize max display radius.
    for zone_name, zone_mask_np in zone_masks.items(): # Iterate through zone masks.
        if np.sum(zone_mask_np) > 0: # If mask is not empty.
            # Find the maximum distance of any white pixel in the mask from the fiber_center_xy.
            # This is an approximation of the outer radius of the zone.
            y_coords, x_coords = np.where(zone_mask_np > 0) # Get coordinates of mask pixels.
            if y_coords.size > 0: # If pixels found.
                distances_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2) # Calculate distances.
                zone_outer_radius_px = np.max(distances_from_center) if distances_from_center.size > 0 else 0 # Get max distance.
                max_display_radius = max(max_display_radius, zone_outer_radius_px) # Update max display radius.

                zone_bgr = zone_color_map_bgr.get(zone_name, (128,128,128)) # Get zone color.
                zone_rgb_normalized = (zone_bgr[2]/255.0, zone_bgr[1]/255.0, zone_bgr[0]/255.0) # Convert to RGB.
                
                ax.plot(np.linspace(0, 2 * np.pi, 100), [zone_outer_radius_px] * 100, # Plot zone circle.
                        color=zone_rgb_normalized, linestyle='--', label=zone_name if zone_outer_radius_px > 0 else None)
    
    ax.set_rmax(max_display_radius * 1.1 if max_display_radius > 0 else (max(radii_px)*1.2 if radii_px else 100) ) # Set radial limit.
    ax.set_rticks(np.linspace(0, ax.get_rmax(), 5)) # Set radial ticks.
    ax.set_rlabel_position(22.5) # Position radial labels.
    ax.grid(True) # Enable grid.
    ax.set_title(f"Defect Distribution: {output_path.stem.replace('_histogram','')}", va='bottom') # Set title.
    if any(label is not None for label in ax.get_legend_handles_labels()[1]): # Check if any labels exist for legend
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0)) # Add legend.


    try:
        plt.tight_layout() # Adjust layout.
        fig.savefig(output_path, dpi=get_config().get("reporting",{}).get("annotated_image_dpi", 150)) # Save figure.
        plt.close(fig) # Close the figure to free memory.
        logging.info(f"Polar defect histogram saved successfully to '{output_path}'.")
        return True # Return True on success.
    except Exception as e: # Handle errors during saving.
        logging.error(f"Failed to save polar defect histogram to '{output_path}': {e}")
        plt.close(fig) # Ensure figure is closed even on error.
        return False # Return False on failure.

# --- Main function for testing this module (optional) ---
if __name__ == "__main__":
    # This block is for testing the reporting module independently.
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s') # Basic logging.

    # --- Dummy Data for Testing (mimicking outputs from analysis.py and image_processing.py) ---
    dummy_image_path = "test_report_image.png" # Dummy image path.
    # Create a dummy image if it doesn't exist.
    if not Path(dummy_image_path).exists(): # Check if dummy image exists.
        img = np.full((300, 400, 3), (200, 200, 200), dtype=np.uint8) # Create dummy image.
        cv2.circle(img, (200,150), 80, (180,180,180), -1) # "Cladding"
        cv2.circle(img, (200,150), 30, (150,150,150), -1) # "Core"
        cv2.imwrite(dummy_image_path, img) # Save dummy image.

    dummy_original_bgr = cv2.imread(dummy_image_path) # Load dummy BGR image.
    if dummy_original_bgr is None: # Check if loading failed.
        logging.error(f"Failed to create/load dummy image for reporting test: {dummy_image_path}")
        exit() # Exit if failed.

    dummy_analysis_results = { # Dummy analysis results.
        "image_filename": dummy_image_path,
        "overall_status": "FAIL",
        "failure_reasons": ["Zone 'Core': Scratch 'D1_S1' size (5.50µm) exceeds limit (3.0µm)"],
        "total_defect_count": 2,
        "characterized_defects": [
            {
                "defect_id": "D1_S1", "zone": "Core", "classification": "Scratch", "confidence_score": 0.95,
                "centroid_x_px": 190, "centroid_y_px": 140, "area_px": 50, "length_px": 22, "width_px": 2.5,
                "aspect_ratio_oriented": 8.8, "bbox_x_px": 180, "bbox_y_px": 130, "bbox_w_px": 15, "bbox_h_px": 25,
                "rotated_rect_center_px": (190.0, 140.0), "rotated_rect_angle_deg": 45.0,
                "contour_points_px": [[180,130],[195,130],[195,155],[180,155]], # Simplified
                "area_um2": 12.5, "length_um": 11.0, "width_um": 1.25
            },
            {
                "defect_id": "D1_P1", "zone": "Cladding", "classification": "Pit/Dig", "confidence_score": 0.88,
                "centroid_x_px": 230, "centroid_y_px": 170, "area_px": 75, "length_px": 10, "width_px": 9, # length/width for Pit/Dig are from minAreaRect
                "aspect_ratio_oriented": 1.1, "bbox_x_px": 225, "bbox_y_px": 165, "bbox_w_px": 10, "bbox_h_px": 10,
                "rotated_rect_center_px": (230.0, 170.0), "rotated_rect_angle_deg": 0.0,
                "contour_points_px": [[225,165],[235,165],[235,175],[225,175]], # Simplified
                "area_um2": 18.75, "effective_diameter_um": 4.89, "length_um":5.0, "width_um":4.5
            }
        ]
    }
    dummy_localization_data = { # Dummy localization data.
        "cladding_center_xy": (200, 150),
        "cladding_radius_px": 80.0,
        "core_center_xy": (200, 150),
        "core_radius_px": 30.0
    }
    dummy_fiber_type = "single_mode_pc" # Dummy fiber type.
    # Create dummy zone masks for histogram
    dummy_zone_masks_hist = {} # Initialize dummy zone masks for histogram.
    _h, _w = dummy_original_bgr.shape[:2] # Get image height and width.
    _Y, _X = np.ogrid[:_h, :_w] # Create coordinate grids.
    _center_x, _center_y = dummy_localization_data["cladding_center_xy"] # Get center.
    _dist_sq = (_X - _center_x)**2 + (_Y - _center_y)**2 # Calculate squared distance.
    
    # Create some representative zone masks based on localization
    _core_r_px = dummy_localization_data["core_radius_px"] # Get core radius.
    _clad_r_px = dummy_localization_data["cladding_radius_px"] # Get cladding radius.
    dummy_zone_masks_hist["Core"] = (_dist_sq < _core_r_px**2).astype(np.uint8) * 255 # Core mask.
    dummy_zone_masks_hist["Cladding"] = ((_dist_sq >= _core_r_px**2) & (_dist_sq < _clad_r_px**2)).astype(np.uint8) * 255 # Cladding mask.
    dummy_zone_masks_hist["Contact"] = (_dist_sq >= _clad_r_px**2).astype(np.uint8) * 255 # Contact mask (everything outside cladding).


    test_output_dir = Path("test_reporting_output") # Define test output directory.
    test_output_dir.mkdir(exist_ok=True) # Create directory if it doesn't exist.

    # --- Test Case 1: Generate Annotated Image ---
    logging.info("\n--- Test Case 1: Generate Annotated Image ---")
    annotated_img_path = test_output_dir / f"{Path(dummy_image_path).stem}_annotated.png" # Define path.
    success_annotated = generate_annotated_image( # Generate annotated image.
        dummy_original_bgr, dummy_analysis_results, dummy_localization_data, dummy_zone_masks_hist, dummy_fiber_type, annotated_img_path
    )
    logging.info(f"Annotated image generation success: {success_annotated}")

    # --- Test Case 2: Generate CSV Report ---
    logging.info("\n--- Test Case 2: Generate CSV Report ---")
    csv_report_path = test_output_dir / f"{Path(dummy_image_path).stem}_report.csv" # Define path.
    success_csv = generate_defect_csv_report(dummy_analysis_results, csv_report_path) # Generate CSV.
    logging.info(f"CSV report generation success: {success_csv}")

    # --- Test Case 3: Generate Polar Defect Histogram ---
    logging.info("\n--- Test Case 3: Generate Polar Defect Histogram ---")
    histogram_path = test_output_dir / f"{Path(dummy_image_path).stem}_histogram.png" # Define path.
    success_hist = generate_polar_defect_histogram( # Generate histogram.
        dummy_analysis_results, dummy_localization_data, dummy_zone_masks_hist, dummy_fiber_type, histogram_path
    )
    logging.info(f"Polar histogram generation success: {success_hist}")
    
    # Test with no defects
    logging.info("\n--- Test Case 4: Reporting with NO defects ---")
    dummy_analysis_no_defects = dummy_analysis_results.copy()
    dummy_analysis_no_defects["characterized_defects"] = []
    dummy_analysis_no_defects["total_defect_count"] = 0
    dummy_analysis_no_defects["overall_status"] = "PASS"
    dummy_analysis_no_defects["failure_reasons"] = []

    no_defect_csv_path = test_output_dir / f"{Path(dummy_image_path).stem}_no_defects_report.csv"
    generate_defect_csv_report(dummy_analysis_no_defects, no_defect_csv_path)

    no_defect_hist_path = test_output_dir / f"{Path(dummy_image_path).stem}_no_defects_histogram.png"
    generate_polar_defect_histogram(dummy_analysis_no_defects, dummy_localization_data, dummy_zone_masks_hist, dummy_fiber_type, no_defect_hist_path)


    # Clean up dummy image if it was created by this test script
    if Path(dummy_image_path).exists() and dummy_image_path == "test_report_image.png":
        Path(dummy_image_path).unlink()
        logging.info(f"Cleaned up dummy image: {dummy_image_path}")
