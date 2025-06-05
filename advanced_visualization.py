#!/usr/bin/env python3
# advanced_visualization.py

"""
D-Scope Blink: Advanced Visualization Module
===========================================
Provides interactive visualization using Napari for detailed inspection.
"""
import numpy as np
import logging
import cv2
from typing import Dict, Any, List, Optional

try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False
    logging.warning("Napari not available. Interactive visualization disabled.")


class InteractiveVisualizer:
    """
    Interactive visualization of fiber inspection results using Napari.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.viewer = None
        
    def show_inspection_results(self, 
                              original_image: np.ndarray,
                              defect_masks: Dict[str, np.ndarray],
                              zone_masks: Dict[str, np.ndarray],
                              analysis_results: Dict[str, Any],
                              interactive: bool = True) -> Optional[Any]:
        """
        Display inspection results in an interactive Napari viewer.
        """
        if not NAPARI_AVAILABLE:
            logging.warning("Napari not available. Skipping interactive visualization.")
            return None
            
        try:
            # Create viewer
            self.viewer = napari.Viewer(title='D-Scope Blink - Inspection Results')
            
            # Add original image
            self.viewer.add_image(original_image, name='Original Image')
            
            # Add zone masks with different colors
            zone_colors = {
                'Core': 'red',
                'Cladding': 'green',
                'Adhesive': 'yellow',
                'Contact': 'magenta'
            }
            
            for zone_name, mask in zone_masks.items():
                if np.any(mask):
                    self.viewer.add_labels(
                        mask.astype(int),
                        name=f'Zone: {zone_name}',
                        opacity=0.3,
                        color={1: zone_colors.get(zone_name, 'gray')}
                    )
            
            # Add defect masks
            # --- Start of Edit 1: Robust Initialization of all_defects ---
            if original_image.ndim == 3:
                shape_2d = original_image.shape[:2]
            elif original_image.ndim == 2:
                shape_2d = original_image.shape
            else:
                logging.error(f"Original image has unsupported dimension: {original_image.ndim}")
                return None
            
            all_defects = np.zeros(shape_2d, dtype=int)
            # --- End of Edit 1 ---
            defect_id_counter = 1 # Renamed to avoid conflict with defect_id from analysis_results
            
            for defect in analysis_results.get('characterized_defects', []):
                # Create a mask for this specific defect
                defect_mask = np.zeros_like(all_defects) # Uses the correctly shaped all_defects
                
                # Use contour points if available
                if 'contour_points_px' in defect:
                    # Ensure contour points are np.int32 and pass color as a tuple [cite: 147, 148]
                    contour_points = np.array(defect['contour_points_px']).astype(np.int32)
                    cv2.fillPoly(defect_mask, [contour_points], (defect_id_counter,)) 
                    all_defects[defect_mask > 0] = defect_id_counter
                    defect_id_counter += 1
            
            if np.any(all_defects):
                self.viewer.add_labels(
                    all_defects,
                    name='Detected Defects',
                    opacity=0.7
                )

            # --- Start of Edit 2: Correct Coordinate Order and Variable Naming for Defect ID Text ---
            # --- Start of Edit 2: Correct Coordinate Order and Variable Naming for Defect ID Text ---
            for defect in analysis_results.get('characterized_defects', []):
                cx, cy = defect.get('centroid_x_px', 0), defect.get('centroid_y_px', 0)
                defect_label_id = defect.get('defect_id', '') # Use a distinct variable name for the defect's string ID
                self.viewer.add_text(
                    text=f"ID: {defect_label_id}",
                    position=(cy, cx), # Corrected order: (row, column) or (y, x)
                    face_color='yellow', # Changed 'color' to 'face_color'
                    size=12,
                    anchor='center',
                    name=f"Defect {defect_label_id}"
                )

                        
            # Add text annotations (Napari >=0.4)
            status = analysis_results.get('overall_status', 'UNKNOWN')
            status_color = 'green' if status == 'PASS' else 'red'

            # Place status text at top-left corner (10,10)
            self.viewer.add_text(
                text=f"Status: {status}",
                position=(10, 10), # (row, column) for Napari
                face_color=status_color,
                size=20,                   # Font size
                anchor='upper_left',
                name='Overall Status'
            )
            
            if interactive:
                napari.run()
            
            return self.viewer
            
        except Exception as e:
            logging.error(f"Napari visualization failed: {e}")
            # Optionally, re-raise or handle more gracefully
            # if self.viewer: self.viewer.close() # Close viewer if it was partially created
            return None
    
    def close(self):
        """Close the viewer."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None # Ensure viewer is reset after closing