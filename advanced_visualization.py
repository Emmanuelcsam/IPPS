#!/usr/bin/env python3
# advanced_visualization.py

"""
D-Scope Blink: Advanced Visualization Module
===========================================
Provides interactive visualization using Napari for detailed inspection.
"""
import numpy as np
import logging
import cv2                              # <<<<<< Add this line
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
            all_defects = np.zeros_like(original_image[:,:,0])
            defect_id = 1
            
            for defect in analysis_results.get('characterized_defects', []):
                # Create a mask for this specific defect
                defect_mask = np.zeros_like(all_defects)
                
                # Use contour points if available
                if 'contour_points_px' in defect:
                    contour = np.array(defect['contour_points_px'])
                    cv2.fillPoly(defect_mask, [contour], defect_id)
                    all_defects[defect_mask > 0] = defect_id
                    defect_id += 1
            
            if np.any(all_defects):
                self.viewer.add_labels(
                    all_defects,
                    name='Detected Defects',
                    opacity=0.7
                )
            for defect in analysis_results.get('characterized_defects', []):
                cx, cy = defect.get('centroid_x_px', 0), defect.get('centroid_y_px', 0)
                defect_id = defect.get('defect_id', '')
                self.viewer.add_text(
                    text=f"ID: {defect_id}",
                    position=(cx, cy),
                    color='yellow',
                    size=12,
                    anchor='center',
                    name=f"Defect {defect_id}"
                )
                        
            # Add text annotations (Napari >=0.4)
            status = analysis_results.get('overall_status', 'UNKNOWN')
            status_color = 'green' if status == 'PASS' else 'red'

            # Place status text at top-left corner (10,10)
            self.viewer.add_text(
                text=f"Status: {status}",
                position=(10, 10),
                color=status_color,
                size=20,                   # Font size
                anchor='upper_left',
                name='Overall Status'
            )

            
            if interactive:
                napari.run()
            
            return self.viewer
            
        except Exception as e:
            logging.error(f"Napari visualization failed: {e}")
            return None
    
    def close(self):
        """Close the viewer."""
        if self.viewer:
            self.viewer.close()