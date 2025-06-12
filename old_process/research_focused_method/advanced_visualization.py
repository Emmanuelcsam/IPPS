#!/usr/bin/env python3
# advanced_visualization.py

"""
Advanced Visualization Module
===========================================
Provides interactive visualization using Napari for detailed inspection.
"""
# Import numpy for numerical array operations - essential for image data manipulation
import numpy as np
# Import logging module for tracking events and debugging during visualization
import logging
# Import OpenCV (cv2) for image processing operations like drawing contours
import cv2
# Import type hints for better code documentation and IDE support
from typing import Dict, Any, List, Optional

# Try to import napari with error handling since it's an optional dependency
try:
    # Import napari - an interactive multi-dimensional image viewer for Python
    import napari
    # Set flag to True indicating napari is successfully imported and available
    NAPARI_AVAILABLE = True
# Handle the case where napari is not installed
except ImportError:
    # Set flag to False when napari import fails
    NAPARI_AVAILABLE = False
    # Log a warning message to inform user that interactive visualization is disabled
    logging.warning("Napari not available. Interactive visualization disabled.")


# Define the main visualization class that handles interactive display of inspection results
class InteractiveVisualizer:
    """
    Interactive visualization of fiber inspection results using Napari.
    """
    
    # Initialize the visualizer class
    def __init__(self):
        """Initialize the visualizer."""
        # Set viewer attribute to None - will hold the napari viewer instance when created
        self.viewer = None
        
    # Main method to display inspection results in an interactive viewer
    def show_inspection_results(self, 
                              original_image: np.ndarray,  # The original fiber optic image
                              defect_masks: Dict[str, np.ndarray],  # Dictionary of defect masks by type
                              zone_masks: Dict[str, np.ndarray],  # Dictionary of zone masks (core, cladding, etc.)
                              analysis_results: Dict[str, Any],  # Analysis results including defect characterizations
                              interactive: bool = True) -> Optional[Any]:  # Whether to run interactively
        """
        Display inspection results in an interactive Napari viewer.
        """
        # Check if napari is available before proceeding
        if not NAPARI_AVAILABLE:
            # Log warning and return None if napari is not available
            logging.warning("Napari not available. Skipping interactive visualization.")
            return None
            
        # Wrap visualization in try-except to handle any napari-related errors gracefully
        try:
            # Create viewer
            # Initialize a new napari viewer window with a descriptive title
            self.viewer = napari.Viewer(title=' Inspection Results')
            
            # Add original image
            # Add the original fiber optic image as the base layer in the viewer
            self.viewer.add_image(original_image, name='Original Image')
            
            # Add zone masks with different colors
            # Define color mapping for different fiber zones for visual distinction
            zone_colors = {
                'Core': 'red',  # Core zone will be displayed in red
                'Cladding': 'green',  # Cladding zone will be displayed in green
                'Adhesive': 'yellow',  # Adhesive zone will be displayed in yellow
                'Contact': 'magenta'  # Contact zone will be displayed in magenta
            }
            
            # Iterate through each zone mask and add it to the viewer
            for zone_name, mask in zone_masks.items():
                # Only add zone masks that contain actual data (non-zero values)
                if np.any(mask):
                    # Add zone mask as a labels layer with transparency and appropriate color
                    self.viewer.add_labels(
                        mask.astype(int),  # Convert mask to integer type for labels layer
                        name=f'Zone: {zone_name}',  # Descriptive name for the layer
                        opacity=0.3,  # Set transparency to 30% so underlying image is visible
                        color={1: zone_colors.get(zone_name, 'gray')}  # Map label value 1 to zone color
                    )
            
            # Add defect masks
            # --- Start of Edit 1: Robust Initialization of all_defects ---
            # Determine the 2D shape of the image regardless of whether it's grayscale or color
            if original_image.ndim == 3:  # Color image with 3 dimensions
                shape_2d = original_image.shape[:2]  # Extract height and width, ignoring color channels
            elif original_image.ndim == 2:  # Grayscale image with 2 dimensions
                shape_2d = original_image.shape  # Use shape as-is
            else:  # Handle unexpected image dimensions
                # Log error for unsupported image dimensions
                logging.error(f"Original image has unsupported dimension: {original_image.ndim}")
                return None  # Exit early if image format is not supported
            
            # Initialize empty defect mask with correct 2D shape to accumulate all defects
            all_defects = np.zeros(shape_2d, dtype=int)
            # --- End of Edit 1 ---
            # Initialize counter for unique defect IDs (renamed to avoid variable name conflict)
            defect_id_counter = 1
            
            # Process each characterized defect from the analysis results
            for defect in analysis_results.get('characterized_defects', []):
                # Create a mask for this specific defect
                # Initialize empty mask with same shape as all_defects
                defect_mask = np.zeros_like(all_defects)
                
                # Use contour points if available
                # Check if defect has contour points data
                if 'contour_points_px' in defect:
                    # Ensure contour points are np.int32 and pass color as a tuple
                    # Convert contour points to proper numpy array with int32 type for cv2.fillPoly
                    contour_points = np.array(defect['contour_points_px']).astype(np.int32)
                    # Fill the polygon defined by contour points with the current defect ID
                    cv2.fillPoly(defect_mask, [contour_points], (defect_id_counter,)) 
                    # Copy defect mask values to the all_defects accumulator mask
                    all_defects[defect_mask > 0] = defect_id_counter
                    # Increment counter for next defect to have unique ID
                    defect_id_counter += 1
            
            # Add the accumulated defects mask to viewer if any defects were found
            if np.any(all_defects):
                # Add defects as a labels layer with higher opacity for visibility
                self.viewer.add_labels(
                    all_defects,  # The accumulated defect mask with unique IDs
                    name='Detected Defects',  # Layer name
                    opacity=0.7  # 70% opacity to make defects clearly visible
                )

            # Add text annotations for defects
            # Iterate through each defect to add ID labels at their centers
            for defect in analysis_results.get('characterized_defects', []):
                # Extract defect centroid coordinates (default to 0 if not found)
                cx, cy = defect.get('centroid_x_px', 0), defect.get('centroid_y_px', 0)
                # Get defect ID string (use distinct variable name to avoid conflicts)
                defect_label_id = defect.get('defect_id', '')
                # Add text annotation at defect centroid
                self.viewer.add_text(
                    text=f"ID: {defect_label_id}",  # Display defect ID
                    position=(cy, cx),  # Position in (row, column) format for Napari
                    face_color='yellow',  # Yellow text color for visibility
                    size=12,  # Font size in points
                    anchor='center',  # Center the text on the position
                    name=f"Defect {defect_label_id}"  # Unique layer name for this annotation
                )

                        
            # Add text annotations (Napari >=0.4)
            # Extract overall pass/fail status from analysis results
            status = analysis_results.get('overall_status', 'UNKNOWN')
            # Set color based on status - green for PASS, red for FAIL
            status_color = 'green' if status == 'PASS' else 'red'

            # Place status text at top-left corner (10,10)
            # Add overall status annotation to the viewer
            self.viewer.add_text(
                text=f"Status: {status}",  # Display pass/fail status
                position=(10, 10),  # Position at top-left corner in (row, column) format
                face_color=status_color,  # Color matches pass/fail status
                size=20,  # Larger font size for prominence
                anchor='upper_left',  # Anchor to upper left of position
                name='Overall Status'  # Layer name for status text
            )
            
            # Run napari event loop if interactive mode is enabled
            if interactive:
                napari.run()  # Start the napari GUI event loop for user interaction
            
            # Return the viewer instance for potential further manipulation
            return self.viewer
            
        # Catch any exceptions that occur during visualization
        except Exception as e:
            # Log the error with details about what went wrong
            logging.error(f"Napari visualization failed: {e}")
            # Optionally, re-raise or handle more gracefully
            # if self.viewer: self.viewer.close() # Close viewer if it was partially created
            return None  # Return None to indicate visualization failure
    
    # Method to close the viewer and clean up resources
    def close(self):
        """Close the viewer."""
        # Check if viewer exists before attempting to close
        if self.viewer:
            # Call napari's close method to shut down the viewer window
            self.viewer.close()
            # Reset viewer reference to None to indicate it's closed
            self.viewer = None