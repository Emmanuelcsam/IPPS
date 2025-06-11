#!/usr/bin/env python3
"""
Zone Generator Module
Creates masks for fiber zones (core and cladding)
"""
import cv2
import numpy as np


def create_circular_mask(image_shape, center, radius):
    """Create a circular mask"""
    h, w = image_shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = center
    
    dist_sq = (X - cx)**2 + (Y - cy)**2
    mask = (dist_sq <= radius**2).astype(np.uint8) * 255
    
    return mask


def generate_zone_masks(image_shape, fiber_params, um_per_px=None):
    """
    Generate masks for core and cladding zones
    
    Args:
        image_shape: (height, width) of the image
        fiber_params: dict with fiber localization data
        um_per_px: optional micrometers per pixel for scaling
    
    Returns:
        dict with 'core' and 'cladding' masks
    """
    if fiber_params is None:
        return {}
    
    masks = {}
    
    # Get parameters
    cladding_center = fiber_params.get('cladding_center')
    cladding_radius = fiber_params.get('cladding_radius')
    core_center = fiber_params.get('core_center', cladding_center)
    core_radius = fiber_params.get('core_radius')
    
    if cladding_center is None or cladding_radius is None:
        return masks
    
    # Create core mask
    if core_center and core_radius:
        masks['core'] = create_circular_mask(image_shape, core_center, core_radius)
    
    # Create cladding mask (excluding core)
    cladding_full = create_circular_mask(image_shape, cladding_center, cladding_radius)
    
    if 'core' in masks:
        # Subtract core from cladding
        masks['cladding'] = cv2.subtract(cladding_full, masks['core'])
    else:
        masks['cladding'] = cladding_full
    
    return masks


def create_boundary_exclusion_mask(image_shape, fiber_params, boundary_width=3):
    """
    Create mask to exclude zone boundaries
    Useful for avoiding false defects at boundaries
    """
    h, w = image_shape[:2]
    exclusion_mask = np.zeros((h, w), dtype=np.uint8)
    
    if fiber_params is None:
        return exclusion_mask
    
    cladding_center = fiber_params.get('cladding_center')
    core_radius = fiber_params.get('core_radius', 0)
    cladding_radius = fiber_params.get('cladding_radius', 0)
    
    if cladding_center and core_radius > 0:
        cx, cy = cladding_center
        
        # Core-cladding boundary exclusion
        cv2.circle(exclusion_mask, (cx, cy), 
                  int(core_radius + boundary_width), 255, -1)
        cv2.circle(exclusion_mask, (cx, cy), 
                  int(max(0, core_radius - boundary_width)), 0, -1)
        
        # Cladding outer boundary exclusion
        if cladding_radius > 0:
            cv2.circle(exclusion_mask, (cx, cy), 
                      int(min(h//2, cladding_radius + boundary_width)), 255, -1)
            cv2.circle(exclusion_mask, (cx, cy), 
                      int(max(0, cladding_radius - boundary_width)), 0, -1)
    
    return exclusion_mask


# Test function
if __name__ == "__main__":
    # Test parameters
    test_shape = (200, 200)
    test_fiber_params = {
        'cladding_center': (100, 100),
        'cladding_radius': 80,
        'core_center': (100, 100),
        'core_radius': 8
    }
    
    # Generate masks
    masks = generate_zone_masks(test_shape, test_fiber_params)
    print(f"Generated masks: {list(masks.keys())}")
    
    # Test boundary exclusion
    exclusion = create_boundary_exclusion_mask(test_shape, test_fiber_params)
    print(f"Boundary exclusion mask shape: {exclusion.shape}")
    
    # Visualize
    if masks:
        combined = np.zeros(test_shape, dtype=np.uint8)
        if 'core' in masks:
            combined = cv2.add(combined, masks['core'] // 2)
        if 'cladding' in masks:
            combined = cv2.add(combined, masks['cladding'] // 4)
        cv2.imwrite("test_zones.png", combined)
        print("Saved visualization to test_zones.png")
        
        # Cleanup
        import os
        os.remove("test_zones.png")
