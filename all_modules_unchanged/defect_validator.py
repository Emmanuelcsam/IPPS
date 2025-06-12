#!/usr/bin/env python3
"""
Defect Validator Module
Validates detected defects based on contrast and shape analysis
"""
import cv2
import numpy as np


def calculate_local_contrast(image, defect_mask, dilation_size=5):
    """
    Calculate contrast between defect and surrounding region
    
    Args:
        image: original grayscale image
        defect_mask: binary mask of detected defect
        dilation_size: size for surrounding region
    
    Returns:
        contrast: contrast value
        defect_mean: mean intensity of defect
        surround_mean: mean intensity of surrounding
    """
    # Get defect pixels
    defect_pixels = image[defect_mask > 0]
    if len(defect_pixels) == 0:
        return 0, 0, 0
    
    # Create surrounding region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                      (dilation_size, dilation_size))
    dilated = cv2.dilate(defect_mask, kernel)
    surround_mask = cv2.subtract(dilated, defect_mask)
    
    # Get surrounding pixels
    surround_pixels = image[surround_mask > 0]
    if len(surround_pixels) == 0:
        return 0, np.mean(defect_pixels), 0
    
    defect_mean = np.mean(defect_pixels)
    surround_mean = np.mean(surround_pixels)
    contrast = abs(defect_mean - surround_mean)
    
    return contrast, defect_mean, surround_mean


def analyze_shape(component_mask):
    """
    Analyze shape properties of a defect component
    
    Args:
        component_mask: binary mask of single component
    
    Returns:
        dict with shape properties
    """
    # Find contour
    contours, _ = cv2.findContours(component_mask, 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    contour = contours[0]
    area = cv2.contourArea(contour)
    
    if area < 1:
        return None
    
    # Calculate properties
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0
    
    # Circularity (1.0 for perfect circle)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Solidity (ratio of area to convex hull area)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    return {
        'area': area,
        'perimeter': perimeter,
        'aspect_ratio': aspect_ratio,
        'circularity': circularity,
        'solidity': solidity,
        'is_linear': aspect_ratio > 3.0 or circularity < 0.3
    }


def validate_defects(defect_mask, original_image, zone_mask, 
                    zone_name="", min_contrast=10, min_area=3):
    """
    Validate detected defects
    
    Args:
        defect_mask: binary mask of detected defects
        original_image: original grayscale image
        zone_mask: binary mask of the zone
        zone_name: name of the zone
        min_contrast: minimum contrast threshold
        min_area: minimum area in pixels
    
    Returns:
        validated_mask: validated defect mask
    """
    validated_mask = np.zeros_like(defect_mask)
    
    # Zone-specific parameters
    if zone_name.lower() == 'core':
        min_contrast *= 1.5  # Stricter for core
        contrast_weight = 0.7
        shape_weight = 0.3
    else:
        contrast_weight = 0.5
        shape_weight = 0.5
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        defect_mask, connectivity=8)
    
    for i in range(1, num_labels):
        # Get component mask
        component_mask = (labels == i).astype(np.uint8) * 255
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Skip very small components
        if area < min_area:
            continue
        
        # Calculate contrast
        contrast, defect_mean, surround_mean = calculate_local_contrast(
            original_image, component_mask)
        
        # Analyze shape
        shape_props = analyze_shape(component_mask)
        if shape_props is None:
            continue
        
        # Validation score
        contrast_score = min(contrast / min_contrast, 2.0)  # Cap at 2.0
        
        # Shape score (favor elongated for scratches, compact for spots)
        if shape_props['is_linear']:
            shape_score = 1.5  # Bonus for linear defects
        else:
            shape_score = shape_props['circularity']
        
        # Combined score
        total_score = (contrast_weight * contrast_score + 
                      shape_weight * shape_score)
        
        # Validate
        if total_score >= 1.0 or (shape_props['is_linear'] and contrast >= min_contrast * 0.8):
            validated_mask = cv2.bitwise_or(validated_mask, component_mask)
    
    return validated_mask


def remove_boundary_defects(defect_mask, fiber_params, boundary_width=3):
    """
    Remove defects on zone boundaries
    
    Args:
        defect_mask: binary defect mask
        fiber_params: fiber localization parameters
        boundary_width: width of boundary exclusion zone
    
    Returns:
        cleaned_mask: defect mask with boundary defects removed
    """
    if fiber_params is None:
        return defect_mask
    
    h, w = defect_mask.shape
    
    # Import zone_generator for boundary exclusion mask
    try:
        from zone_generator import create_boundary_exclusion_mask
        
        exclusion_mask = create_boundary_exclusion_mask(
            defect_mask.shape, fiber_params, boundary_width)
        
        # Remove defects in exclusion zone
        cleaned_mask = cv2.bitwise_and(defect_mask, 
                                      cv2.bitwise_not(exclusion_mask))
    except ImportError:
        print("Warning: zone_generator module not found, skipping boundary removal")
        cleaned_mask = defect_mask
    
    return cleaned_mask


if __name__ == "__main__":
    # Create test image with contrast variations
    test_img = np.ones((100, 100), dtype=np.uint8) * 128
    
    # Add high contrast defect - fixed: removed tuple wrapper
    cv2.circle(test_img, (30, 30), 5, (200,), -1)

    # Add low contrast defect - fixed: removed tuple wrapper
    cv2.circle(test_img, (70, 70), 5, (140,), -1)

    # Create test defect mask - fixed: removed tuple wrapper
    defect_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(defect_mask, (30, 30), 5, (255,), -1)
    cv2.circle(defect_mask, (70, 70), 5, (255,), -1)
    
    # Create zone mask
    zone_mask = np.ones((100, 100), dtype=np.uint8) * 255
    
    # Validate
    validated = validate_defects(defect_mask, test_img, zone_mask, 
                               min_contrast=15)
    
    print(f"Original defect pixels: {np.sum(defect_mask > 0)}")
    print(f"Validated defect pixels: {np.sum(validated > 0)}")
