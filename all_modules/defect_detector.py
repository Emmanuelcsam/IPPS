#!/usr/bin/env python3
"""
Defect Detector Module
Implements DO2MR and LEI algorithms for defect detection
"""
import cv2
import numpy as np


def do2mr_detection(image, zone_mask, gamma=1.5, kernel_size=5):
    """
    Difference of Opening and Max Residual (DO2MR) detection
    
    Args:
        image: grayscale image
        zone_mask: binary mask for the zone
        gamma: sensitivity parameter
        kernel_size: morphological kernel size
    
    Returns:
        defect_mask: binary defect mask
    """
    # Apply zone mask
    masked = cv2.bitwise_and(image, image, mask=zone_mask)
    
    # Create morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Max and min filtering
    dilated = cv2.dilate(masked, kernel)
    eroded = cv2.erode(masked, kernel)
    
    # Calculate residual
    residual = cv2.subtract(dilated, eroded)
    
    # Apply median filter to reduce noise
    residual = cv2.medianBlur(residual, 3)
    
    # Calculate threshold using robust statistics
    zone_pixels = residual[zone_mask > 0].astype(np.float64)
    if len(zone_pixels) < 100:
        return np.zeros_like(image)
    
    # Use median and MAD for robustness
    median_val = np.median(zone_pixels)
    mad = np.median(np.abs(zone_pixels - median_val))
    std_robust = 1.4826 * mad  # Conversion factor
    
    # Adaptive threshold (cast to Python float)
    threshold = float(median_val + gamma * std_robust)
    
    # Apply threshold
    _, defect_mask = cv2.threshold(residual, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply zone mask to result
    defect_mask = cv2.bitwise_and(defect_mask, zone_mask)
    
    # Morphological cleanup
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel_clean)
    
    return defect_mask


def lei_scratch_detection(image, zone_mask, kernel_lengths=[7, 11, 15], angle_step=15):
    """
    Linear Element Inspection (LEI) for scratch detection
    
    Args:
        image: grayscale image
        zone_mask: binary mask for the zone
        kernel_lengths: list of line kernel lengths
        angle_step: angle step in degrees
    
    Returns:
        scratch_mask: binary scratch mask
    """
    # Apply zone mask
    masked = cv2.bitwise_and(image, image, mask=zone_mask)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked)
    
    # Initialize scratch map
    scratch_map = np.zeros_like(enhanced, dtype=np.float32)
    
    # Apply directional filters
    for angle in range(0, 180, angle_step):
        for kernel_length in kernel_lengths:
            # Create line kernel
            kernel = np.zeros((kernel_length, kernel_length), dtype=np.float32)
            center = kernel_length // 2
            kernel[center, :] = 1.0 / kernel_length
            
            # Rotate kernel
            M = cv2.getRotationMatrix2D((center, center), angle, 1)
            rotated_kernel = cv2.warpAffine(kernel, M, (kernel_length, kernel_length))
            
            # Apply filter
            response = cv2.filter2D(enhanced, cv2.CV_32F, rotated_kernel)
            
            # Update maximum response
            scratch_map = np.maximum(scratch_map, response)
    
    # Normalize (use scratch_map itself as the destination)
    scratch_map = cv2.normalize(scratch_map, scratch_map, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(scratch_map, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 15, -2)
    
    # Apply zone mask
    result = cv2.bitwise_and(binary, zone_mask)
    
    # Connect scratch fragments
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_connect)
    
    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        
        # Keep only linear structures
        if area < 10 or aspect_ratio < 2.5:
            result[labels == i] = 0
    
    return result


def matrix_variance_detection(image, zone_mask, window_size=3, z_threshold=2.0):
    """
    Matrix variance detection for local anomalies
    
    Args:
        image: grayscale image
        zone_mask: binary mask for the zone
        window_size: local window size
        z_threshold: z-score threshold
    
    Returns:
        anomaly_mask: binary anomaly mask
    """
    h, w = image.shape[:2]
    result = np.zeros((h, w), dtype=np.uint8)
    half_window = window_size // 2
    
    # Process each pixel
    for y in range(half_window, h - half_window):
        for x in range(half_window, w - half_window):
            if zone_mask[y, x] == 0:
                continue
            
            # Get local window
            window = image[y-half_window:y+half_window+1,
                          x-half_window:x+half_window+1]
            
            # Calculate statistics
            center_val = float(image[y, x])
            mean_val = np.mean(window)
            std_val = np.std(window)
            
            # Check for anomaly
            if std_val > 0:
                z_score = abs(center_val - mean_val) / std_val
                if z_score > z_threshold:
                    result[y, x] = 255
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    return result


def detect_defects(image, zone_mask, zone_name="", algorithms=['do2mr', 'lei']):
    """
    Combined defect detection using multiple algorithms
    
    Args:
        image: grayscale image
        zone_mask: binary mask for the zone
        zone_name: name of the zone (for parameter adjustment)
        algorithms: list of algorithms to use
    
    Returns:
        combined_mask: combined defect mask
    """
    combined_mask = np.zeros_like(image, dtype=np.uint8)
    
    if 'do2mr' in algorithms:
        # Adjust gamma based on zone
        gamma = 1.2 if zone_name.lower() == 'core' else 1.5
        do2mr_mask = do2mr_detection(image, zone_mask, gamma=gamma)
        combined_mask = cv2.bitwise_or(combined_mask, do2mr_mask)
    
    if 'lei' in algorithms:
        lei_mask = lei_scratch_detection(image, zone_mask)
        combined_mask = cv2.bitwise_or(combined_mask, lei_mask)
    
    if 'matrix' in algorithms:
        matrix_mask = matrix_variance_detection(image, zone_mask)
        combined_mask = cv2.bitwise_or(combined_mask, matrix_mask)
    
    return combined_mask


# Test function
if __name__ == "__main__":
    # Create test image with defects
    test_img = np.ones((200, 200), dtype=np.uint8) * 128
    
    # Add some defects
    cv2.circle(test_img, (50, 50), 3, (200,), -1)  # Bright spot
    cv2.line(test_img, (100, 20), (120, 180), (80,), 1)  # Scratch
    
    # Create test zone mask
    zone_mask = np.ones((200, 200), dtype=np.uint8) * 255
    
    # Test detection
    defects = detect_defects(test_img, zone_mask, algorithms=['do2mr', 'lei'])
    
    num_defect_pixels = np.sum(defects > 0)
    print(f"Detected {num_defect_pixels} defect pixels")
