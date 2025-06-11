#!/usr/bin/env python3
"""
Fiber Locator Module
Detects fiber cladding and core using Hough circles and intensity analysis
"""
import cv2
import numpy as np


def find_cladding(image):
    """
    Find fiber cladding using HoughCircles
    Returns: (center_x, center_y), radius
    """
    h, w = image.shape[:2]
    min_radius = int(min(h, w) * 0.1)
    max_radius = int(min(h, w) * 0.45)
    
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(h, w) * 0.15),
        param1=70,
        param2=35,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if len(circles) > 0:
            x, y, r = circles[0]  # Take the first (strongest) circle
            return (x, y), r
    
    # Fallback: Use contours
    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5:
            (x, y), radius = cv2.minEnclosingCircle(largest)
            return (int(x), int(y)), int(radius)
    
    return None, None


def find_core(image, cladding_center, cladding_radius):
    """
    Find fiber core using intensity analysis
    Returns: (center_x, center_y), radius
    """
    if cladding_center is None:
        return None, None
    
    cx, cy = cladding_center
    
    # Create mask for core search area (inner 30% of cladding)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    search_radius = int(cladding_radius * 0.3)
    cv2.circle(mask, (cx, cy), search_radius, 255, -1)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 1)
    
    # Intensity-based detection: analyze radial profile
    max_radius = search_radius
    best_radius = 0
    max_gradient = 0
    
    for r in range(3, max_radius, 2):
        # Create ring mask
        ring_mask = np.zeros_like(mask)
        cv2.circle(ring_mask, (cx, cy), r + 2, 255, -1)
        cv2.circle(ring_mask, (cx, cy), r, 0, -1)
        
        # Calculate mean intensity in ring
        ring_pixels = blurred[ring_mask > 0]
        if len(ring_pixels) > 10:
            inner_mask = np.zeros_like(mask)
            cv2.circle(inner_mask, (cx, cy), r, 255, -1)
            inner_pixels = blurred[inner_mask > 0]
            
            if len(inner_pixels) > 10:
                # Gradient between inner and ring
                gradient = abs(np.mean(inner_pixels) - np.mean(ring_pixels))
                if gradient > max_gradient:
                    max_gradient = gradient
                    best_radius = r
    
    # Validate detected radius
    if best_radius > 3 and best_radius < cladding_radius * 0.15:
        return cladding_center, best_radius
    
    # Fallback: Use typical ratio (9µm core in 125µm cladding)
    fallback_radius = int(cladding_radius * 0.072)
    return cladding_center, fallback_radius


def locate_fiber(image):
    """
    Complete fiber localization
    Returns: dict with cladding and core parameters
    """
    cladding_center, cladding_radius = find_cladding(image)
    
    if cladding_center is None:
        return None
    
    core_center, core_radius = find_core(image, cladding_center, cladding_radius)
    
    return {
        'cladding_center': cladding_center,
        'cladding_radius': cladding_radius,
        'core_center': core_center if core_center else cladding_center,
        'core_radius': core_radius if core_radius else int(cladding_radius * 0.072)
    }


# Test function
if __name__ == "__main__":
    # Create test image
    test_img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(test_img, (100, 100), 80, 200, -1)  # Cladding
    cv2.circle(test_img, (100, 100), 8, 100, -1)   # Core
    
    # Test localization
    result = locate_fiber(test_img)
    if result:
        print(f"Cladding: center={result['cladding_center']}, radius={result['cladding_radius']}")
        print(f"Core: center={result['core_center']}, radius={result['core_radius']}")
    else:
        print("Failed to locate fiber")
