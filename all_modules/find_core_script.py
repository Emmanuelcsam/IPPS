#!/usr/bin/env python3
"""Find fiber core using intensity analysis"""
import cv2
import numpy as np

# File path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# First find cladding (needed as reference)
blurred = cv2.GaussianBlur(gray, (5, 5), 1)
h, w = gray.shape
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=int(min(h, w) * 0.15),
    param1=70,
    param2=35,
    minRadius=int(min(h, w) * 0.1),
    maxRadius=int(min(h, w) * 0.45)
)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    cx, cy, cladding_r = circles[0]
    
    # Find core using intensity gradient analysis
    search_radius = int(cladding_r * 0.3)
    best_radius = 0
    max_gradient = 0
    
    for r in range(3, search_radius, 2):
        # Create masks
        ring_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(ring_mask, (cx, cy), r + 2, 255, -1)
        cv2.circle(ring_mask, (cx, cy), r - 1, 0, -1)
        
        inner_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(inner_mask, (cx, cy), r, 255, -1)
        
        # Calculate gradient
        ring_pixels = blurred[ring_mask > 0]
        inner_pixels = blurred[inner_mask > 0]
        
        if len(ring_pixels) > 10 and len(inner_pixels) > 10:
            gradient = abs(np.mean(inner_pixels) - np.mean(ring_pixels))
            if gradient > max_gradient:
                max_gradient = gradient
                best_radius = r
    
    # Use detected or fallback radius
    core_radius = best_radius if best_radius > 3 else int(cladding_r * 0.072)
    print(f"Core found: center=({cx}, {cy}), radius={core_radius}")
    
    # Draw and save result
    output = image.copy()
    cv2.circle(output, (cx, cy), core_radius, (0, 0, 255), 2)
    cv2.imwrite("core_detected.jpg", output)
else:
    print("No fiber detected")