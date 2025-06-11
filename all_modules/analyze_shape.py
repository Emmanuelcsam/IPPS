#!/usr/bin/env python3
"""
Analyze shape properties of detected components
"""
import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create binary mask (threshold or edge detection)
_, binary = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Analyze each contour
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    
    # Skip very small contours
    if area < 10:
        continue
    
    # Calculate shape properties
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0
    
    # Circularity (1.0 for perfect circle)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Solidity (ratio of area to convex hull area)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Check if linear
    is_linear = aspect_ratio > 3.0 or circularity < 0.3
    
    print(f"\nContour {i}:")
    print(f"  Area: {area:.0f} pixels")
    print(f"  Aspect ratio: {aspect_ratio:.2f}")
    print(f"  Circularity: {circularity:.2f}")
    print(f"  Solidity: {solidity:.2f}")
    print(f"  Is linear: {is_linear}")
    
    # Draw contour on image
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    color = (0, 0, 255) if is_linear else (0, 255, 0)
    cv2.drawContours(result, [contour], -1, color, 2)
    cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 1)
    
    cv2.imshow(f'Shape Analysis - Contour {i}', result)
    cv2.waitKey(0)

cv2.destroyAllWindows()