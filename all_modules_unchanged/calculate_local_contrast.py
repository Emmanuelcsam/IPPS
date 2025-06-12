#!/usr/bin/env python3
"""
Calculate local contrast between a region and its surroundings
"""
import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create a sample defect mask (threshold dark regions)
_, defect_mask = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)

# Calculate contrast for first detected component
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)

if num_labels > 1:
    # Get first component
    component_mask = (labels == 1).astype(np.uint8) * 255
    
    # Get defect pixels
    defect_pixels = image[component_mask > 0]
    
    # Create surrounding region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(component_mask, kernel)
    surround_mask = cv2.subtract(dilated, component_mask)
    
    # Get surrounding pixels
    surround_pixels = image[surround_mask > 0]
    
    # Calculate contrast
    defect_mean = np.mean(defect_pixels)
    surround_mean = np.mean(surround_pixels)
    contrast = abs(defect_mean - surround_mean)
    
    print(f"Defect mean intensity: {defect_mean:.2f}")
    print(f"Surrounding mean intensity: {surround_mean:.2f}")
    print(f"Local contrast: {contrast:.2f}")
    
    # Visualize
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result[component_mask > 0] = [0, 0, 255]  # Red for defect
    result[surround_mask > 0] = [0, 255, 0]  # Green for surrounding
    
    cv2.imshow('Local Contrast Analysis', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No components found")