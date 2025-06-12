#!/usr/bin/env python3
"""
Validate components based on contrast threshold
"""
import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create defect mask (threshold dark regions)
_, defect_mask = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)

# Parameters
min_contrast = 15
min_area = 3

# Find connected components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)

# Create validated mask
validated_mask = np.zeros_like(defect_mask)

print(f"Found {num_labels - 1} components")

for i in range(1, num_labels):
    # Get component
    component_mask = (labels == i).astype(np.uint8) * 255
    area = stats[i, cv2.CC_STAT_AREA]
    
    # Skip small components
    if area < min_area:
        print(f"Component {i}: Rejected - too small ({area} pixels)")
        continue
    
    # Calculate contrast
    defect_pixels = image[component_mask > 0]
    
    # Dilate to get surrounding
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(component_mask, kernel)
    surround_mask = cv2.subtract(dilated, component_mask)
    surround_pixels = image[surround_mask > 0]
    
    if len(surround_pixels) > 0:
        contrast = abs(np.mean(defect_pixels) - np.mean(surround_pixels))
        
        # Validate based on contrast
        if contrast >= min_contrast:
            validated_mask = cv2.bitwise_or(validated_mask, component_mask)
            print(f"Component {i}: Validated - contrast={contrast:.1f}, area={area}")
        else:
            print(f"Component {i}: Rejected - low contrast ({contrast:.1f})")

# Visualize results
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
result[defect_mask > 0] = [0, 255, 255]  # Yellow for all detected
result[validated_mask > 0] = [0, 0, 255]  # Red for validated

cv2.imshow('Original', image)
cv2.imshow('All Detected', defect_mask)
cv2.imshow('Validated Only', validated_mask)
cv2.imshow('Comparison', result)
cv2.waitKey(0)
cv2.destroyAllWindows()