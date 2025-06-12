#!/usr/bin/env python3
"""Matrix Variance Detection for Local Anomalies"""
import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create zone mask (full image)
zone_mask = np.ones_like(image) * 255

# Matrix variance parameters
window_size = 3
z_threshold = 2.0

# Initialize result
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

# Display result
cv2.imshow('Original', image)
cv2.imshow('Matrix Variance Anomalies', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite('matrix_variance_result.png', result)