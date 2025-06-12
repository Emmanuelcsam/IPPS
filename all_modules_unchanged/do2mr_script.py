#!/usr/bin/env python3
"""DO2MR (Difference of Opening and Max Residual) Detection"""
import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create zone mask (full image)
zone_mask = np.ones_like(image) * 255

# DO2MR parameters
gamma = 1.5
kernel_size = 5

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
median_val = np.median(zone_pixels)
mad = np.median(np.abs(zone_pixels - median_val))
std_robust = 1.4826 * mad
threshold = float(median_val + gamma * std_robust)

# Apply threshold
_, defect_mask = cv2.threshold(residual, threshold, 255, cv2.THRESH_BINARY)

# Apply zone mask to result
defect_mask = cv2.bitwise_and(defect_mask, zone_mask)

# Morphological cleanup
kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel_clean)

# Display result
cv2.imshow('Original', image)
cv2.imshow('DO2MR Defects', defect_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite('do2mr_result.png', defect_mask)