#!/usr/bin/env python3
"""LEI (Linear Element Inspection) for Scratch Detection"""
import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create zone mask (full image)
zone_mask = np.ones_like(image) * 255

# LEI parameters
kernel_lengths = [7, 11, 15]
angle_step = 15

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

# Normalize
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

# Display result
cv2.imshow('Original', image)
cv2.imshow('LEI Scratches', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite('lei_result.png', result)