#!/usr/bin/env python3
"""Median Filtering for Noise Reduction"""
import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Add some salt and pepper noise for demonstration
noisy = image.copy()
noise = np.random.random((image.shape[0], image.shape[1]))
noisy[noise < 0.05] = 0  # Salt noise
noisy[noise > 0.95] = 255  # Pepper noise

# Apply median filters with different kernel sizes
median_3 = cv2.medianBlur(noisy, 3)
median_5 = cv2.medianBlur(noisy, 5)
median_7 = cv2.medianBlur(noisy, 7)

# Also show Gaussian blur for comparison
gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Noisy', noisy)
cv2.imshow('Median Filter (3x3)', median_3)
cv2.imshow('Median Filter (5x5)', median_5)
cv2.imshow('Median Filter (7x7)', median_7)
cv2.imshow('Gaussian Blur (5x5)', gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save results
cv2.imwrite('noisy_image.png', noisy)
cv2.imwrite('median_filtered_3.png', median_3)
cv2.imwrite('median_filtered_5.png', median_5)
cv2.imwrite('gaussian_filtered.png', gaussian)