#!/usr/bin/env python3
"""Calculate statistics of intensity matrix"""

import cv2
import numpy as np

# Image path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load image as grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Failed to load image")
else:
    # Calculate statistics
    print("Intensity Matrix Statistics:")
    print(f"Shape: {img.shape}")
    print(f"Total pixels: {img.size}")
    print(f"Min intensity: {np.min(img)}")
    print(f"Max intensity: {np.max(img)}")
    print(f"Mean intensity: {np.mean(img):.2f}")
    print(f"Median intensity: {np.median(img):.2f}")
    print(f"Std deviation: {np.std(img):.2f}")
    print(f"Variance: {np.var(img):.2f}")
    
    # Percentiles
    print("\nPercentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th percentile: {np.percentile(img, p):.2f}")