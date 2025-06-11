#!/usr/bin/env python3
"""Load a saved intensity matrix and perform operations"""

import cv2
import numpy as np

# Load previously saved intensity matrix
try:
    intensity_matrix = np.load("intensity_matrix.npy")
    print(f"Loaded intensity matrix: {intensity_matrix.shape}")
    
    # Apply threshold
    _, binary = cv2.threshold(intensity_matrix, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("binary_threshold.jpg", binary)
    print("Created binary threshold image")
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(intensity_matrix, (5, 5), 0)
    cv2.imwrite("blurred_intensity.jpg", blurred)
    print("Created blurred intensity image")
    
    # Detect edges using Canny
    edges = cv2.Canny(intensity_matrix, 50, 150)
    cv2.imwrite("edges.jpg", edges)
    print("Created edge detection image")
    
except FileNotFoundError:
    print("intensity_matrix.npy not found. Run script 8 first to create it.")