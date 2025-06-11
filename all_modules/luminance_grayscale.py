#!/usr/bin/env python3
"""Convert image to grayscale using luminance method"""

import cv2
import numpy as np

# Image path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load image
img = cv2.imread(image_path)

if img is None:
    print(f"Failed to load image")
else:
    # Check if already grayscale
    if len(img.shape) == 2:
        grayscale = img
    else:
        # Convert BGR to grayscale using luminance formula
        # OpenCV uses BGR format, so: 0.114*B + 0.587*G + 0.299*R
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        grayscale = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
    
    print(f"Grayscale shape: {grayscale.shape}")
    print(f"Intensity range: [{np.min(grayscale)}, {np.max(grayscale)}]")
    
    # Save result
    cv2.imwrite("grayscale_luminance.jpg", grayscale)