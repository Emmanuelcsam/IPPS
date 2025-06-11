#!/usr/bin/env python3
"""Convert image to grayscale using min channel value"""

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
        # Convert to grayscale using min value across channels
        grayscale = np.min(img, axis=2).astype(np.uint8)
    
    print(f"Grayscale shape: {grayscale.shape}")
    print(f"Intensity range: [{np.min(grayscale)}, {np.max(grayscale)}]")
    
    # Save result
    cv2.imwrite("grayscale_min.jpg", grayscale)