#!/usr/bin/env python3
"""Save intensity matrix as CSV with x,y coordinates"""

import cv2
import numpy as np

# Image path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load image as grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Failed to load image")
else:
    height, width = img.shape
    
    # Create CSV with header
    with open("intensity_with_coords.csv", "w") as f:
        # Write header
        f.write("x,y,intensity\n")
        
        # Write data
        for y in range(height):
            for x in range(width):
                f.write(f"{x},{y},{img[y, x]}\n")
    
    print(f"Saved intensity matrix with coordinates to intensity_with_coords.csv")
    print(f"Total entries: {height * width}")