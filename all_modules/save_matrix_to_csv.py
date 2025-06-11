#!/usr/bin/env python3
"""Save intensity matrix to CSV format using only numpy"""

import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create CSV data with x, y, intensity
height, width = img.shape
with open('intensity_data.csv', 'w') as f:
    f.write('x,y,intensity\n')
    for y in range(height):
        for x in range(width):
            f.write(f'{x},{y},{img[y, x]}\n')

print(f"Saved {height * width} pixels to intensity_data.csv")