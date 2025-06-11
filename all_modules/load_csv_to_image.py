#!/usr/bin/env python3
"""Load CSV data back to image using only numpy"""

import cv2
import numpy as np

# Read CSV manually
max_x = 0
max_y = 0
data_points = []

with open('intensity_data.csv', 'r') as f:
    next(f)  # Skip header
    for line in f:
        x, y, intensity = line.strip().split(',')
        x, y, intensity = int(x), int(y), int(intensity)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        data_points.append((x, y, intensity))

# Create image matrix
img = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)

# Fill matrix with intensity values
for x, y, intensity in data_points:
    img[y, x] = intensity

# Save reconstructed image
cv2.imwrite('reconstructed_from_csv.png', img)

print(f"Reconstructed image of size {img.shape} from CSV")