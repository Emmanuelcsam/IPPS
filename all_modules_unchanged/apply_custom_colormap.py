#!/usr/bin/env python3
"""Apply custom black-to-red colormap"""

import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create custom black to red colormap
colored = np.zeros((*img.shape, 3), dtype=np.uint8)
colored[:, :, 2] = img  # Red channel only

# Save colored image
cv2.imwrite('custom_red_colormap.png', colored)

print("Applied custom black-to-red colormap")