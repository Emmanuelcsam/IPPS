#!/usr/bin/env python3
"""Apply jet colormap to grayscale image"""

import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply jet colormap
colored = cv2.applyColorMap(img, cv2.COLORMAP_JET)

# Save colored image
cv2.imwrite('jet_colormap.png', colored)

print("Applied jet colormap and saved to jet_colormap.png")