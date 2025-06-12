#!/usr/bin/env python3
"""Apply hot colormap to grayscale image"""

import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply hot colormap
colored = cv2.applyColorMap(img, cv2.COLORMAP_HOT)

# Save colored image
cv2.imwrite('hot_colormap.png', colored)

print("Applied hot colormap and saved to hot_colormap.png")