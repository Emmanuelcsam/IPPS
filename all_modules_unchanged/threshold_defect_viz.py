#!/usr/bin/env python3
"""Create defect visualization using thresholds"""

import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create colored defect visualization
colored = np.zeros((*img.shape, 3), dtype=np.uint8)

# Define thresholds for different "defect" levels
# Background: 0-50 (black)
# Level 1: 51-100 (green)
# Level 2: 101-150 (blue)
# Level 3: 151-255 (red)

colored[img <= 50] = [0, 0, 0]          # Black
colored[(img > 50) & (img <= 100)] = [0, 255, 0]    # Green
colored[(img > 100) & (img <= 150)] = [255, 0, 0]   # Blue
colored[img > 150] = [0, 0, 255]        # Red

# Save defect visualization
cv2.imwrite('defect_visualization.png', colored)

print("Created defect visualization with color coding")