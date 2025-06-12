#!/usr/bin/env python3
"""Save intensity matrix as image"""

import cv2
import numpy as np

# Load saved matrix or create from image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Save as image file
output_path = 'output_grayscale.png'
cv2.imwrite(output_path, img)

print(f"Saved image to: {output_path}")