#!/usr/bin/env python3
"""Apply square root enhancement to moderately emphasize small differences"""
import cv2
import numpy as np

# Load image and create difference map
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
grad = np.abs(np.gradient(img.astype(np.float32))[0])

# Normalize to 0-1
if grad.max() > 0:
    grad_norm = grad / grad.max()
else:
    grad_norm = grad

# Square root enhancement (power of 0.5)
enhanced = np.power(grad_norm, 0.5)
enhanced = (enhanced * 255).astype(np.uint8)

cv2.imwrite('sqrt_enhanced.png', enhanced)