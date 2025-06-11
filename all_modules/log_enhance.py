#!/usr/bin/env python3
"""Apply logarithmic enhancement to emphasize small differences"""
import cv2
import numpy as np

# Load difference map (use any edge detection result)
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create simple difference map using gradient
grad = np.abs(np.gradient(img.astype(np.float32))[0])

# Normalize to 0-1
if grad.max() > 0:
    grad_norm = grad / grad.max()
else:
    grad_norm = grad

# Logarithmic enhancement
epsilon = 1e-10
enhanced = np.log(grad_norm + epsilon * 2.0)
enhanced = enhanced - enhanced.min()
if enhanced.max() > 0:
    enhanced = (enhanced / enhanced.max() * 255).astype(np.uint8)

cv2.imwrite('log_enhanced.png', enhanced)