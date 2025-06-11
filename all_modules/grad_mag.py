#!/usr/bin/env python3
"""Calculate gradient magnitude to find intensity differences"""
import cv2
import numpy as np

# Load grayscale image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Calculate gradients
grad_x = np.zeros_like(img)
grad_y = np.zeros_like(img)
grad_x[:, 1:] = img[:, 1:] - img[:, :-1]
grad_y[1:, :] = img[1:, :] - img[:-1, :]

# Gradient magnitude
grad_mag = np.sqrt(grad_x**2 + grad_y**2)

# Normalize to 0-255
if grad_mag.max() > 0:
    grad_mag = (grad_mag / grad_mag.max() * 255).astype(np.uint8)

cv2.imwrite('gradient_magnitude.png', grad_mag)