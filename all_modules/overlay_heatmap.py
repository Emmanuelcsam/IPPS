#!/usr/bin/env python3
"""Overlay heatmap on original image"""
import cv2
import numpy as np

# Load original image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
original = cv2.imread(img_path)

# Create edge detection
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# Create red heatmap
heatmap = np.zeros_like(original)
heatmap[:, :, 2] = edges  # Red channel

# Overlay with 50% opacity
overlay = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)

cv2.imwrite('overlay_result.png', overlay)