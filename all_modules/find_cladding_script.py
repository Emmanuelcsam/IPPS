#!/usr/bin/env python3
"""Find fiber cladding (outer circle) using HoughCircles"""
import cv2
import numpy as np

# File path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load and convert to grayscale
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 1)

# Detect circles
h, w = gray.shape
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=int(min(h, w) * 0.15),
    param1=70,
    param2=35,
    minRadius=int(min(h, w) * 0.1),
    maxRadius=int(min(h, w) * 0.45)
)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    x, y, r = circles[0]
    print(f"Cladding found: center=({x}, {y}), radius={r}")
    
    # Draw and save result
    output = image.copy()
    cv2.circle(output, (x, y), r, (0, 255, 0), 2)
    cv2.imwrite("cladding_detected.jpg", output)
else:
    print("No cladding detected")