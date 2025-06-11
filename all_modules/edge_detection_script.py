#!/usr/bin/env python3
"""Detect edges using Canny edge detection"""
import cv2
import numpy as np

# File path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load and convert to grayscale
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 1)

# Detect edges
edges = cv2.Canny(blurred, 50, 150)

# Display and save
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("edges_detected.jpg", edges)