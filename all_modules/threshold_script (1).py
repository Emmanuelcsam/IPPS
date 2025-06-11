#!/usr/bin/env python3
"""Apply adaptive threshold for fiber segmentation"""
import cv2
import numpy as np

# File path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load and convert to grayscale
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 1)

# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(blurred, 255, 
                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 11, 2)

# Display and save
cv2.imshow("Threshold", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("threshold.jpg", thresh)