#!/usr/bin/env python3
"""Convert image to grayscale"""
import cv2

# Load and convert to grayscale
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(f"Grayscale shape: {gray.shape}")
cv2.imshow("Grayscale Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()