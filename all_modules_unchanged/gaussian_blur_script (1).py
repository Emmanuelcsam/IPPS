#!/usr/bin/env python3
"""Apply Gaussian blur"""
import cv2

# Load image and apply blur
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image, (5, 5), 0)

cv2.imshow("Original", image)
cv2.imshow("Gaussian Blur", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()