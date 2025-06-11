#!/usr/bin/env python3
"""Load and display image"""
import cv2

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path)

if image is None:
    print(f"Failed to load image: {img_path}")
else:
    print(f"Image shape: {image.shape}")
    cv2.imshow("Loaded Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()