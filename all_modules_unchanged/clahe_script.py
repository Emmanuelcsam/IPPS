#!/usr/bin/env python3
"""Apply CLAHE enhancement"""
import cv2

# Load image and apply CLAHE
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(image)

cv2.imshow("Original", image)
cv2.imshow("CLAHE Enhanced", enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()