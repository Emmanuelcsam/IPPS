#!/usr/bin/env python3
"""Save processed image"""
import cv2

# Load, process and save
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply simple processing (example: blur)
processed = cv2.GaussianBlur(image, (5, 5), 0)

# Save processed image
output_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38_processed.jpg"
cv2.imwrite(output_path, processed)
print(f"Saved processed image to: {output_path}")