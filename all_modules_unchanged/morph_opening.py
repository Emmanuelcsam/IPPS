import cv2
import numpy as np

# Load binary image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create small elliptical kernel
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Apply morphological opening
result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_small)

# Save result
cv2.imwrite('8_morph_opened.jpg', result)
print("Morphological opening applied")
