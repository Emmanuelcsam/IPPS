import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply threshold first
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Morphological opening
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Display result
cv2.imshow('Morphological Opening', opened)
cv2.waitKey(0)
cv2.destroyAllWindows()