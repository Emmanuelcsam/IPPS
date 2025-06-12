import cv2
import numpy as np

# Load binary image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
binary = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

# Create morphological kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))

# Apply morphological closing
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Save result
cv2.imwrite('cleaned_result.jpg', cleaned)
print("Morphological cleanup applied")
