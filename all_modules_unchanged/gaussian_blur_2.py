import cv2
import numpy as np

# Load grayscale image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Save result
cv2.imwrite("blurred_output.jpg", blurred)
print("Gaussian blur applied")