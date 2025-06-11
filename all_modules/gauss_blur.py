import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)

# Display result
cv2.imshow('Gaussian Blur', blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()