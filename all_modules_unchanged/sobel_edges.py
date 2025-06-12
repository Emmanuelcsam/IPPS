import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Sobel
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Magnitude
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

# Normalize
sobel_mag = (sobel_mag / sobel_mag.max() * 255).astype(np.uint8)
cv2.imwrite("sobel_edges.png", sobel_mag)