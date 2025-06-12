import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Laplacian
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = np.abs(laplacian)

# Normalize
laplacian = (laplacian / laplacian.max() * 255).astype(np.uint8)
cv2.imwrite("laplacian_edges.png", laplacian)