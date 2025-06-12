import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)

# Create circular mask
h, w = img.shape
center = (w//2, h//2)
radius = 50

mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask, center, radius, 255, -1)

# Save mask
cv2.imwrite("circular_mask.jpg", mask)