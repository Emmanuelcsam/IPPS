import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)

# Create annular mask
h, w = img.shape
center = (w//2, h//2)
inner_radius = 50
outer_radius = 100

outer_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(outer_mask, center, outer_radius, 255, -1)

inner_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(inner_mask, center, inner_radius, 255, -1)

annular_mask = cv2.subtract(outer_mask, inner_mask)

# Save mask
cv2.imwrite("annular_mask.jpg", annular_mask)