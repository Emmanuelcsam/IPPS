import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)

# Extract core region
h, w = img.shape
center = (w//2, h//2)
core_radius = 50

core_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(core_mask, center, core_radius, 255, -1)

core_region = cv2.bitwise_and(img, img, mask=core_mask)

# Save core region
cv2.imwrite("core_region.jpg", core_region)