import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)

# Extract cladding region
h, w = img.shape
center = (w//2, h//2)
inner_radius = 50
outer_radius = 100

# Create annular mask for cladding
outer_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(outer_mask, center, outer_radius, 255, -1)

inner_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(inner_mask, center, inner_radius, 255, -1)

cladding_mask = cv2.subtract(outer_mask, inner_mask)

cladding_region = cv2.bitwise_and(img, img, mask=cladding_mask)

# Save cladding region
cv2.imwrite("cladding_region.jpg", cladding_region)