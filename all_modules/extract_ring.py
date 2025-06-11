import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path)

# Define circles (adjust these values based on your image)
center = (img.shape[1] // 2, img.shape[0] // 2)
inner_radius = min(img.shape[0], img.shape[1]) // 4
outer_radius = min(img.shape[0], img.shape[1]) // 2

# Create masks
inner_mask = np.zeros(img.shape[:2], np.uint8)
outer_mask = np.zeros(img.shape[:2], np.uint8)
cv2.circle(inner_mask, center, inner_radius, 255, -1)
cv2.circle(outer_mask, center, outer_radius, 255, -1)

# Create ring mask by subtracting inner from outer
ring_mask = cv2.subtract(outer_mask, inner_mask)

# Extract ring
ring = cv2.bitwise_and(img, img, mask=ring_mask)

# Save results
cv2.imwrite("ring_mask.png", ring_mask)
cv2.imwrite("ring.png", ring)
print(f"Extracted ring: center={center}, inner_r={inner_radius}, outer_r={outer_radius}")