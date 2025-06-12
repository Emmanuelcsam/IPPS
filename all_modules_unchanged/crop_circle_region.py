import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path)

# Define circle
center = (img.shape[1] // 2, img.shape[0] // 2)
radius = min(img.shape[0], img.shape[1]) // 3

# Create circular mask
mask = np.zeros(img.shape[:2], np.uint8)
cv2.circle(mask, center, radius, 255, -1)

# Apply mask
masked = cv2.bitwise_and(img, img, mask=mask)

# Find bounding box of the circle
x = center[0] - radius
y = center[1] - radius
w = h = 2 * radius

# Crop to bounding box
cropped = masked[y:y+h, x:x+w]

# Save result
cv2.imwrite("cropped_circle.png", cropped)
print(f"Cropped circular region: size={cropped.shape[:2]}")