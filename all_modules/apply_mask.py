import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)

# Create a simple circular mask (example)
mask = np.zeros(img.shape, dtype=np.uint8)
center = (img.shape[1]//2, img.shape[0]//2)
radius = min(center[0], center[1]) // 2
cv2.circle(mask, center, radius, 255, -1)

# Apply mask
masked_img = cv2.bitwise_and(img, img, mask=mask)
print(f"Masked pixels: {np.count_nonzero(masked_img)}")