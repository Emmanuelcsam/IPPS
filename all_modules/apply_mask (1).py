import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)

# Create a sample mask (circular)
h, w = img.shape
center = (w//2, h//2)
mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask, center, 50, 255, -1)

# Apply mask
masked_image = cv2.bitwise_and(img, img, mask=mask)

# Save result
cv2.imwrite("masked_image.jpg", masked_image)