import cv2
import numpy as np

# Create base image
img = np.ones((300, 300), dtype=np.uint8) * 128

# Add a circle
cv2.circle(img, (150, 150), 100, 100, -1)

# Add scratches
cv2.line(img, (50, 50), (250, 250), 160, 2)
cv2.line(img, (100, 150), (200, 150), 155, 1)
cv2.line(img, (150, 50), (150, 250), 158, 1)

# Add noise
noise = np.random.normal(0, 5, img.shape)
img = np.clip(img + noise, 0, 255).astype(np.uint8)

# Save test image
cv2.imwrite('test_scratched_image.jpg', img)
print("Test image with scratches created")
