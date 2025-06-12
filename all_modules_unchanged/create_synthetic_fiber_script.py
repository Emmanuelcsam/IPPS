import cv2
import numpy as np

# Create synthetic fiber image
size = 400
img = np.ones((size, size), dtype=np.uint8) * 128
center = size // 2

# Add circular fiber regions
cv2.circle(img, (center, center), int(size * 0.375), 100, -1)  # Cladding
cv2.circle(img, (center, center), int(size * 0.125), 60, -1)   # Core

# Add noise
noise = np.random.normal(0, 5, img.shape)
img = np.clip(img + noise, 0, 255).astype(np.uint8)

# Display result
cv2.imshow('Synthetic Fiber', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite('synthetic_fiber.jpg', img)