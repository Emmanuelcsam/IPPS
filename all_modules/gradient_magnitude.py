import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32)

# Calculate gradients
grad_x = np.zeros_like(img)
grad_y = np.zeros_like(img)

grad_x[:, 1:] = img[:, 1:] - img[:, :-1]
grad_y[1:, :] = img[1:, :] - img[:-1, :]

# Gradient magnitude
gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

# Normalize to 0-255
gradient_mag = (gradient_mag / gradient_mag.max() * 255).astype(np.uint8)
cv2.imwrite("gradient_magnitude.png", gradient_mag)