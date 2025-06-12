import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)

# Gaussian blur first
blurred = cv2.GaussianBlur(img, (5, 5), 1.4)

# Calculate gradients
grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Gradient magnitude
gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

# Normalize
gradient_mag = (gradient_mag / gradient_mag.max() * 255).astype(np.uint8)
cv2.imwrite("canny_style_gradient.png", gradient_mag)