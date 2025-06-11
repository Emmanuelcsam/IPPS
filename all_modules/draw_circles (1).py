import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path)

# Example circle parameters (replace with your detected circles)
# Format: (center_x, center_y, radius)
core_circle = (150, 150, 40)
cladding_circle = (150, 150, 80)

# Draw circles
# Core in green
cv2.circle(img, (core_circle[0], core_circle[1]), core_circle[2], (0, 255, 0), 2)
# Cladding in red
cv2.circle(img, (cladding_circle[0], cladding_circle[1]), cladding_circle[2], (0, 0, 255), 2)
# Center point in blue
cv2.circle(img, (core_circle[0], core_circle[1]), 3, (255, 0, 0), -1)

# Save result
cv2.imwrite("circles_drawn.jpg", img)
print("Circles drawn on image")