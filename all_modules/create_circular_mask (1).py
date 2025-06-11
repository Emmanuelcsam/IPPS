import cv2
import numpy as np

# Load image to get dimensions
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
height, width = img.shape

# Define circle parameters (example values - adjust based on your detected circles)
center_x, center_y = 150, 150  # Center coordinates
radius = 40  # Radius

# Create circular mask using numpy
y_indices, x_indices = np.ogrid[:height, :width]
distance_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
mask = np.zeros((height, width), dtype=np.uint8)
mask[distance_from_center <= radius] = 255

# Save mask
cv2.imwrite("circular_mask.jpg", mask)
print(f"Created circular mask at ({center_x}, {center_y}) with radius {radius}")