import cv2
import numpy as np

# Load image to get dimensions
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
height, width = img.shape

# Define circle parameters (adjust based on detected circles)
center_x, center_y = 150, 150  # Center coordinates
inner_radius = 40  # Core radius
outer_radius = 80  # Cladding outer radius

# Create annular mask
y_indices, x_indices = np.ogrid[:height, :width]
distance_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
mask = np.zeros((height, width), dtype=np.uint8)
mask[(distance_from_center > inner_radius) & (distance_from_center <= outer_radius)] = 255

# Save mask
cv2.imwrite("annular_mask.jpg", mask)
print(f"Created annular mask: inner radius {inner_radius}, outer radius {outer_radius}")