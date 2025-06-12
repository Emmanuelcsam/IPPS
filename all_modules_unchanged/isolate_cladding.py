import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create annular mask (example values - adjust based on your circles)
center_x, center_y = 150, 150
inner_radius, outer_radius = 40, 80

y_indices, x_indices = np.ogrid[:img.shape[0], :img.shape[1]]
distance = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
mask = np.zeros(img.shape, dtype=np.uint8)
mask[(distance > inner_radius) & (distance <= outer_radius)] = 255

# Apply mask to isolate cladding
cladding_only = cv2.bitwise_and(img, img, mask=mask)

# Save result
cv2.imwrite("cladding_isolated.jpg", cladding_only)
print("Cladding region isolated")