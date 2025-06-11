import cv2
import numpy as np

# Load image and mask
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Load pre-created mask or create one
# Option 1: Load existing mask
# mask = cv2.imread("circular_mask.jpg", cv2.IMREAD_GRAYSCALE)

# Option 2: Create mask here (using example values)
center_x, center_y, radius = 150, 150, 40
y_indices, x_indices = np.ogrid[:img.shape[0], :img.shape[1]]
distance = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
mask = np.zeros(img.shape, dtype=np.uint8)
mask[distance <= radius] = 255

# Apply mask to isolate core
core_only = cv2.bitwise_and(img, img, mask=mask)

# Save result
cv2.imwrite("core_isolated.jpg", core_only)
print("Core region isolated")