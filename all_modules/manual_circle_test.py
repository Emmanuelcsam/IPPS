import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
height, width = img.shape

# Manual circle parameters - adjust these based on your image
# You can estimate by looking at the image
center_x = width // 2  # Center of image
center_y = height // 2
core_radius = 30       # Adjust based on your fiber
cladding_radius = 60   # Adjust based on your fiber

print(f"Image size: {width}x{height}")
print(f"Using center: ({center_x}, {center_y})")
print(f"Core radius: {core_radius}")
print(f"Cladding radius: {cladding_radius}")

# Create core mask
y_indices, x_indices = np.ogrid[:height, :width]
distance = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
core_mask = np.zeros((height, width), dtype=np.uint8)
core_mask[distance <= core_radius] = 255

# Create cladding mask
cladding_mask = np.zeros((height, width), dtype=np.uint8)
cladding_mask[(distance > core_radius) & (distance <= cladding_radius)] = 255

# Apply masks and save
core_result = cv2.bitwise_and(img, img, mask=core_mask)
cladding_result = cv2.bitwise_and(img, img, mask=cladding_mask)

cv2.imwrite("manual_core.jpg", core_result)
cv2.imwrite("manual_cladding.jpg", cladding_result)
cv2.imwrite("manual_core_mask.jpg", core_mask)
cv2.imwrite("manual_cladding_mask.jpg", cladding_mask)

print("Manual masks created and saved!")