import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path)

# Define inner circle (adjust these values based on your image)
center_x = img.shape[1] // 2
center_y = img.shape[0] // 2
radius = min(img.shape[0], img.shape[1]) // 4  # Adjust radius as needed

# Create mask for inner circle
mask = np.zeros(img.shape[:2], np.uint8)
cv2.circle(mask, (center_x, center_y), radius, 255, -1)

# Extract inner circle
inner_circle = cv2.bitwise_and(img, img, mask=mask)

# Save result
cv2.imwrite("inner_circle.png", inner_circle)
print(f"Extracted inner circle: center=({center_x}, {center_y}), radius={radius}")