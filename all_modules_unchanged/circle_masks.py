import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Define circles (example values - adjust based on your detection)
# Format: (center_x, center_y, radius)
inner_circle = (200, 200, 80)  # Adjust these values
outer_circle = (200, 200, 150)  # Adjust these values

# Create masks
inner_mask = np.zeros((height, width), np.uint8)
outer_mask = np.zeros((height, width), np.uint8)

# Draw filled circles on masks
cv2.circle(inner_mask, (inner_circle[0], inner_circle[1]), inner_circle[2], 255, -1)
cv2.circle(outer_mask, (outer_circle[0], outer_circle[1]), outer_circle[2], 255, -1)

# Create ring mask (outer minus inner)
ring_mask = cv2.subtract(outer_mask, inner_mask)

# Display masks
cv2.imshow('Inner Mask', inner_mask)
cv2.imshow('Outer Mask', outer_mask)
cv2.imshow('Ring Mask', ring_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save masks
cv2.imwrite('inner_mask.png', inner_mask)
cv2.imwrite('ring_mask.png', ring_mask)