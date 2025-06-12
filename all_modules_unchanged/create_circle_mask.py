import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path)

# Create a circular mask
center = (img.shape[1]//2, img.shape[0]//2)  # Center of image
radius = min(img.shape[0], img.shape[1])//3  # 1/3 of smallest dimension

mask = np.zeros(img.shape[:2], np.uint8)
cv2.circle(mask, center, radius, 255, -1)

# Apply mask to image
masked_img = cv2.bitwise_and(img, img, mask=mask)

# Save results
cv2.imwrite("circle_mask.png", mask)
cv2.imwrite("masked_image.png", masked_img)
print(f"Created circle mask at center {center} with radius {radius}")