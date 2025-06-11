import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply threshold to create binary mask
threshold_value = 127  # Adjust as needed
_, binary_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

# Apply mask to original image
masked_img = cv2.bitwise_and(img, img, mask=binary_mask)

# Save results
cv2.imwrite("binary_mask.png", binary_mask)
cv2.imwrite("threshold_masked.png", masked_img)
print(f"Created binary mask with threshold={threshold_value}")