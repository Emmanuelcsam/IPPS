import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply binary threshold
threshold_value = 127
_, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# Display result
cv2.imshow('Original', image)
cv2.imshow('Binary Threshold', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite('threshold_result.jpg', binary_image)