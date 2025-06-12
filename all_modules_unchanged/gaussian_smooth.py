import cv2
import numpy as np

# Load grayscale image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian smoothing
smoothed = cv2.GaussianBlur(image, (5, 5), 1.0)

# Save result
cv2.imwrite('2_smoothed.jpg', smoothed)
print("Gaussian smoothing applied")
