import cv2
import numpy as np

# Load grayscale image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization
enhanced = cv2.equalizeHist(image)

# Save result
cv2.imwrite('enhanced.jpg', enhanced)
print("Histogram equalization applied")
