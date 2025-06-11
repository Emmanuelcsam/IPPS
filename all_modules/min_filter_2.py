import cv2
import numpy as np

# Load grayscale image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create structuring element
kernel_size = 7
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

# Apply minimum filtering (erosion)
Imin = cv2.erode(image, kernel)

# Save result
cv2.imwrite('4_min_filtered.jpg', Imin)
print("Minimum filtering applied")
