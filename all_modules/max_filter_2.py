import cv2
import numpy as np

# Load grayscale image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create structuring element
kernel_size = 7
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

# Apply maximum filtering (dilation)
Imax = cv2.dilate(image, kernel)

# Save result
cv2.imwrite('3_max_filtered.jpg', Imax)
print("Maximum filtering applied")
