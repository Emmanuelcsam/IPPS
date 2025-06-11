import cv2
import numpy as np

# Load grayscale image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply median filtering
Ir_smooth = cv2.medianBlur(image, 3)

# Save result
cv2.imwrite('6_median_filtered.jpg', Ir_smooth)
print("Median filtering applied")
