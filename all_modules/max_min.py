import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Calculate min-max residual
kernel_size = 7
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
max_img = cv2.dilate(img, kernel)
min_img = cv2.erode(img, kernel)
residual = max_img - min_img

# Display result
cv2.imshow('Min-Max Residual', residual)
cv2.waitKey(0)
cv2.destroyAllWindows()