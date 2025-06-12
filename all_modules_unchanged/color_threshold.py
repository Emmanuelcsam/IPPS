import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path)

# Convert to different color spaces
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Apply various thresholding methods
# Simple threshold
_, simple_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Adaptive threshold
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)

# Otsu's threshold
_, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Color-based threshold (HSV)
# Define range for detecting specific colors (adjust as needed)
lower_bound = np.array([0, 0, 100])
upper_bound = np.array([180, 30, 255])
color_mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Save thresholded images
cv2.imwrite('threshold_simple.jpg', simple_thresh)
cv2.imwrite('threshold_adaptive.jpg', adaptive_thresh)
cv2.imwrite('threshold_otsu.jpg', otsu_thresh)
cv2.imwrite('threshold_color.jpg', color_mask)

# Display results
cv2.imshow('Simple Threshold', simple_thresh)
cv2.imshow('Adaptive Threshold', adaptive_thresh)
cv2.imshow('Otsu Threshold', otsu_thresh)
cv2.imshow('Color Mask', color_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()