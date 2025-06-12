import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# DO2MR process
# 1. Gaussian blur
smoothed = cv2.GaussianBlur(img, (5, 5), 1.0)

# 2. Min-max filtering
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
max_img = cv2.dilate(smoothed, kernel)
min_img = cv2.erode(smoothed, kernel)
residual = max_img - min_img

# 3. Median filter
residual_smooth = cv2.medianBlur(residual, 3)

# 4. Threshold
mu = np.mean(residual_smooth)
sigma = np.std(residual_smooth)
gamma = 2.5
threshold = mu + gamma * sigma
_, defects = cv2.threshold(residual_smooth, threshold, 255, cv2.THRESH_BINARY)

# 5. Clean up
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
defects = cv2.morphologyEx(defects, cv2.MORPH_OPEN, kernel_small)

# Display result
cv2.imshow('DO2MR Defects', defects)
cv2.waitKey(0)
cv2.destroyAllWindows()