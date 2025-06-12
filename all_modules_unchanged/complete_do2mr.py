import cv2
import numpy as np

# Load and preprocess image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Gaussian smoothing
smoothed = cv2.GaussianBlur(gray, (5, 5), 1.0)

# 2. Min-Max filtering
kernel_size = 7
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
Imax = cv2.dilate(smoothed, kernel)
Imin = cv2.erode(smoothed, kernel)

# 3. Residual calculation
Ir = Imax - Imin

# 4. Smooth residual
Ir_smooth = cv2.medianBlur(Ir, 3)

# 5. Threshold segmentation
mu = np.mean(Ir_smooth)
sigma = np.std(Ir_smooth)
gamma = 2.5
threshold = mu + gamma * sigma
_, binary = cv2.threshold(Ir_smooth, threshold, 255, cv2.THRESH_BINARY)

# 6. Morphological opening
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

# Save final result
cv2.imwrite('do2mr_result.jpg', result)
print(f"Detected {np.sum(result > 0)} defect pixels")
