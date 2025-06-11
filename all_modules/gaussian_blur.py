import cv2
import numpy as np

# Load difference map
diff_map = cv2.imread("gradient_magnitude.png", cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur
blur_radius = 3
blurred = cv2.GaussianBlur(diff_map, (blur_radius*2+1, blur_radius*2+1), 0)

cv2.imwrite("blurred_diff.png", blurred)