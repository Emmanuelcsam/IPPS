import cv2
import numpy as np

# Load difference map
diff_map = cv2.imread("gradient_magnitude.png", cv2.IMREAD_GRAYSCALE)

# Apply threshold
threshold = 50  # Adjust as needed
diff_map[diff_map < threshold] = 0

cv2.imwrite("thresholded_diff.png", diff_map)