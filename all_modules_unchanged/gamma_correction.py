import cv2
import numpy as np

# Load difference map
diff_map = cv2.imread("gradient_magnitude.png", cv2.IMREAD_GRAYSCALE)

# Apply gamma correction
gamma = 0.5  # < 1 emphasizes low values, > 1 emphasizes high values
diff_map = diff_map.astype(np.float32) / 255.0
diff_map = np.power(diff_map, gamma) * 255.0
diff_map = diff_map.astype(np.uint8)

cv2.imwrite("gamma_corrected.png", diff_map)