import cv2
import numpy as np

# Load edge/difference map (use any of the above outputs)
diff_map = cv2.imread("gradient_magnitude.png", cv2.IMREAD_GRAYSCALE)

# Create black-to-red heatmap
heatmap = np.zeros((*diff_map.shape, 3), dtype=np.uint8)
heatmap[:, :, 2] = diff_map  # Red channel in BGR

cv2.imwrite("black_to_red_heatmap.png", heatmap)