import cv2
import numpy as np

# Load difference map
diff_map = cv2.imread("gradient_magnitude.png", cv2.IMREAD_GRAYSCALE)

# Create custom gradient: Black -> Red -> Yellow
heatmap = np.zeros((*diff_map.shape, 3), dtype=np.uint8)

# Red channel: increases linearly
heatmap[:, :, 2] = diff_map

# Green channel: increases after halfway point (creates yellow)
halfway = 128
mask = diff_map > halfway
heatmap[mask, 1] = ((diff_map[mask] - halfway) * 2).astype(np.uint8)

cv2.imwrite("custom_gradient_heatmap.png", heatmap)