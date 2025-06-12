import cv2
import numpy as np

# Load difference map
diff_map = cv2.imread("gradient_magnitude.png", cv2.IMREAD_GRAYSCALE)

# Apply OpenCV heat colormap
heatmap = cv2.applyColorMap(diff_map, cv2.COLORMAP_HOT)

cv2.imwrite("heat_colormap.png", heatmap)