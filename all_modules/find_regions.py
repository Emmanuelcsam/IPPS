import cv2
import numpy as np

# Load difference map
diff_map = cv2.imread("gradient_magnitude.png", cv2.IMREAD_GRAYSCALE)

# Threshold to find high difference areas
threshold = np.percentile(diff_map, 90)  # Top 10%
high_diff_mask = (diff_map > threshold).astype(np.uint8)

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(high_diff_mask, connectivity=8)

print(f"Found {num_labels - 1} high difference regions")

# Draw bounding boxes
output = cv2.cvtColor(diff_map, cv2.COLOR_GRAY2BGR)
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite("regions_highlighted.png", output)