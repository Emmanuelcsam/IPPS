import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32)

# Calculate max difference to 8-connected neighbors
height, width = img.shape
diff_map = np.zeros_like(img)

# Pad image for boundary handling
padded = np.pad(img, 1, mode='edge')

# Check all 8 neighbors
offsets = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)]

for dy, dx in offsets:
    neighbor = padded[1+dy:height+1+dy, 1+dx:width+1+dx]
    diff = np.abs(img - neighbor)
    diff_map = np.maximum(diff_map, diff)

# Normalize
diff_map = (diff_map / diff_map.max() * 255).astype(np.uint8)
cv2.imwrite("max_neighbor_diff.png", diff_map)