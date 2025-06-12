import cv2
import numpy as np

# Load difference map
diff_map = cv2.imread("gradient_magnitude.png", cv2.IMREAD_GRAYSCALE)

# Create binary mask of non-zero areas
mask = (diff_map > 0).astype(np.uint8) * 255

# Create colored visualization
mask_viz = np.zeros((*mask.shape, 3), dtype=np.uint8)
mask_viz[:, :, 1] = mask  # Green channel

# Add contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mask_viz, contours, -1, (0, 255, 255), 1)  # Yellow contours

cv2.imwrite("binary_mask.png", mask_viz)