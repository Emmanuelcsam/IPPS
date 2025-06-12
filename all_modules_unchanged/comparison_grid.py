import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create sample defect masks (replace with your actual masks)
h, w = original.shape

# Region defects
region_defects = np.zeros_like(original)
cv2.circle(region_defects, (w//3, h//3), 20, 255, -1)
cv2.circle(region_defects, (2*w//3, 2*h//3), 25, 255, -1)

# Scratches
scratches = np.zeros_like(original)
cv2.line(scratches, (50, 50), (w-50, h-50), 255, 3)
cv2.line(scratches, (w-50, 50), (50, h-50), 255, 2)

# All defects combined
all_defects = cv2.bitwise_or(region_defects, scratches)

# Resize images to fit in grid
scale = 0.5
new_h, new_w = int(h * scale), int(w * scale)
img1 = cv2.resize(original, (new_w, new_h))
img2 = cv2.resize(region_defects, (new_w, new_h))
img3 = cv2.resize(scratches, (new_w, new_h))
img4 = cv2.resize(all_defects, (new_w, new_h))

# Create 2x2 grid
top_row = np.hstack([img1, img2])
bottom_row = np.hstack([img3, img4])
grid = np.vstack([top_row, bottom_row])

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(grid, 'Original', (10, 30), font, 0.7, 255, 2)
cv2.putText(grid, 'Region Defects', (new_w + 10, 30), font, 0.7, 255, 2)
cv2.putText(grid, 'Scratches', (10, new_h + 30), font, 0.7, 255, 2)
cv2.putText(grid, 'All Defects', (new_w + 10, new_h + 30), font, 0.7, 255, 2)

# Save and display
cv2.imwrite('comparison_grid.jpg', grid)
cv2.imshow('Comparison Grid', grid)
cv2.waitKey(0)
cv2.destroyAllWindows()