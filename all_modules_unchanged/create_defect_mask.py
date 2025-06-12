import cv2
import numpy as np

# Load image to get dimensions
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
h, w = image.shape

# Create region defects mask
region_mask = np.zeros((h, w), dtype=np.uint8)

# Add circular defects
cv2.circle(region_mask, (w//4, h//4), 15, 255, -1)
cv2.circle(region_mask, (3*w//4, h//3), 20, 255, -1)
cv2.ellipse(region_mask, (w//2, 2*h//3), (40, 20), 45, 0, 360, 255, -1)

# Apply morphological operations for realistic shapes
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel)

# Save region defects mask
cv2.imwrite('region_defects_mask.jpg', region_mask)

# Create scratches mask
scratch_mask = np.zeros((h, w), dtype=np.uint8)

# Add line defects (scratches)
cv2.line(scratch_mask, (50, 100), (w-100, h-50), 255, 2)
cv2.line(scratch_mask, (w-50, 50), (100, h//2), 255, 3)

# Add curved scratch using polylines
pts = np.array([[w//2, 50], [w//2+50, 150], [w//2, 250], [w//2-50, 350]], np.int32)
cv2.polylines(scratch_mask, [pts], False, 255, 2)

# Save scratches mask
cv2.imwrite('scratches_mask.jpg', scratch_mask)

# Combine all defects
all_defects = cv2.bitwise_or(region_mask, scratch_mask)
cv2.imwrite('all_defects_mask.jpg', all_defects)

# Display results
cv2.imshow('Region Defects', region_mask)
cv2.imshow('Scratches', scratch_mask)
cv2.imshow('All Defects', all_defects)
cv2.waitKey(0)
cv2.destroyAllWindows()