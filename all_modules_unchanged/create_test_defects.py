#!/usr/bin/env python3
"""
Create test defect patterns for validation testing
"""
import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create synthetic test image with your image dimensions
test_img = original.copy()
h, w = test_img.shape

# Add artificial defects with different contrasts
# High contrast defect (dark spot)
cv2.circle(test_img, (w//4, h//4), 10, 50, -1)

# Medium contrast defect
cv2.circle(test_img, (3*w//4, h//4), 10, 100, -1)

# Low contrast defect
cv2.circle(test_img, (w//2, h//2), 10, 120, -1)

# Linear defect (scratch)
cv2.line(test_img, (w//4, 3*h//4), (3*w//4, 3*h//4), 60, 3)

# Create corresponding defect mask
defect_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(defect_mask, (w//4, h//4), 10, 255, -1)
cv2.circle(defect_mask, (3*w//4, h//4), 10, 255, -1)
cv2.circle(defect_mask, (w//2, h//2), 10, 255, -1)
cv2.line(defect_mask, (w//4, 3*h//4), (3*w//4, 3*h//4), 255, 3)

# Calculate contrasts for each defect
print("Test defect contrasts:")
for i, (x, y, r) in enumerate([(w//4, h//4, 10), (3*w//4, h//4, 10), (w//2, h//2, 10)]):
    # Get defect region
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    
    # Get surrounding
    dilated = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    surround = cv2.subtract(dilated, mask)
    
    defect_mean = np.mean(test_img[mask > 0])
    surround_mean = np.mean(test_img[surround > 0])
    contrast = abs(defect_mean - surround_mean)
    
    print(f"  Defect {i+1}: contrast = {contrast:.1f}")

# Display results
cv2.imshow('Original Image', original)
cv2.imshow('Test Image with Defects', test_img)
cv2.imshow('Defect Mask', defect_mask)

# Show difference
diff = cv2.absdiff(original, test_img)
cv2.imshow('Difference', diff)

cv2.waitKey(0)
cv2.destroyAllWindows()