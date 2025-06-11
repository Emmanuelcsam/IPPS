#!/usr/bin/env python3
"""Connected Components Analysis"""
import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create binary image using threshold
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Create colored output for visualization
output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

# Assign random colors to each component (skip background which is label 0)
for i in range(1, num_labels):
    # Get component statistics
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    
    # Generate random color
    color = np.random.randint(50, 255, size=3).tolist()
    
    # Color the component
    output[labels == i] = color
    
    # Draw bounding box
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
    # Print component info
    print(f"Component {i}: Area={area}, Width={w}, Height={h}")

# Filter components by size
min_area = 50
filtered = binary.copy()
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        filtered[labels == i] = 0

# Display results
cv2.imshow('Original', image)
cv2.imshow('Binary', binary)
cv2.imshow('Connected Components', output)
cv2.imshow('Filtered by Size', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save results
cv2.imwrite('connected_components.png', output)
cv2.imwrite('filtered_components.png', filtered)