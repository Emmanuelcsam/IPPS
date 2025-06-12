#!/usr/bin/env python3
"""
Demonstrate morphological operations for defect analysis
"""
import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create binary mask
_, binary = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)

# Different kernel shapes and sizes
kernels = {
    'Small Circle': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    'Medium Circle': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    'Large Circle': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    'Square': cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    'Cross': cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
}

# Apply different operations
for kernel_name, kernel in kernels.items():
    # Dilation (expand regions)
    dilated = cv2.dilate(binary, kernel)
    
    # Erosion (shrink regions)
    eroded = cv2.erode(binary, kernel)
    
    # Opening (remove small objects)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Closing (fill small holes)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Gradient (outline)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    
    # Create surrounding mask (useful for contrast calculation)
    surround = cv2.subtract(dilated, binary)
    
    # Display results
    display = np.hstack([
        cv2.resize(binary, (200, 200)),
        cv2.resize(dilated, (200, 200)),
        cv2.resize(surround, (200, 200))
    ])
    
    cv2.imshow(f'{kernel_name} - Original | Dilated | Surround', display)
    
    print(f"\n{kernel_name} kernel:")
    print(f"  Original pixels: {np.sum(binary > 0)}")
    print(f"  Dilated pixels: {np.sum(dilated > 0)}")
    print(f"  Surround pixels: {np.sum(surround > 0)}")

cv2.waitKey(0)
cv2.destroyAllWindows()