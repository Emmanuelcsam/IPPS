#!/usr/bin/env python3
"""Display original and grayscale images using OpenCV"""

import cv2
import numpy as np

# Image path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load images
img_color = cv2.imread(image_path)
img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img_color is None:
    print(f"Failed to load image")
else:
    # Display original image
    cv2.imshow('Original Image', img_color)
    
    # Display grayscale image
    cv2.imshow('Grayscale Image', img_gray)
    
    # Display with colormap (heat map effect)
    img_heat = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    cv2.imshow('Intensity Heatmap', img_heat)
    
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()