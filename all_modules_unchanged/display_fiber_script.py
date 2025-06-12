#!/usr/bin/env python3
"""Display fiber with cladding and core circles"""
import cv2
import numpy as np

# File path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find cladding
blurred = cv2.GaussianBlur(gray, (5, 5), 1)
h, w = gray.shape
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=int(min(h, w) * 0.15),
    param1=70,
    param2=35,
    minRadius=int(min(h, w) * 0.1),
    maxRadius=int(min(h, w) * 0.45)
)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    cx, cy, cladding_r = circles[0]
    
    # Estimate core radius (typical ratio)
    core_r = int(cladding_r * 0.072)
    
    # Draw visualization
    vis_img = image.copy()
    cv2.circle(vis_img, (cx, cy), cladding_r, (0, 255, 0), 2)
    cv2.circle(vis_img, (cx, cy), core_r, (0, 0, 255), 2)
    cv2.putText(vis_img, "Cladding", (cx + cladding_r + 5, cy), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis_img, "Core", (cx + core_r + 5, cy + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Display
    cv2.imshow("Fiber Detection", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    cv2.imwrite("fiber_complete.jpg", vis_img)
else:
    print("No fiber detected")