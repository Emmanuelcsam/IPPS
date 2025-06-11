#!/usr/bin/env python3
"""Create radial intensity profile from fiber center"""
import cv2
import numpy as np

# File path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find fiber center (using HoughCircles)
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
    cx, cy, r = circles[0]
    
    # Calculate radial profile
    max_radius = min(cx, cy, w-cx, h-cy)
    profile = []
    
    for radius in range(0, max_radius, 2):
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, 2)
        
        pixels = gray[mask > 0]
        if len(pixels) > 0:
            profile.append(np.mean(pixels))
    
    # Create profile visualization
    profile_img = np.ones((256, len(profile)*2, 3), dtype=np.uint8) * 255
    profile_norm = np.array(profile)
    profile_norm = 255 - (profile_norm * 255 / profile_norm.max()).astype(int)
    
    for i in range(1, len(profile_norm)):
        cv2.line(profile_img, 
                 ((i-1)*2, profile_norm[i-1]), 
                 (i*2, profile_norm[i]), 
                 (0, 0, 0), 2)
    
    # Display
    cv2.imshow("Radial Intensity Profile", profile_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("radial_profile.jpg", profile_img)
    
    print(f"Profile calculated from center ({cx}, {cy})")
else:
    print("No fiber detected")