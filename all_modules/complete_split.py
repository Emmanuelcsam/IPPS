import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = gray.shape

# Detect circles using Hough Transform
blurred = cv2.GaussianBlur(gray, (9, 9), 2)
circles = cv2.HoughCircles(
    blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
    param1=50, param2=30, minRadius=10, maxRadius=min(height, width) // 2
)

if circles is not None:
    circles = np.around(circles).astype(np.uint16)[0]
    
    # Sort by radius and take the two largest
    circles_sorted = sorted(circles, key=lambda x: x[2])[:2]
    inner = circles_sorted[0]
    outer = circles_sorted[1]
    
    # Create masks
    inner_mask = np.zeros((height, width), np.uint8)
    outer_mask = np.zeros((height, width), np.uint8)
    cv2.circle(inner_mask, (inner[0], inner[1]), inner[2], 255, -1)
    cv2.circle(outer_mask, (outer[0], outer[1]), outer[2], 255, -1)
    ring_mask = cv2.subtract(outer_mask, inner_mask)
    
    # Extract regions
    inner_only = cv2.bitwise_and(image, image, mask=inner_mask)
    ring_only = cv2.bitwise_and(image, image, mask=ring_mask)
    
    # Save results
    cv2.imwrite('inner_circle_result.png', inner_only)
    cv2.imwrite('outer_ring_result.png', ring_only)
    print("Saved: inner_circle_result.png and outer_ring_result.png")
else:
    print("No circles detected!")