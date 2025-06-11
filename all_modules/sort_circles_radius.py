import cv2
import numpy as np

# Load and detect circles
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles
circles = cv2.HoughCircles(
    blurred, cv2.HOUGH_GRADIENT, 1, 50,
    param1=50, param2=30,
    minRadius=10, maxRadius=min(img.shape[:2]) // 2
)

if circles is not None:
    # Convert and sort by radius (3rd column)
    circles = circles[0]
    sorted_circles = sorted(circles, key=lambda x: x[2])
    
    # Draw circles with different colors based on size
    for i, circle in enumerate(sorted_circles):
        color = (0, 255 - i*50, i*50)  # gradient from green to red
        cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), color, 2)
        # Label with radius
        cv2.putText(img, f"r={int(circle[2])}", 
                   (int(circle[0]-20), int(circle[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imwrite("sorted_circles_output.jpg", img)