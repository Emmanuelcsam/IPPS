import cv2
import numpy as np

# Load and preprocess image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Detect circles with fixed parameters
circles = cv2.HoughCircles(
    blurred, 
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=50,
    param2=30,
    minRadius=5,
    maxRadius=200
)

if circles is not None:
    circles = np.around(circles[0]).astype(np.uint16)
    print(f"Found {len(circles)} circles")
    for i, (x, y, r) in enumerate(circles):
        print(f"Circle {i+1}: center=({x}, {y}), radius={r}")
else:
    print("No circles found")