import cv2
import numpy as np

# Load and blur image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Detect circles
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.0,
    minDist=50,
    param1=50,
    param2=30,
    minRadius=10,
    maxRadius=200
)

if circles is not None:
    circles = np.around(circles).astype(np.uint16)[0, :]
    circles = sorted(circles, key=lambda x: x[2])  # Sort by radius
    
    print(f"Detected {len(circles)} circles:")
    for i, (x, y, r) in enumerate(circles):
        print(f"Circle {i}: Center=({x}, {y}), Radius={r}")
    
    # Save circle parameters
    np.save("detected_circles.npy", circles)
else:
    print("No circles detected")