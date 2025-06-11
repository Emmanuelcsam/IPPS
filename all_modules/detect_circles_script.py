import cv2
import numpy as np

# Load and prepare image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Blur image first for better circle detection
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Detect circles
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=10, maxRadius=100
    )
    
    if circles is None:
        print("No circles detected")
    else:
        circles = np.uint16(np.around(circles[0]))
        print(f"Detected {len(circles)} circles")
        
        # Draw detected circles
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i, (x, y, r) in enumerate(circles):
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
            print(f"Circle {i+1}: center=({x},{y}), radius={r}")
        
        cv2.imshow("Detected Circles", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()