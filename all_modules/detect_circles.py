import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect circles using Hough Transform
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                          param1=50, param2=30, minRadius=10, maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))
    output = img.copy()
    
    for i in circles[0, :]:
        # Draw circle and center
        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    cv2.imwrite("detected_circles.png", output)
    print(f"Detected {len(circles[0])} circles")
    for idx, circle in enumerate(circles[0]):
        print(f"Circle {idx}: center=({circle[0]}, {circle[1]}), radius={circle[2]}")