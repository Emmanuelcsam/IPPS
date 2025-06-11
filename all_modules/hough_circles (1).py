import cv2
import numpy as np

# Load and prepare image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles using Hough Transform
circles = cv2.HoughCircles(
    blurred, cv2.HOUGH_GRADIENT, 1, 50,
    param1=50, param2=30,
    minRadius=10, maxRadius=min(img.shape[:2]) // 2
)

# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imwrite("hough_circles_output.jpg", img)