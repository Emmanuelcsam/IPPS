import cv2
import numpy as np

# Load image and detect circles
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

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

# Draw circles if found
if circles is not None:
    circles = np.around(circles[0]).astype(np.uint16)
    for x, y, r in circles:
        # Draw circle outline
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        # Draw center point
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

# Display result
cv2.imshow('Circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()