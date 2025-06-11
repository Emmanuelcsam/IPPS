import cv2
import numpy as np

# Read image
img = cv2.imread('C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect circles
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=10, maxRadius=200)

# Create mask and draw circles
mask = np.zeros(gray.shape, dtype=np.uint8)
if circles is not None:
    circles = np.around(circles).astype(np.uint16)
    for i in circles[0, :]:
        cv2.circle(mask, (i[0], i[1]), i[2], (255,), -1)

# Apply mask
result = cv2.bitwise_and(img, img, mask=mask)

# Display result
cv2.imshow('Circles Only', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
