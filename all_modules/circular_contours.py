import cv2
import numpy as np

# Load and prepare image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find circular contours
for c in contours:
    area = cv2.contourArea(c)
    if area > 100:
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            roundness = 4 * np.pi * area / (perimeter ** 2)
            if roundness > 0.7:  # circular shape
                (x, y), r = cv2.minEnclosingCircle(c)
                cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)

# Save result
cv2.imwrite("circular_contours_output.jpg", img)