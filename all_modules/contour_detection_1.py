import cv2
import numpy as np

# Load and prepare image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Save result
cv2.imwrite("contours_output.jpg", img)