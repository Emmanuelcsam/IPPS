import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply different thresholding methods
_, thresh_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)

# Display results
cv2.imshow('Original', gray)
cv2.imshow('Binary Threshold', thresh_binary)
cv2.imshow('Otsu Threshold', thresh_otsu)
cv2.imshow('Adaptive Threshold', thresh_adaptive)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save results
cv2.imwrite('thresh_otsu.png', thresh_otsu)