import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold first
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Define kernel
kernel = np.ones((5, 5), np.uint8)

# Apply morphological operations
morph_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel)
morph_erode = cv2.erode(thresh, kernel, iterations=1)
morph_dilate = cv2.dilate(thresh, kernel, iterations=1)

# Display results
cv2.imshow('Original Threshold', thresh)
cv2.imshow('Morphological Close', morph_close)
cv2.imshow('Morphological Open', morph_open)
cv2.imshow('Erode', morph_erode)
cv2.imshow('Dilate', morph_dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save cleaned result
cv2.imwrite('morphed_image.png', morph_open)