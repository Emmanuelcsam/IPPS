import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path)

# Convert to grayscale if needed
if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray = image.copy()

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

# Apply CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blurred)

# Apply edge detection for defect boundaries
edges = cv2.Canny(enhanced, 50, 150)

# Save preprocessed images
cv2.imwrite('preprocessed_gray.jpg', gray)
cv2.imwrite('preprocessed_blurred.jpg', blurred)
cv2.imwrite('preprocessed_enhanced.jpg', enhanced)
cv2.imwrite('preprocessed_edges.jpg', edges)

# Display results
cv2.imshow('Original', gray)
cv2.imshow('Blurred', blurred)
cv2.imshow('Enhanced', enhanced)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()