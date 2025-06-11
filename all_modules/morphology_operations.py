import cv2
import numpy as np

# Load image and create a binary mask
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create binary image using threshold
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Define kernels for morphological operations
kernel_3x3 = np.ones((3, 3), np.uint8)
kernel_5x5 = np.ones((5, 5), np.uint8)
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Apply morphological operations
erosion = cv2.erode(binary, kernel_3x3, iterations=1)
dilation = cv2.dilate(binary, kernel_3x3, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_5x5)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_5x5)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel_ellipse)

# Remove small noise
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cleaned = np.zeros_like(binary)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:  # Keep only regions larger than 100 pixels
        cv2.drawContours(cleaned, [contour], -1, 255, -1)

# Save results
cv2.imwrite('morph_erosion.jpg', erosion)
cv2.imwrite('morph_dilation.jpg', dilation)
cv2.imwrite('morph_opening.jpg', opening)
cv2.imwrite('morph_closing.jpg', closing)
cv2.imwrite('morph_gradient.jpg', gradient)
cv2.imwrite('morph_cleaned.jpg', cleaned)

# Display results
cv2.imshow('Original Binary', binary)
cv2.imshow('Erosion', erosion)
cv2.imshow('Dilation', dilation)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.imshow('Cleaned', cleaned)
cv2.waitKey(0)
cv2.destroyAllWindows()