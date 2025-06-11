import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# Print contour information
print(f"Found {len(contours)} contours")
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 100:  # Only print significant contours
        perimeter = cv2.arcLength(contour, True)
        print(f"Contour {i}: Area={area:.1f}, Perimeter={perimeter:.1f}")

# Display result
cv2.imshow('All Contours', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite('all_contours.png', output)