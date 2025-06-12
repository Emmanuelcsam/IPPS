import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create binary image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create output image
output = image.copy()
contour_mask = np.zeros_like(gray)

# Draw and analyze contours
for i, contour in enumerate(contours):
    # Calculate contour properties
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Filter small contours
    if area > 50:
        # Draw contour
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 1)
        
        # Add text with area
        cv2.putText(output, f'A:{int(area)}', (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# Count defects
num_defects = len([c for c in contours if cv2.contourArea(c) > 50])
cv2.putText(output, f'Defects found: {num_defects}', (10, 30), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Save results
cv2.imwrite('contours_detected.jpg', output)
cv2.imwrite('contours_mask.jpg', contour_mask)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Binary', binary)
cv2.imshow('Contours Detected', output)
cv2.imshow('Contour Mask', contour_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()