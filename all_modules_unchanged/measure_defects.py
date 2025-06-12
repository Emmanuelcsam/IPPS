import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create binary defect mask (adjust threshold as needed)
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# Clean up with morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Create measurement output
output = image.copy()
measurements = np.zeros_like(gray)

# Analyze each component (skip background at index 0)
for i in range(1, num_labels):
    # Get component stats
    x, y, w, h, area = stats[i]
    cx, cy = centroids[i]
    
    # Filter small components
    if area > 20:
        # Calculate additional properties
        component_mask = (labels == i).astype(np.uint8) * 255
        contour, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contour:
            # Circularity
            perimeter = cv2.arcLength(contour[0], True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Draw bounding box
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 1)
            
            # Draw centroid
            cv2.circle(output, (int(cx), int(cy)), 3, (255, 0, 0), -1)
            
            # Add measurements as text
            cv2.putText(output, f'#{i}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(measurements, f'Area: {area}', (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1)
            cv2.putText(measurements, f'AR: {aspect_ratio:.2f}', (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1)

# Create summary
total_defect_area = np.sum(binary > 0)
defect_percentage = (total_defect_area / (gray.shape[0] * gray.shape[1])) * 100
cv2.putText(output, f'Total defects: {num_labels-1}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
cv2.putText(output, f'Defect coverage: {defect_percentage:.2f}%', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Save results
cv2.imwrite('defect_measurements.jpg', output)
cv2.imwrite('defect_labels.jpg', (labels * 255 / num_labels).astype(np.uint8))
cv2.imwrite('measurement_text.jpg', measurements)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Binary Defects', binary)
cv2.imshow('Defect Measurements', output)
cv2.waitKey(0)
cv2.destroyAllWindows()