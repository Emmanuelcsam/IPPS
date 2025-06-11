import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path)

# Convert to grayscale to find non-zero pixels
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find all non-zero pixel coordinates
coords = cv2.findNonZero(gray)

if coords is not None:
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(coords)
    
    # Crop image
    cropped = img[y:y+h, x:x+w]
    
    # Save result
    cv2.imwrite("cropped_to_content.png", cropped)
    print(f"Original size: {img.shape[:2]}")
    print(f"Cropped size: {cropped.shape[:2]}")
    print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")
else:
    print("No non-zero content found")