import cv2
import numpy as np

# Load image in grayscale
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Apply threshold
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Apply adaptive threshold
    adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Display results
    cv2.imshow("Original", img)
    cv2.imshow("Binary Threshold", binary)
    cv2.imshow("Adaptive Threshold", adaptive)
    cv2.waitKey(0)
    cv2.destroyAllWindows()