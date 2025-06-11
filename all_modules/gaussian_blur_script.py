import cv2
import numpy as np

# Load image in grayscale
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Display both images
    cv2.imshow("Original", img)
    cv2.imshow("Blurred", blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()