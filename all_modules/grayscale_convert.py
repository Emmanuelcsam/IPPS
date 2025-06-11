import cv2
import numpy as np

# Convert image to grayscale
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path)

if img is not None:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Display both
    cv2.imshow('Original', img)
    cv2.imshow('Grayscale', gray)
    
    # Optional: save grayscale image
    # cv2.imwrite('grayscale_output.jpg', gray)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Failed to load image from {image_path}")