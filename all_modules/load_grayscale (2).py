import cv2
import numpy as np

# Load image and convert to grayscale
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

# Save result
cv2.imwrite("grayscale_output.jpg", gray)
print(f"Image shape: {gray.shape}")
print(f"Intensity range: [{gray.min()}, {gray.max()}]")