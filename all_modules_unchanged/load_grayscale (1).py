import cv2
import numpy as np

# Load image and convert to grayscale
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Save result
cv2.imwrite('grayscale.jpg', gray)
print(f"Grayscale image shape: {gray.shape}")
