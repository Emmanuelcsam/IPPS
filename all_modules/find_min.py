import cv2
import numpy as np

# Load image and find minimum value
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)
min_value = np.min(img)
print(f"Minimum pixel value: {min_value}")