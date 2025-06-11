import cv2
import numpy as np

# Load image and find maximum value
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)
max_value = np.max(img)
print(f"Maximum pixel value: {max_value}")