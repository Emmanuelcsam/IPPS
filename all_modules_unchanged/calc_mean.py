import cv2
import numpy as np

# Load image and calculate mean
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)
mean_value = np.mean(img)
print(f"Mean pixel value: {mean_value:.2f}")