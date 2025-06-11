import cv2
import numpy as np

# Load image and calculate all stats
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)

print(f"Mean: {np.mean(img):.2f}")
print(f"Std: {np.std(img):.2f}")
print(f"Min: {np.min(img)}")
print(f"Max: {np.max(img)}")
print(f"Pixels: {img.size}")