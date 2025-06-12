import cv2
import numpy as np

# Load image and calculate standard deviation
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)
std_value = np.std(img)
print(f"Standard deviation: {std_value:.2f}")