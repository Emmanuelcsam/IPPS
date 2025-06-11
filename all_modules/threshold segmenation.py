import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply threshold
gamma = 2.5
mean = np.mean(img)
std = np.std(img)
threshold = mean + gamma * std
_, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

# Display result
cv2.imshow('Thresholded', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()