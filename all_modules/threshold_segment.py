import cv2
import numpy as np

# Load grayscale image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Calculate statistics
mu = np.mean(image)
sigma = np.std(image)

# Apply threshold
gamma = 2.5
threshold = mu + gamma * sigma
_, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

# Save result
cv2.imwrite('7_thresholded.jpg', binary)
print(f"Threshold value: {threshold:.2f}")
