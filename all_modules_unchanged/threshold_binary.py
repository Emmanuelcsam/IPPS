import cv2
import numpy as np

# Load a scratch strength map (using the grayscale image as example)
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
strength_map = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Compute threshold (mean + 2*std)
mu = np.mean(strength_map)
sigma = np.std(strength_map)
threshold = mu + 2.0 * sigma

# Apply threshold
_, binary = cv2.threshold(strength_map, threshold, 255, cv2.THRESH_BINARY)

# Save result
cv2.imwrite('binary_map.jpg', binary)
print(f"Threshold: {threshold:.2f}, Binary pixels: {np.sum(binary > 0)}")
