import cv2
import numpy as np

# Load example binary maps (using same image for demo)
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
binary1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
_, binary1 = cv2.threshold(binary1, 127, 255, cv2.THRESH_BINARY)

# Create second binary map with different threshold
binary2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
_, binary2 = cv2.threshold(binary2, 100, 255, cv2.THRESH_BINARY)

# Combine using bitwise OR
result = cv2.bitwise_or(binary1, binary2)

# Save result
cv2.imwrite('combined_binary.jpg', result)
print("Binary maps combined")
