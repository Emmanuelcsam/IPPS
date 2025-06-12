import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Enhance
enhanced = cv2.equalizeHist(image)

# Detect scratches using simple edge detection at multiple angles
angles = [0, 45, 90, 135]
result = np.zeros_like(enhanced)

for angle in angles:
    # Rotate image
    center = (enhanced.shape[1] // 2, enhanced.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(enhanced, M, (enhanced.shape[1], enhanced.shape[0]))
    
    # Detect horizontal edges in rotated image
    kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
    edges = cv2.filter2D(rotated, -1, kernel)
    
    # Rotate back
    M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
    edges_back = cv2.warpAffine(edges, M_inv, (enhanced.shape[1], enhanced.shape[0]))
    
    # Threshold and combine
    _, binary = cv2.threshold(edges_back, 127, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_or(result, binary)

# Clean up
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

# Save result
cv2.imwrite('lei_detection_result.jpg', result)
print(f"Detected {np.sum(result > 0)} scratch pixels")
