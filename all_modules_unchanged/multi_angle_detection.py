import cv2
import numpy as np

# Load and enhance image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
enhanced = cv2.equalizeHist(image)

# Process multiple angles
angles = [0, 45, 90, 135]
h, w = enhanced.shape
combined_result = np.zeros((h, w), dtype=np.uint8)

for angle in angles:
    # Simple edge detection at angle as proxy for scratch detection
    # (Simplified version - in real LEI, use the scratch strength computation)
    kernel_size = 3
    theta = np.radians(angle)
    
    # Create directional kernel
    kernel = np.zeros((kernel_size, kernel_size))
    cx, cy = kernel_size // 2, kernel_size // 2
    kernel[cy, cx] = 2
    kernel[int(cy - np.sin(theta)), int(cx + np.cos(theta))] = -1
    kernel[int(cy + np.sin(theta)), int(cx - np.cos(theta))] = -1
    
    # Apply filter
    filtered = cv2.filter2D(enhanced, -1, kernel)
    _, binary = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY)
    
    # Combine results
    combined_result = cv2.bitwise_or(combined_result, binary)

# Save result
cv2.imwrite('multi_angle_scratches.jpg', combined_result)
print(f"Processed {len(angles)} angles")
