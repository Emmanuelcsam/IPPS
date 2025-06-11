import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Detect scratches at 0 degrees (horizontal)
h, w = img.shape
strength_map = np.zeros((h, w), dtype=np.float32)
line_length = 20
half_len = line_length // 2
offset = 3
angle = 0
theta = np.radians(angle)

for y in range(half_len, h - half_len):
    for x in range(half_len, w - half_len):
        # Center line values
        red_vals = []
        for i in range(-half_len, half_len + 1):
            px = int(x + i * np.cos(theta))
            py = int(y + i * np.sin(theta))
            if 0 <= px < w and 0 <= py < h:
                red_vals.append(float(img[py, px]))
        
        # Parallel lines values
        gray_vals = []
        for sign in [-1, 1]:
            for i in range(-half_len, half_len + 1):
                px = int(x + i * np.cos(theta) + sign * offset * np.sin(theta))
                py = int(y + i * np.sin(theta) - sign * offset * np.cos(theta))
                if 0 <= px < w and 0 <= py < h:
                    gray_vals.append(float(img[py, px]))
        
        if red_vals and gray_vals:
            strength_map[y, x] = max(0, 2 * np.mean(red_vals) - np.mean(gray_vals))

# Normalize and display
strength_map = (strength_map / strength_map.max() * 255).astype(np.uint8)
cv2.imshow('Scratch Strength (0 degrees)', strength_map)
cv2.waitKey(0)
cv2.destroyAllWindows()