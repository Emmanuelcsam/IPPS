import cv2
import numpy as np

# Load enhanced image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
image = cv2.equalizeHist(image)

# Parameters
angle = 45  # degrees
line_length = 20
offset = 3

# Compute scratch strength at given angle
h, w = image.shape
strength_map = np.zeros((h, w), dtype=np.float32)
theta = np.radians(angle)
half_len = line_length // 2

for y in range(half_len, h - half_len):
    for x in range(half_len, w - half_len):
        # Red branch (center line)
        red_vals = []
        for i in range(-half_len, half_len + 1):
            px = int(x + i * np.cos(theta))
            py = int(y + i * np.sin(theta))
            if 0 <= px < w and 0 <= py < h:
                red_vals.append(image[py, px])
        
        # Gray branches (parallel lines)
        gray_vals = []
        for sign in [-1, 1]:
            for i in range(-half_len, half_len + 1):
                px = int(x + i * np.cos(theta) + sign * offset * np.sin(theta))
                py = int(y + i * np.sin(theta) - sign * offset * np.cos(theta))
                if 0 <= px < w and 0 <= py < h:
                    gray_vals.append(image[py, px])
        
        if red_vals and gray_vals:
            red_avg = np.mean(red_vals)
            gray_avg = np.mean(gray_vals)
            strength_map[y, x] = max(0, 2 * red_avg - gray_avg)

# Normalize and save
strength_norm = (strength_map / strength_map.max() * 255).astype(np.uint8)
cv2.imwrite(f'strength_map_{angle}deg.jpg', strength_norm)
print(f"Scratch strength computed at {angle}°")
