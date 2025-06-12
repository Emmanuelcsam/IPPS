import cv2
import numpy as np

# Load image and mask
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create mask (example for core - adjust as needed)
center_x, center_y, radius = 150, 150, 40
y_indices, x_indices = np.ogrid[:img.shape[0], :img.shape[1]]
distance = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
mask = np.zeros(img.shape, dtype=np.uint8)
mask[distance <= radius] = 255

# Extract pixels within mask
masked_pixels = img[mask == 255]

if len(masked_pixels) > 0:
    mean_val = np.mean(masked_pixels)
    std_val = np.std(masked_pixels)
    min_val = np.min(masked_pixels)
    max_val = np.max(masked_pixels)
    
    print(f"Region statistics:")
    print(f"  Mean intensity: {mean_val:.1f}")
    print(f"  Std deviation: {std_val:.1f}")
    print(f"  Min/Max: {min_val}/{max_val}")
    print(f"  Pixel count: {len(masked_pixels)}")
else:
    print("No pixels found in mask")