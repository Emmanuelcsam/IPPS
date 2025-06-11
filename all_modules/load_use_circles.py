import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Load saved circles from detect_circles.py output
try:
    circles = np.load("detected_circles.npy")
    
    if len(circles) >= 1:
        # Use first circle (core)
        x, y, r = circles[0]
        print(f"Using circle: Center=({x}, {y}), Radius={r}")
        
        # Create and apply mask
        height, width = img.shape
        y_indices, x_indices = np.ogrid[:height, :width]
        distance = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[distance <= r] = 255
        
        result = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite("loaded_circle_result.jpg", result)
        print("Result saved!")
    else:
        print("No circles found in saved file")
        
except FileNotFoundError:
    print("No saved circles found. Run detect_circles.py first.")