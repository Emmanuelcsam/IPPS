import cv2
import numpy as np

# Configuration
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Step 1: Load and preprocess
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 2: Detect circles
circles = cv2.HoughCircles(
    blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=50,
    param1=50, param2=30, minRadius=10, maxRadius=200
)

if circles is not None:
    circles = np.around(circles).astype(np.uint16)[0, :]
    circles = sorted(circles, key=lambda x: x[2])  # Sort by radius
    
    if len(circles) >= 2:
        # Use detected circles
        core_x, core_y, core_r = circles[0]
        clad_x, clad_y, clad_r = circles[1]
        
        # Create masks
        height, width = gray.shape
        y_indices, x_indices = np.ogrid[:height, :width]
        
        # Core mask
        distance_core = np.sqrt((x_indices - core_x)**2 + (y_indices - core_y)**2)
        core_mask = np.zeros((height, width), dtype=np.uint8)
        core_mask[distance_core <= core_r] = 255
        
        # Cladding mask (annular)
        distance_clad = np.sqrt((x_indices - clad_x)**2 + (y_indices - clad_y)**2)
        cladding_mask = np.zeros((height, width), dtype=np.uint8)
        cladding_mask[(distance_clad > core_r) & (distance_clad <= clad_r)] = 255
        
        # Apply masks
        core_only = cv2.bitwise_and(gray, gray, mask=core_mask)
        cladding_only = cv2.bitwise_and(gray, gray, mask=cladding_mask)
        
        # Draw circles on original
        img_circles = img.copy()
        cv2.circle(img_circles, (core_x, core_y), core_r, (0, 255, 0), 2)
        cv2.circle(img_circles, (clad_x, clad_y), clad_r, (0, 0, 255), 2)
        
        # Save all results
        cv2.imwrite("result_core.jpg", core_only)
        cv2.imwrite("result_cladding.jpg", cladding_only)
        cv2.imwrite("result_circles.jpg", img_circles)
        
        print(f"Core: Center=({core_x}, {core_y}), Radius={core_r}")
        print(f"Cladding: Center=({clad_x}, {clad_y}), Radius={clad_r}")
        print("All results saved!")
    else:
        print(f"Only {len(circles)} circle(s) detected. Need at least 2.")
else:
    print("No circles detected. Try adjusting HoughCircles parameters.")